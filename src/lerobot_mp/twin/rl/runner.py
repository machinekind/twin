"""Polityka na zywym blizniaku - na symulacji albo na prawdziwym ramieniu.

    runner = PolicyRunner(twin, Policy.load(path))
    runner.goal = np.array([0.25, 0.05, 0.10])      # reach: cel w ukladzie podstawy
    runner.start()                                  # watek z czestotliwoscia polityki
    ...
    runner.stop()                                   # ramie zostaje tam, gdzie stoi

Petla robi dokladnie to, co srodowisko w treningu - obserwacja z `task.observe`,
akcja przez `task.apply_action` - tylko katy nie pochodza z fizyki, a z serw
(`Twin.status.measured`). Cel stawow NIE idzie do serw wprost: trafia do
`Twin.set_target`, a stamtad przez `SafetySupervisor` (limity pozycji
i predkosci, STOP), tak samo jak kazdy inny rozkaz w tej aplikacji.

Polityka bierze ramie na wlasnosc (`Twin.claim("polityka")`): nie ruszy, gdy
jedzie fala albo identyfikacja, a Dom/STOP/ponowne polaczenie ja zatrzymuja.

Straznicy, ktorych srodowisko treningowe nie ma, a prawdziwe ramie potrzebuje:

* **stop przy rozjezdzie** - gdy zmierzona poza odjezdza od celu o wiecej niz
  `max_tracking_deg` (staw zablokowany, kolizja, serwo nie nadaza), polityka
  sie zatrzymuje, a ramie staje w ZMIERZONEJ pozie - nie dociska dalej;
* **limit czasu** - epizod na ramieniu trwa tyle, ile w treningu;
* **koniec po sukcesie** - `end_on_success` taktow sukcesu z rzedu konczy
  epizod (lift: kostka nad blatem), zamiast pozwalac polityce krecic ramieniem
  z kostka do limitow stawow przez reszte epizodu; sukces lift liczy sie tylko
  z pozy swiezej (kamery albo "w dloni"), nigdy z "ostatnie widziane", i tylko
  gdy szczeka trzyma (stoi, ciasniej rozkazana, nie domyka sie w trakcie serii);
* **cel reach w obszarze treningu** - cel spoza niego (pod blatem, za
  podstawa) jest rzutowany na obszar, z ktorego losowano cele w treningu;
* **pickplace** - cel odlozenia to cel z panelu rzutowany na blat (z = 0) i na
  pierscien celow z treningu (`project_target`). Sukces liczy sie tylko z kostki
  WIDZIANEJ lezacej (kamery, nie "w dloni", nie "ostatnie widziane"), na celu,
  puszczonej (szczeka nie trzyma, rozkaz chwytaka otwarty), nieruchomej w trakcie
  serii i tylko, gdy wczesniej byla niesiona (zatrzask jak w treningu) - kostka
  dopchnieta po blacie sie nie liczy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import task as tk
from .policy import Policy, validate_cube_height_error_bound

logger = logging.getLogger(__name__)

#: Nazwa wlasciciela ramienia (`Twin.claim`).
OWNER = "polityka"

#: Zrodla pozy kostki (`CubeTracker.source`), z ktorych wolno liczyc sukces lift: swieza
#: detekcja z kamer, kostka niesiona w szczekach albo prawda symulacji.
FRESH_CUBE_SOURCES = ("kamery", "w dloni", "symulacja")
#: Zrodla, z ktorych wolno liczyc sukces pickplace: kostka WIDZIANA lezaca (nie "w dloni" -
#: tam poza idzie za szczekami) albo prawda symulacji.
RESTING_CUBE_SOURCES = ("kamery", "symulacja")
#: pickplace: w trakcie serii sukcesu kostka nie przesuwa sie o wiecej niz tyle [m].
PLACE_STILL = 0.005


class NoCube(RuntimeError):
    """Zadanie z kostka, a percepcja nie wie, gdzie ona jest."""


@dataclass
class RunnerStatus:
    running: bool = False
    step: int = 0
    hz: float = 0.0
    distance: float = float("nan")
    success: bool = False
    stopped_because: str = ""
    last_action: list[float] = field(default_factory=list)
    #: Ile taktow z rzedu zadanie jest wykonane.
    success_streak: int = 0
    #: Elapsed time since the first continuously successful observation.
    success_hold_s: float = 0.0
    #: reach: cel przyciety do obszaru treningu (panel pokazuje, ze kula stoi gdzie indziej).
    #: pickplace: cel odlozenia przesuniety na pierscien celow z treningu.
    goal_clamped: bool = False
    #: pickplace: kostka byla niesiona w tym epizodzie (zatrzask).
    carried: bool = False


def project_goal(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """Najblizszy punkt obszaru, z ktorego `sample_goals` losowal cele reach.

    Promien od osi podstawy w `goal_radius`, wysokosc w `goal_height`, x > 0,05 m.
    Polityka poza nim nigdy nie byla - cel pod blatem prowadzil szczeki w blat.
    """
    g = np.asarray(goal, float).copy()
    g[2] = np.clip(g[2], *task.goal_height)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.goal_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    # x > 0,05 przy promieniu r to kat od osi x najwyzej arccos(0,05 / r); margines 1 mm.
    b_max = float(np.arccos(min(1.0, 0.051 / r_new)))
    bearing = float(np.clip(bearing, -b_max, b_max))
    g[0], g[1] = r_new * np.cos(bearing), r_new * np.sin(bearing)
    return g


def project_target(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """Cel odlozenia pickplace z punktu `goal` (uklad podstawy): na blat (z = 0) i na pierscien celow.

    Promien od osi podstawy w `target_radius`, kierunek w `target_bearing` - tam
    `sample_targets` losowal cele w treningu; poza nim polityka nigdy nie odkladala.
    """
    g = np.asarray(goal, float)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.target_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    bearing = float(np.clip(bearing, *task.target_bearing))
    return np.array([r_new * np.cos(bearing), r_new * np.sin(bearing), 0.0])


class PolicyRunner:
    def __init__(self, twin, policy: Policy, *, max_tracking_deg: float = 25.0,
                 cube_provider: Callable[[], tuple | None] | None = None,
                 episode_limit: bool = True, cube_source: Callable[[], str] | None = None):
        from ..kinematics import RobotKinematics

        spec = twin.workspace.spec()
        robot = getattr(policy.meta, "robot", "so101")
        if robot != spec.name:
            raise RuntimeError(f"polityka uczona na {robot}, stanowisko ma {spec.name}")
        if policy.meta.act_dim != len(spec.joints):
            raise RuntimeError(f"polityka ma {policy.meta.act_dim} wyjsc, a ramie {spec.name} "
                               f"{len(spec.joints)} stawow")
        self.twin = twin
        self.policy = policy
        self.task = policy.task
        self.cube_height_error_bound_m = validate_cube_height_error_bound(
            getattr(policy.meta, "cube_height_error_bound_m", 0.0), self.task.name)
        self._cube_yaw = tk.cube_yaw_tracker(getattr(policy.meta, "cube_yaw_mode", "canonical"),
            task=self.task, robot=spec.name, fold=bool(policy.meta.randomization.get("fold_yaw", False)))
        # Wlasna kinematyka (wlasne MjData): ta ze sceny liczy tez panel (IK uchwytu)
        # z innego watku, a `tcp()` pisze do swojego MjData.
        self.kin = RobotKinematics(spec)
        self.limits = tk.Limits.of(self.kin)
        #: Indeks chwytaka i liczba stawow ramienia (chwytak ostatni, `Limits.of`).
        self.g, self.na = self.limits.grip, self.limits.n_arm
        self.status = RunnerStatus()
        self._goal = np.array([0.22, 0.0, 0.12])
        #: lift: (polozenie (3,), obrot (3, 3)) kostki w ukladzie podstawy - z wizji albo z symulacji.
        #: Moze zwrocic trzeci element: zrodlo pozy (`CubeTracker.source`), patrz `FRESH_CUBE_SOURCES`.
        self.cube_provider = cube_provider
        #: Zrodlo pozy kostki, gdy dostawca zwraca tylko (polozenie, obrot) - np.
        #: `lambda: tracker.source`. Bez obu na prawdziwym ramieniu sukces lift sie nie liczy.
        self.cube_source = cube_source
        self._cube_src: str | None = None
        self.max_tracking = np.radians(max_tracking_deg)
        self.episode_limit = episode_limit
        #: Ile sekund ramie stoi, czekajac na pierwsza pozycje kostki z kamer.
        self.wait_for_cube = 3.0
        end = getattr(self.task, "end_on_success", None)
        #: Po ilu taktach sukcesu z rzedu epizod sie konczy (0 = nigdy). Z zadania (K8);
        #: polityki zapisane przed tym polem - lift 10, reach 0 (cel przeciagany na zywo).
        self.end_on_success = int(end) if end is not None else (10 if self.task.cube else 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.q_cmd = self.limits.home.copy()
        self.prev_action = np.zeros(self.task.act_dim)
        self._hw_lo, self._hw_hi = self.limits.lo, self.limits.hi
        self._t_start: float | None = None
        self._t_hz, self._ticks = 0.0, 0
        self._cube: np.ndarray | None = None
        #: Szczeka "trzyma" (sukces lift): jak w `CubeTracker` - stoi (< `grip_still` miedzy
        #: taktami), szerzej niz rozkaz o `grip_block_margin` i niz zamknieta o `grip_open_margin`.
        #: Jednostki MJCF chwytaka (`spec.jaw`; SO-101: rad).
        jaw = spec.jaw
        self.grip_still, self.grip_block_margin, self.grip_open_margin = jaw.still, jaw.margin, jaw.margin
        #: W trakcie serii sukcesu szczeka nie domyka sie o wiecej niz tyle (SO-101: 0,03 rad ~ 3 jednostki).
        self.grip_creep = jaw.creep
        self._grip_prev: float | None = None
        self._streak_grip = 0.0
        self._success_since: float | None = None
        #: pickplace: kostka byla niesiona (jak `task.carry_latch` w treningu) i gdzie byla na poczatku serii.
        self._carried = False
        self._streak_cube: np.ndarray | None = None

    # ------------------------------------------------------------------ cel
    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value) -> None:
        g = np.asarray(value, float)
        if self.task.name == "reach":
            p = project_goal(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p - g) > 1e-4)
            g = p
        elif self.task.target:
            # Cel odlozenia lezy na blacie - z = 0 to nie przyciecie, tylko definicja.
            p = project_target(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p[:2] - g[:2]) > 1e-4)
            g = p
        self._goal = g

    @property
    def target(self) -> np.ndarray:
        """pickplace: cel odlozenia (uklad podstawy, z = 0) - `goal` rzutowany przez `project_target`."""
        return project_target(self.task, self._goal)

    # ------------------------------------------------------------ obserwacja
    def observation(self, measured: dict[str, float]) -> np.ndarray:
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        target = None
        if self.task.cube:
            got = self.cube_provider() if self.cube_provider else None
            if got is None:
                raise NoCube(f"zadanie {self.task.name} potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = got[0], got[1]
            self._cube_src = str(got[2]) if len(got) > 2 else (self.cube_source() if self.cube_source else None)
            if not self.twin.status.simulated and self._cube_src not in ("kamery", "w dloni"):
                raise NoCube("Physical policy requires a current camera or held-object estimate; "
                             f"received {self._cube_src!r}")
            self._cube = np.asarray(cube_pos, float).copy()
            if self.policy.meta.randomization.get("fold_yaw", False):
                # W treningu polityka widziala obrot zlozony do +-45 st. (jak z detektora);
                # poza "w dloni" i prawda z symulacji tego nie maja - skladamy tak samo.
                rotation = np.asarray(cube_rot, float)
                cube_rot = (self._cube_yaw(rotation) if self._cube_yaw is not None
                            else tk.fold_yaw(np, rotation))
        if self.task.target:
            target = self.target[None]
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], tcp[None], self.prev_action[None],
                         goal=self.goal[None], cube_pos=cube_pos[None], cube_rot=cube_rot[None], target=target)
        return obs[0].astype(np.float32)

    # ------------------------------------------------------------------ petla
    def start(self, *, threaded: bool = True, gen: int | None = None,
              guard: Callable[[], bool] | None = None) -> None:
        """Bierze ramie i rusza. `threaded=False`: bez watku - takty robi `step_once`.

        `gen` (`Twin.preempt_gen` sprzed ladowania polityki) i `guard` (np. licznik ruchu
        panelu) ida do `Twin.claim`: STOP albo Dom wcisniete w trakcie startu (108-142 ms
        ladowania) nie moga zostac nadpisane przez polityke, ktora dopiero bierze ramie.
        """
        if not self.twin.connected:
            raise RuntimeError("ramie blizniaka nie jest polaczone")
        if not self.twin.status.simulated and not self.twin.gripper_measured():
            raise RuntimeError("Physical policies require calibrated gripper feedback; "
                               "a commanded opening is not a measured grasp")
        self.stop()
        # Najpierw wlasnosc: fala albo identyfikacja w toku, aktywny STOP albo odebranie
        # w trakcie startu -> RuntimeError, nic nie rusza.
        kw = {k: v for k, v in (("gen", gen), ("guard", guard)) if v is not None}
        self.twin.claim(OWNER, preempt=self._preempted, **kw)
        measured = self.twin.joints() if self.twin.status.simulated else dict(self.twin.status.measured)
        self._hw_lo, self._hw_hi = self._hardware_limits()
        self.q_cmd = np.clip(self.kin.to_q(measured), self._hw_lo, self._hw_hi)
        grip = self._squeezing_grip_cmd(measured)
        if grip is not None:
            self.q_cmd[self.g] = grip
        self.prev_action = np.zeros(self.task.act_dim)
        self.status = RunnerStatus(running=True, goal_clamped=self.status.goal_clamped)
        self._cube, self._cube_src = None, None
        if self._cube_yaw is not None:
            self._cube_yaw.reset()
        self._grip_prev = None
        self._carried, self._streak_cube = False, None
        self._success_since = None
        self._stop.clear()
        self._t_start, self._ticks = None, 0            # zegar startuje z pierwszym taktem
        # Sprzeglo tylko, jesli ramie wciaz nasze - sprawdzenie i wlaczenie razem w `Twin`.
        # Wczesniej STOP miedzy claim a sprzeglem, a potem panel bioracy ramie: runner
        # wylaczal sprzeglo panelu, a pole wyboru w panelu dalej pokazywalo "wlaczone".
        if not self.twin.set_engaged(True, owner=OWNER):
            reason = getattr(self.twin, "preempt_reason", "") or f"ramie ma: {self.twin.owner or 'nikt'}"
            self.status.running = False
            self.status.stopped_because = f"przerwana: {reason}"
            raise RuntimeError(f"ramie odebrane ({reason})")
        if threaded:
            # Watek w zmiennej lokalnej: odebranie ramienia miedzy utworzeniem a startem
            # watku (Dom z panelu) zeruje `self._thread` w `stop()`.
            th = threading.Thread(target=self._loop, name="polityka", daemon=True)
            self._thread = th
            th.start()

    def _squeezing_grip_cmd(self, measured: dict[str, float]) -> float | None:
        """Kat rozkazu chwytaka [rad] z `Twin.status.command`, gdy chwytak SCISKA (rozkaz ciasniej niz pomiar).

        Start od zmierzonego kata szczeki to zerowa sila: ponowne Uruchom z kostka w szczekach
        (po STOP, Dom, Zatrzymaj) puszczalo chwyt pierwsza akcja, a `_jaw_holding` nigdy nie
        widzial szczeki "szerzej niz rozkaz" - sukces lift nie mogl sie policzyc. W treningu
        q_cmd trzymanej kostki tez jest ciasniejszy od pomiaru. Inne stawy - jak dotad, od pomiaru.
        """
        name = self.kin.spec.gripper
        cmd = (getattr(self.twin.status, "command", None) or {}).get(name)
        if cmd is None or name not in measured:
            return None
        g = self.g
        q_cmd = float(self.kin.to_q({**measured, name: float(cmd)})[g])
        if float(self.kin.to_q(measured)[g]) - q_cmd <= self.grip_block_margin:
            return None
        return float(np.clip(q_cmd, self._hw_lo[g], self._hw_hi[g]))

    def _hardware_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Limity treningu przeciete z limitami, ktore naprawde obowiazuja na ramieniu (`Twin.joint_limits`).

        Normalizacja obserwacji zostaje na limitach treningu - zmiana skali
        bylaby dla polityki inna obserwacja. Przycinamy tylko cel: inaczej
        q_cmd stal do 7 st. za limitem serwa (ramie nr 2, wrist_flex +88).
        """
        lo, hi = self.limits.lo.copy(), self.limits.hi.copy()
        fn = getattr(self.twin, "joint_limits", None)
        hw = fn() if fn is not None else {}
        for k, name in enumerate(self.kin.spec.joints):
            if name in hw:
                a, b = self.kin.to_q({name: hw[name][0]})[k], self.kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = self.limits.lo[bad], self.limits.hi[bad]
        return lo, hi

    def _preempted(self) -> None:
        """Twin odebral ramie (Dom, STOP, polaczenie, inny wlasciciel) - tylko zatrzymanie watku."""
        self.stop(f"przerwana: {getattr(self.twin, 'preempt_reason', '') or 'ramie odebrane'}")

    def stop(self, reason: str = "") -> None:
        self._stop.set()
        th = self._thread
        # Watek jeszcze nie wystartowal (odebranie tuz po `Thread()`): join rzucal RuntimeError,
        # `status.running` zostawal True i panel widzial "polityka" do recznego Stop.
        # Taki watek po starcie zobaczy `_stop` i od razu wyjdzie.
        if th is not None and th is not threading.current_thread() and th.is_alive():
            th.join(timeout=2.0)
        self._thread = None
        if self.status.running:
            self.status.running = False
            self.status.stopped_because = reason or "zatrzymana"
            self._let_go(hold_measured=False)             # ramie trzyma ostatni cel

    def _loop(self) -> None:
        period = 1.0 / self.task.control_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self.step_once(t0):
                return
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def step_once(self, now: float | None = None) -> bool:
        """Jeden takt polityki. False = polityka skonczyla (powod w `status.stopped_because`).

        `now` - czas taktu [s]; watek podaje zegar, test - czas symulowany
        (wtedy `wait_for_cube` i czestotliwosc licza sie w tym samym czasie).
        """
        if not self.status.running:
            return False
        now = time.monotonic() if now is None else now
        if self._t_start is None:
            self._t_start = self._t_hz = now
        try:
            if not self.twin.connected:
                self._halt("ramie rozlaczone")
                return False
            measured = dict(self.twin.status.measured) or self.twin.joints()
            q = self.kin.to_q(measured)
            lag = np.abs(q[:self.na] - self.q_cmd[:self.na]).max()
            if lag > self.max_tracking:
                self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                           hold_measured=True)
                return False
            holding = self._jaw_holding(q[self.g])
            obs = self.observation(measured)
            action = self.policy.act(obs)
            q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], action[None])[0]
            self.q_cmd = np.clip(q_cmd, self._hw_lo, self._hw_hi)
            self.twin.set_target(self.kin.from_q(self.q_cmd), owner=OWNER)
            self.prev_action = action
            st = self.status
            st.step += 1
            st.stopped_because = ""
            st.last_action = action.tolist()
            if self.task.name == "reach":
                st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
                st.success = st.distance < self.task.success_dist
            elif self.task.target:
                st.success = self._placed(holding)
                st.carried = self._carried
            else:
                # Na ramieniu nie ma czujnikow kontaktu szczek - sukcesem jest kostka
                # (z kamer albo "w dloni") wyzej nad blatem niz `lift_height`, i to tylko,
                # gdy szczeka ja TRZYMA (stoi, ciasniej rozkazana niz jest). Zmierzone
                # (blizniak, 1 z 18 sukcesow): seria 1 s przy 10,9 cm, a kostka obracala
                # sie w szczekach, szczeka domykala sie 31,8 -> 0 przez 3,5 s i kostka spadla
                # 4 s po "zadanie wykonane".
                # Szczeka "stoi" z taktu na takt nawet przy powolnym domykaniu (9 jednostek/s
                # to 0,005 rad na takt) - dlatego tez: w trakcie serii nie domyka sie o wiecej
                # niz `grip_creep` od jej poczatku.
                if holding and st.success_streak > 0 and q[self.g] < self._streak_grip - self.grip_creep:
                    holding = False
                st.success = (self._cube is not None and self._cube_fresh() and holding
                              and self._cube[2] - self.task.cube_half
                              > self.task.lift_height + self.cube_height_error_bound_m)
                if st.success and st.success_streak == 0:
                    self._streak_grip = float(q[self.g])
            st.success_streak = st.success_streak + 1 if st.success else 0
            if st.success:
                if self._success_since is None:
                    self._success_since = now
                st.success_hold_s = max(0.0, now - self._success_since)
            else:
                self._success_since = None
                st.success_hold_s = 0.0
            self._ticks += 1
            if now - self._t_hz >= 1.0:
                st.hz, self._ticks, self._t_hz = self._ticks / (now - self._t_hz), 0, now
            # N observations span only N-1 intervals. Count alone previously
            # finished a 2 s hold at 1.95 s, and faster calls could shorten it
            # further. Require both consecutive observations and elapsed time.
            if (self.end_on_success > 0 and st.success_streak >= self.end_on_success
                    and st.success_hold_s >= self.end_on_success / self.task.control_hz - 1e-9):
                self._halt("zadanie wykonane")
                return False
            if self.episode_limit and st.step >= self.task.episode_steps:
                self._halt("koniec epizodu")
                return False
        except NoCube as exc:
            # Na starcie percepcja moze jeszcze nie miec pierwszej detekcji - ramie
            # stoi i czekamy; zgubiona kostka W TRAKCIE epizodu zatrzymuje polityke.
            if self.status.step == 0 and now - self._t_start < self.wait_for_cube:
                self.status.stopped_because = "czekam na kostke z kamer"
                return True
            self._halt(str(exc), hold_measured=True)
            return False
        except Exception as exc:                        # ramie sie rozlaczylo, ramie odebrane, blad percepcji...
            if self.twin.owner != OWNER and self._stop.is_set():
                return False                            # zatrzymana przez `stop` w trakcie taktu
            logger.exception("Petla polityki przerwana")
            self._halt(f"{type(exc).__name__}: {exc}", hold_measured=True)
            return False
        return True

    def _placed(self, holding: bool) -> bool:
        """pickplace: sukces tego taktu (i aktualizacja zatrzasku "niesiona", `status.distance`).

        Kostka widziana lezaca (`RESTING_CUBE_SOURCES`) na celu (xy blizej niz `place_radius`,
        spod nizej niz `place_height`), szczeka jej nie trzyma, rozkaz chwytaka otwarty,
        wczesniej byla niesiona, a od poczatku serii nie przesunela sie o `PLACE_STILL`.
        Zatrzask: prawda symulacji albo "w dloni" z kamer wyzej nad blatem niz `lift_height`.
        """
        cube, src, task = self._cube, self._cube_src, self.task
        if cube is None:
            return False
        sim = bool(getattr(self.twin.status, "simulated", False))
        if src is None and sim:
            src = "symulacja"                               # dostawca bez zrodla w sim = prawda sceny
        height = float(cube[2]) - task.cube_half
        tgt = self.target
        d_t = float(np.hypot(*(cube[:2] - tgt[:2])))
        self.status.distance = d_t
        carried_before = self._carried
        if src in ("symulacja", "w dloni") and height > task.lift_height:
            self._carried = True
        grip_open = bool(tk.grip_is_open(np, self.limits, self.q_cmd[None])[0])
        ok = (src in RESTING_CUBE_SOURCES and carried_before and d_t < task.place_radius
              and height < task.place_height and not holding and grip_open)
        if ok and self.status.success_streak > 0 and self._streak_cube is not None \
                and np.linalg.norm(cube - self._streak_cube) >= PLACE_STILL:
            ok = False                                      # kostka jeszcze sie toczy / przesuwa
        if ok and (self.status.success_streak == 0 or self._streak_cube is None):
            self._streak_cube = np.asarray(cube, float).copy()
        return ok

    def _jaw_holding(self, grip_q: float) -> bool:
        """Czy szczeka trzyma: stoi od poprzedniego taktu, jest szerzej niz rozkaz i niz pusta, zamknieta.

        Te same progi, co `CubeTracker` ("w dloni"). Rozkaz z POPRZEDNIEGO taktu - za nim
        serwo wlasnie jedzie. Szczeka, ktora sie domyka (kostka wypada, obraca sie), nie stoi.
        """
        prev, self._grip_prev = self._grip_prev, float(grip_q)
        still = prev is not None and abs(grip_q - prev) < self.grip_still
        squeezing = (grip_q - self.q_cmd[self.g] > self.grip_block_margin
                     and grip_q > self.limits.lo[self.g] + self.grip_open_margin)
        return bool(still and squeezing)

    def _cube_fresh(self) -> bool:
        """Czy poza kostki jest swieza - tylko z takiej wolno liczyc sukces lift.

        Tracker trzyma ostatnia poze do 6 s jako "ostatnie widziane". Zmierzone w
        blizniaku (kamera a): kostka wypadla ze szczek po 0,5 s niesienia, a runner
        liczyl sukces ze starej pozy w powietrzu i konczyl "zadanie wykonane" z kostka
        na blacie. Bez zrodla: symulacja (dostawca = prawda sceny) tak, prawdziwe ramie nie.
        """
        if self._cube_src is None:
            return bool(getattr(self.twin.status, "simulated", False))
        return self._cube_src in FRESH_CUBE_SOURCES

    def _halt(self, reason: str, hold_measured: bool = False) -> None:
        self.status.running = False
        self.status.stopped_because = reason
        self._let_go(hold_measured)
        self._stop.set()
        logger.info("Polityka zatrzymana: %s", reason)

    def _let_go(self, hold_measured: bool) -> None:
        """Oddaje ramie - tylko jesli wciaz je mamy: po odebraniu nalezy juz do kogos innego.

        `hold_measured` (rozjazd, kolizja): ramie staje tam, gdzie JEST. Ostatni cel
        lezy wtedy ~25 st. za przeszkoda i trzymanie go dociskalo serwa do niej
        z pelnym momentem. Zwykly koniec trzyma cel - szczeki dalej sciskaja kostke.
        """
        if self.twin.owner != OWNER:
            return
        if hold_measured:
            self.twin.hold_measured()
        self.twin.set_engaged(False, owner=OWNER)
        self.twin.release(OWNER)
