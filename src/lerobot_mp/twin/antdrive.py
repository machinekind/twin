"""Wlasny sterownik mrowki udajacej prawdziwa (demo): WSAD -> kierunek chodu, ta sama polityka, 20 Hz.

    drive = AntDrive(twin, ant_getter)                  # mrowka `antprawda/` w scenie symulowanej
    drive.start()                                       # watek; w testach: drive.tick() krok po kroku
    drive.keys("w")                                     # pulpit WSAD (`antpad`) - co 0,1 s, puszczenie = ""
    drive.stop("STOP w panelu")                         # stoi w pozie startowej, klawisze ignorowane do clear()

Prawdziwa mrowka jezdzi na wlasnej polityce - realant-rl `robot_teleop.py` po jej stronie - i blizniak jej
niczego nie wysyla (rozkazy z dwoch miejsc naraz sie gryzly). Blizniak ma jej kopie w fizyce, napedzana
odczytami serw i dociagana tagami (`antmirror`). W demo tej prawdziwej mrowki nie ma, wiec udaje ja druga
mrowka w scenie (`scene.STANDIN_PREFIX`), a ten modul to jej `robot_teleop.py`, przeniesiony wiernie (20a6d0d):

**WSAD nie zmienia polityki.** Mrowka jest symetryczna co 90 st., wiec chod w bok albo w tyl to ta sama
polityka "do przodu" z przemianowanymi nogami (adresowanie serw, `antwalk.DIRECTIONS`) i z obserwacja
obrocona o ten sam kat - polityka widzi "swoj" przod w kierunku jazdy. Q / E = obrot w miejscu (otwarta petla,
skalibrowana na robocie). Dwa klawisze naraz - pierwszy. Przejscia, regulator wysokosci, czuwak jak w oryginale.

Kierunki (`frame`): "widok" (domyslnie) - W = od patrzacego w glab widoku 3D panelu, S = do niego, A / D = w lewo /
w prawo ekranu; przy wcisnieciu wybierany jest ten z czterech kierunkow mrowki, ktory lezy najblizej
(`view_dir`), i trzymany do puszczenia klawisza. "mrowka" - W / S / A / D = przod / tyl / lewo / prawo mrowki
(realant-rl: przod miedzy nogami 2 i 3).

Poza w obserwacji polityki (`camera`): "nominal" (domyslnie, jak `robot_teleop.py` - idealny przejazd, mrowka
nie ma kamery), "prawda" (z fizyki) albo "tagi" (kamery stanowiska - tryb `live` z robot_policy.py).

Bezpieczenstwo: klawisze to czuwak - brak odswiezenia przez `DEADMAN` = puszczone; STOP (`stop`) trzyma poze
startowa i ignoruje klawisze do `clear`; przewrocona (os z mrowki > 50 st. od pionu) = STOP.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import antwalk as aw
from .ant import Ant, locate
from .scene import STANDIN_PREFIX

#: Klawisz -> tryb: W S A D - ta sama polityka w czterech kierunkach, Q E - obrot w miejscu (robot_teleop.py).
KEYS = {"w": "przod", "s": "tyl", "a": "lewo", "d": "prawo", "q": "obrot_lewo", "e": "obrot_prawo"}
#: Bez odswiezenia klawiszy przez tyle sekund = puszczone (pulpit sle stan co 0,1 s; robot_teleop.py: 0,5 s).
DEADMAN = 0.4
#: Poza z kamer starsza niz tyle sekund = brak pozy: polityka nie rusza z miejsca.
POSE_TIMEOUT = 0.5
#: Kroki rampy przez poze startowa (0,5 s, robot_teleop.py --ramp_steps).
RAMP_STEPS = 10
#: Os z mrowki dalej od pionu = przewrocona -> STOP.
FALL_TILT_DEG = 50.0
#: Wysokosc korpusu w "kamerze nominalnej" [m] (robot_policy.py --nominal_z: 5,7 cm przeswitu + plyta).
NOMINAL_Z = 0.081


# ------------------------------------------------------------------ mrowka w symulacji
class SimAnt:
    """Mrowka w scenie blizniaka (`AntPlacement.free`): stawy i serwa nog, fizyka sceny.

    Fizyka sceny idzie naprzod w petli ramienia tylko przy polaczonym backendzie `sim`. `advance` dokracza
    scene, gdy nikt inny tego nie robi (czas symulacji stoi), a przy polaczonym ramieniu tylko ustawia cel -
    czas nie biegnie podwojnie. Rozkaz wchodzi po 20 ms kroku, jak w treningu (`act_delay_substeps`).
    """

    kind = "sim"

    def __init__(self, twin, prefix: str = STANDIN_PREFIX):
        self.twin = twin
        self.prefix = prefix
        self._pending: np.ndarray | None = None
        self._t_seen: float | None = None
        self._q_prev: np.ndarray | None = None
        self._qd = np.zeros(8)

    def now(self) -> float:
        """Czas mrowki: czas symulacji (kadry kamer symulowanych sa z tej samej chwili)."""
        with self.twin.lock:
            return float(self.twin.scene.data.time)

    def ready(self) -> str:
        with self.twin.lock:
            ants = self.twin.scene.ants()
        if self.prefix not in ants:
            return "w scenie nie ma mrowki udajacej prawdziwa (lerobot-twin demo --ant)"
        return ""

    def startup(self) -> None:
        self._t_seen, self._q_prev = None, None

    def command(self, action: np.ndarray) -> None:
        self._pending = aw.action_to_ctrl(action)

    def advance(self) -> None:
        with self.twin.lock:
            s = self.twin.scene
            t = float(s.data.time)
            stalled = self._t_seen is None or t - self._t_seen < 0.5 * aw.DT
            if stalled:
                s.step(0.02)
                if self._pending is not None:
                    s.command_ant(self._pending, self.prefix)
                s.step(aw.DT - 0.02)
            elif self._pending is not None:
                s.command_ant(self._pending, self.prefix)
            self._pending = None
            self._t_seen = float(s.data.time)
            q, _ = s.ant_joints(self.prefix)
        # Predkosci z roznic katow, jak na robocie (tam serwa podaja tylko kat).
        self._qd = np.zeros(8) if self._q_prev is None else (q - self._q_prev) / aw.DT
        self._q_prev = q

    def joints(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._q_prev is None:
            with self.twin.lock:
                q, _ = self.twin.scene.ant_joints(self.prefix)
            return q, np.zeros(8)
        return self._q_prev.copy(), self._qd.copy()

    def truth(self) -> np.ndarray | None:
        with self.twin.lock:
            return self.twin.scene.ant_pose(self.prefix)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ petla
@dataclass
class AntStatus:
    #: wylaczona / stoi / przejscie / jazda / obrot / STOP
    mode: str = "wylaczona"
    direction: str = ""
    note: str = ""
    pose_age: float | None = None
    T_ant2base: np.ndarray | None = None
    heading_deg: float | None = None
    steps: int = 0
    keys: str = ""
    #: Regulator wysokosci: sredni kat kostek, przeswit, offset kostek.
    height: str = ""


def nominal_pose(psi: float, theta: float, p0: np.ndarray, v: float, t: float, z: float = NOMINAL_Z) -> np.ndarray:
    """Poza "kamery nominalnej" (robot_policy.py --camera nominal): jazda prosto kursem `psi` z predkoscia
    z treningu, korpus poziomo na wysokosci z treningu - tak widzi siebie polityka w idealnym przejezdzie."""
    T = np.eye(4)
    T[:3, :3] = aw._rz(psi - theta)
    T[:3, 3] = [p0[0] + v * t * np.cos(psi), p0[1] + v * t * np.sin(psi), z]
    return T


class AntDrive:
    """Jedna petla mrowki: klawisze, polityka, przejscia, regulator wysokosci, STOP.

    Przejscia jak `robot_teleop.Teleop.set_mode`: stoi -> jazda od razu z pozy startowej; jazda -> inny kierunek:
    restart obserwatora z nowym przypisaniem nog, bez rampy; obrot -> jazda, cokolwiek -> obrot albo stop: rampa
    przez poze startowa. `tick` = jeden krok 50 ms.
    """

    def __init__(self, twin, get_ant: Callable[[], Ant | None], io=None, policy: aw.WalkPolicy | None = None,
                 camera: str = "nominal", clock: Callable[[], float] = time.monotonic,
                 on_pose: Callable[[np.ndarray], None] | None = None, hold_height: bool = True,
                 frame: str = "widok", view_dir: Callable[[], np.ndarray | None] | None = None):
        self.twin = twin
        self.get_ant = get_ant
        self.io = io if io is not None else SimAnt(twin)
        self.policy = policy or aw.load_policy()
        self.observer = aw.Observer(self.policy.cfg)
        #: Skad obserwacja bierze poze korpusu: "tagi" (kamery stanowiska), "prawda" (tylko symulacja, z fizyki),
        #: "nominal" (idealny przejazd, bez kamery - jak robot_teleop.py domyslnie).
        self.camera = camera
        #: "widok" - WSAD wzgledem widoku 3D (`view_dir`), "mrowka" - wzgledem przodu mrowki.
        self.frame = frame
        self.view_dir = view_dir
        self._latch: tuple[str, str] | None = None             # (wcisniety klawisz, wybrany kierunek)
        self.clock = clock
        self.on_pose = on_pose
        self.height = aw.HeightHold(enabled=hold_height)
        self.status = AntStatus()
        self.start_pose = aw.start_action(self.policy.cfg)
        self.cycles = {"obrot_lewo": aw.turn_cycle(1.0), "obrot_prawo": aw.turn_cycle(-1.0)}
        self._lock = threading.Lock()
        self._keys, self._keys_t = "", -1e9
        self._stopped = ""
        self._last_cmd = self.start_pose.copy()
        self._queue: deque[np.ndarray] = deque()
        self._mode: str | None = None                 # None = stoi, kierunek z DIRECTIONS albo obrot_*
        self._pending_reset = False
        self._jidx = aw.leg_index("przod")
        self._a_prev: np.ndarray | None = None
        self._turn_i = 0
        self._psi = 0.0
        self._nominal: tuple[np.ndarray, float] | None = None     # (start, chwila) przejazdu nominalnego
        self._pose: tuple[float, np.ndarray] | None = None        # ostatnia poza (chwila, T)
        self._thread: threading.Thread | None = None
        self._run = threading.Event()
        self._started = False

    # ---------------------------------------------------------------- wejscia
    def keys(self, pressed: str) -> None:
        """Stan klawiszy z pulpitu (np. "w", "wd", "") - czuwak: trzeba go odswiezac co < `DEADMAN`."""
        order = "".join(k for k in str(pressed).lower() if k in KEYS)
        with self._lock:
            self._keys, self._keys_t = order, self.clock()

    def stop(self, reason: str = "STOP") -> None:
        with self._lock:
            self._stopped = reason or "STOP"

    def clear(self) -> None:
        with self._lock:
            self._stopped = ""

    @property
    def stopped(self) -> str:
        return self._stopped

    # ---------------------------------------------------------------- watek
    def start(self) -> None:
        if self._thread is not None:
            return
        self._run.set()
        self._thread = threading.Thread(target=self._loop, name="mrowka", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        t0, k = time.monotonic(), 0
        while self._run.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - petla mrowki nie moze zginac po cichu
                self.stop(f"blad petli mrowki: {e}")
                self.status.mode, self.status.note = "STOP", self._stopped
            k += 1
            dt = t0 + k * aw.DT - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            elif dt < -1.0:                               # zaleglosci (np. przebudowa sceny) - bez nadrabiania
                t0, k = time.monotonic(), 0

    def close(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.io.close()

    # ---------------------------------------------------------------- poza
    def _measure(self, now: float) -> None:
        """Nowa poza mrowki: z kamer (tryb "tagi") albo z fizyki (pozostale - do WSAD wzgledem widoku i przewrocenia)."""
        T, t = None, now
        if self.camera != "tagi":
            T = self.io.truth()
        else:
            ant = self.get_ant()
            if ant is not None and ant.tags.known():
                frames, times = {}, []
                for cam in self.twin.workspace.cameras:
                    if not cam.enabled or cam.T_cam2base is None:
                        continue
                    try:
                        img, tc = self.twin.cameras.frame_t(cam.name)
                    except Exception:
                        continue
                    if img is None:
                        continue
                    if self.io.kind == "sim":
                        tc = now                          # kadr symulowany renderowany teraz, w czasie symulacji
                    elif not tc or time.monotonic() - tc > 0.5:
                        continue
                    frames[cam.name] = img
                    times.append(tc)
                if frames:
                    fix = locate(frames, self.twin.workspace.cameras, ant)
                    if fix is not None:
                        T, t = fix.T_ant2base, min(times)
        if T is not None:
            self._pose = (t, T)
            if self.camera != "nominal":
                self.observer.add_pose(t, T)
            if self.on_pose is not None:
                self.on_pose(T)

    def _pose_fresh(self, now: float) -> bool:
        return self._pose is not None and now - self._pose[0] <= POSE_TIMEOUT

    # ---------------------------------------------------------------- krok
    def tick(self) -> None:
        wall = self.clock()
        if not self._started:
            why = self.io.ready()
            if why:
                self.status.mode, self.status.note = "wylaczona", why
                return
            self.io.startup()
            self._started = True
            self._last_cmd = self._measured_action()
            self._enqueue_ramp(self.start_pose, 20)           # do pozy startowej (robot_teleop.py)
        self.io.advance()
        now = self.io.now()
        self._measure(now)
        with self._lock:
            keys = self._keys if wall - self._keys_t < DEADMAN else ""
            stopped = self._stopped
        if self._pose is not None:
            tilt = float(np.degrees(np.arccos(np.clip(self._pose[1][2, 2], -1, 1))))
            if tilt > FALL_TILT_DEG and not stopped:
                self.stop(f"mrowka przewrocona ({tilt:.0f} st. od pionu)")
                stopped = self._stopped
        want = None if stopped else self._resolve(keys[:1])
        if want in aw.DIRECTIONS and self.camera == "tagi" and not self._pose_fresh(now):
            self.status.note = "brak pozy mrowki z kamer - polityka bez niej jest slepa; stoi"
            want = None
        self._set_mode(want)
        self._step(now)
        st = self.status
        st.mode = ("STOP" if stopped else "przejscie" if self._queue else "stoi" if self._mode is None else
                   "obrot" if self._mode in self.cycles else "jazda")
        st.direction = self._mode or ""
        st.keys = keys
        st.pose_age = None if self._pose is None else now - self._pose[0]
        if stopped:
            st.note = stopped
        elif st.mode == "jazda" and self.camera == "tagi" and self._pose_fresh(now):
            st.note = ""
        if self._pose is not None:
            st.T_ant2base = self._pose[1]
            st.heading_deg = float(np.degrees(aw.heading(self._pose[1])))
        if self.height.filt is not None:
            st.height = (f"kostki {np.degrees(self.height.filt):.1f} st. ~ przeswit "
                         f"{aw.HeightHold.clearance(self.height.filt) * 100:.1f} cm, offset {self.height.bias:+.3f}")

    def _resolve(self, key: str) -> str | None:
        """Klawisz -> tryb; W S A D wzgledem widoku (`frame`="widok"), wybor trzymany do puszczenia klawisza."""
        if not key:
            self._latch = None
            return None
        if self._latch is not None and self._latch[0] == key:
            return self._latch[1]
        mode = KEYS[key]
        if mode in aw.DIRECTIONS and self.frame == "widok" and self.view_dir is not None and self._pose is not None:
            v = self.view_dir()
            if v is not None and np.linalg.norm(v[:2]) > 1e-6:
                v = np.asarray(v[:2], float) / np.linalg.norm(v[:2])
                left = np.array([-v[1], v[0]])
                desired = {"przod": v, "tyl": -v, "lewo": left, "prawo": -left}[mode]
                h = aw.heading(self._pose[1])
                mode = max(aw.DIRECTIONS, key=lambda d: np.cos(h + aw.DIRECTIONS[d][1] - np.arctan2(desired[1],
                                                                                                    desired[0])))
        self._latch = (key, mode)
        return mode

    def _measured_action(self) -> np.ndarray:
        j = self.io.joints()
        return self._last_cmd if j is None else aw.ctrl_to_action(j[0])

    def _enqueue_ramp(self, target: np.ndarray, n: int = RAMP_STEPS) -> None:
        frm = self._last_cmd
        self._queue.extend(frm + (np.asarray(target, float) - frm) * i / n for i in range(1, n + 1))

    def _set_mode(self, mode: str | None) -> None:
        """Zadany tryb -> nowy tryb z przejsciami w kolejce (robot_teleop.Teleop.set_mode)."""
        if mode == self._mode:
            return
        walking = self._mode in aw.DIRECTIONS
        if mode in aw.DIRECTIONS:
            if not walking and self._mode is not None:    # z obrotu: przez poze startowa
                self._enqueue_ramp(self.start_pose)
            self._jidx, self._pending_reset = aw.leg_index(mode), True
        elif mode in self.cycles:
            self._enqueue_ramp(self.start_pose)
            self._turn_i = 0
        else:
            self._queue.clear()
            self._enqueue_ramp(self.start_pose)
        self._mode = mode

    def _apply(self, cmd: np.ndarray) -> None:
        self._last_cmd = np.clip(np.asarray(cmd, float), -1, 1)
        self.io.command(self._last_cmd)

    def _step(self, now: float) -> None:
        if self._queue:
            self._apply(self._queue.popleft())
            return
        if self._mode in aw.DIRECTIONS:
            j = self.io.joints()
            if j is None:
                self._apply(self._last_cmd)
                return
            jp, jv = j[0][self._jidx], j[1][self._jidx]
            theta = aw.DIRECTIONS[self._mode][1]
            if self._pending_reset:
                self._pending_reset = False
                base = self._pose[1] if self._pose is not None else np.eye(4)
                self._psi = aw.heading(base) + theta
                if self.camera == "nominal":
                    self._nominal = (base[:3, 3].copy(), now)
                    self.observer.poses.clear()
                    self._feed_nominal(now, theta)
                obs = self.observer.reset(now, jp, jv, self._last_cmd[self._jidx], self._psi, theta)
                self.status.steps = 0
            else:
                if self.camera == "nominal":
                    self._feed_nominal(now, theta)
                obs = self.observer.step(now, self._a_prev, jp, jv)
                self.height.update(jp[1::2], self.observer.t)
            a = self.policy.act(obs)
            self._a_prev = a
            phys = np.empty(8)
            phys[self._jidx] = self.height.command(a)
            self._apply(phys)
            self.status.steps += 1
        elif self._mode in self.cycles:
            cycle = self.cycles[self._mode]
            self._apply(cycle[self._turn_i % len(cycle)])
            self._turn_i += 1
        else:
            self._apply(self.start_pose)

    def _feed_nominal(self, now: float, theta: float) -> None:
        p0, t0 = self._nominal
        self.observer.add_pose(now, nominal_pose(self._psi, theta, p0, self.policy.cfg["v_cmd"], now - t0))
