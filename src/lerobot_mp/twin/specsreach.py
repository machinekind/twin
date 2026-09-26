"""Okulary Spectacles -> cel w pudelku zasiegu -> ruch ramienia, z wycofaniem po uderzeniu.

    drv = SpecsReach(twin, relay_url="ws://127.0.0.1:8080/twin", mode="sim")
    drv.start()                                        # watek sieci + watek wykonawcy
    ...
    drv.stop()

Blizniak laczy sie z przekaznikiem (`server.js` z repozytorium specs-lens) jako `/twin`:

* po polaczeniu wysyla nakladke: pudelko zasiegu (`kind: "volume"`) i kulke TCP,
  potem kulke TCP co 0,2 s, a kulke `target` dla przyjetego celu;
* z przekaznika dostaje rejestracje (`registration`: sesja okularow i czy zaufana)
  oraz cele (`target`: punkt dloni albo srodek obiektu w ukladzie podstawy [m]);
* na kazdy cel odpowiada `targetState` (typ zawsze pierwszym kluczem - przekaznik
  czyta go wyrazeniem regularnym): accepted / rejected / moving / reached / backoff / hold.

Watek sieci tylko odbiera i wrzuca cele do kolejki. JEDEN watek wykonawcy robi
wszystko inne: sprawdzenie celu, IK, kolizje, ruch, wycofanie. Kolejnosc sprawdzen
i powody odmowy (`reason`): not connected, stop, busy, not armed, untrusted,
session, outside volume, no depth, ik, jump, collision.

Ruch to wlasna rampa smoothstep 50 Hz (nie `Twin.move`), bo trzeba ja przerwac
w kazdym takcie: nowy cel (jedzie dalej od biezacego rozkazu), uderzenie
(`ImpactGuard` na sluchaczu taktow petli) albo odebranie ramienia (STOP, Dom).
Wlasnosc jak w `Twin.move`: `claim` z `gen`, sprzeglo, `set_target` z wlascicielem
w kazdym takcie, zegar rampy stoi, gdy petla stoi. Po dojechaniu ramie jest
zwalniane, sprzeglo zostaje, cel trzymany.

Tryby: `dry` - wszystkie sprawdzenia i nakladka planowanego celu, bez ruchu
(stan accepted, potem hold); `sim` - rusza tylko ramieniem w symulacji;
`real` - rusza prawdziwym ramieniem, tylko z `real_ok` i po `hold_off_s`
od celu dloni. Na czas ruchu granica predkosci nadzoru (`Twin.set_speed`) spada do
~1,5 x predkosci rampy (+5 st./s), na czas wycofania do 15 st./s; potem wraca.
W `real` zerwane polaczenie z przekaznikiem zatrzymuje jazde (hold). Okulary NIE
maja STOP-u: zatrzymuje Enter/Ctrl-C w konsoli albo STOP w panelu.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import scene as sc
from .collision import CollisionChecker
from .impact import Impact, ImpactGuard
from .kinematics import RobotKinematics
from .reach import ReachVolume
from .runtime import Twin, smoothstep

logger = logging.getLogger(__name__)

MODES = ("dry", "sim", "real")
OWNER = "specs"
#: Kolory nakladki (RGB 0..1).
TCP_COLOR = (1.0, 0.8, 0.1)
TARGET_COLOR = (0.2, 0.9, 0.35)


@dataclass
class Target:
    """Cel z okularow, tak jak przyszedl (po sprawdzeniu typow)."""

    source: str                       # "hand" | "object"
    pos: np.ndarray                   # uklad podstawy [m]
    session: str
    trusted: bool
    ts: Any                           # stempel okularow - odsylany w targetState, nie porownywany z zegarem
    depth: bool = False               # obiekt: glebia zmierzona (nie domyslne 150 cm)
    label: str = ""
    t_rx: float = 0.0                 # chwila odbioru (zegar blizniaka)


@dataclass
class _Leg:
    goal: dict[str, float]
    speed: float                      # st./s najwiekszego stawu
    profile: str                      # "normal" | "descend"
    what: str


class _Abort(Exception):
    """Ruch przerwany z powodu, ktory idzie do `targetState` (stan, powod)."""

    def __init__(self, state: str, reason: str):
        super().__init__(reason)
        self.state, self.reason = state, reason


def parse_target(msg: Mapping[str, Any], now: float) -> Target | None:
    """`target` z przekaznika -> `Target`; None, gdy wiadomosc jest zepsuta."""
    try:
        pos = np.asarray(msg["pos"], float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    source = msg.get("source")
    if source not in ("hand", "object"):
        return None
    return Target(source=source, pos=pos, session=str(msg.get("session") or ""),
                  trusted=msg.get("trusted") is True, ts=msg.get("ts"),
                  depth=msg.get("depthMeasured") is True, label=str(msg.get("label") or ""), t_rx=now)


def down_rotation(spec, p: np.ndarray) -> np.ndarray:
    """Obrot TCP w ukladzie podstawy: os podejscia (`spec.tcp_approach`) w dol (-z),
    os zamykania szczek poziomo, w poprzek kierunku od podstawy do punktu."""
    a = np.asarray(spec.tcp_approach, float)
    c = np.asarray(spec.tcp_closing, float)
    local = np.column_stack([a, c, np.cross(a, c)])
    bearing = math.atan2(float(p[1]), float(p[0]))
    down = np.array([0.0, 0.0, -1.0])
    side = np.array([-math.sin(bearing), math.cos(bearing), 0.0])
    world = np.column_stack([down, side, np.cross(down, side)])
    return world @ local.T


class SpecsReach:
    """Kierowca ramienia z okularow: klient przekaznika i jeden watek wykonawcy."""

    #: Rampa: takt [s], dojechanie: kazdy staw ramienia tyle st. od celu i stoi.
    tick_s = 0.02
    arrive_deg = 2.0
    still_deg_s = 10.0
    arrive_timeout_s = 3.0
    #: Docinanie po dojechaniu: serwo pod ciezarem stoi stale o 1-2 st. od rozkazu (w symulacji
    #: TCP 8 mm za nisko). Rozkaz przesuwany o uchyb, najwyzej `trim_max_deg` od celu IK.
    trim_steps = 4
    trim_deg = 0.2
    trim_max_deg = 2.5
    #: Po koncu rampy straznik ciasniejszy: staw zablokowany przy celu wycofuje sie, zanim
    #: backend A1X zglosi "gluchy" staw (3 st. przez 1 s) i STOP.
    arrive_lag_deg = 2.5
    arrive_lag_s = 0.3
    #: Co ile sekund kulka TCP w nakladce; po ilu sekundach znika kulka celu.
    tcp_period_s = 0.2
    target_linger_s = 2.0
    reconnect_s = 2.0
    #: Tolerancja IK [m] i najwiekszy blad osi podejscia [rad] dla celu obiektu.
    ik_tol = 0.003
    approach_tol = 0.26
    #: Sila chwytu: rozkaz chwytaka i czas na zamkniecie.
    grip_closed = 0.0
    grip_wait_s = 1.2
    #: Symulacja: prog wysilku jako czesc `forcerange` aktuatora (zwykly ruch / opuszczanie).
    sim_effort_frac = 0.8
    sim_effort_frac_descend = 0.5
    #: A1X: prog wysilku w surowej skali ramki 0x052 (zgadywane, patrz docs/TWIN.md).
    real_effort_abs = 20.0
    real_effort_abs_descend = 12.0
    #: Wycofanie: straznik slucha dopiero po tylu sekundach - staw oparty o przeszkode ma jeszcze
    #: przez chwile wysilek ponad prog (symulacja: 27 N m na J1 zaraz po uderzeniu).
    backoff_grace_s = 0.3
    #: Granica predkosci nadzoru na czas wycofania [st./s].
    backoff_cap_deg_s = 15.0
    #: Wysilek stawu w spoczynku ponad taka czesc progu = regula wysilku wylaczona dla stawu w tym ruchu.
    effort_rest_frac = 0.5
    #: Tryb dry: co ile sekund wysilek stawow do logu (liczby do progow na pierwszej sesji).
    dry_effort_period_s = 1.0

    def __init__(self, twin: Twin, *, relay_url: str, volume: ReachVolume | None = None, mode: str = "dry",
                 hover: float = 0.08, touch_dz: float = 0.01, speed_deg_s: float = 20.0,
                 backoff_speed_deg_s: float = 8.0, max_jump_deg: float = 45.0, grasp: bool = False,
                 allow_untrusted: bool = False, real_ok: bool = False, hold_off_s: float = 1.0,
                 on_state: Callable[[dict], None] | None = None):
        if mode not in MODES:
            raise ValueError(f"tryb {mode!r}: dozwolone {', '.join(MODES)}")
        self.twin = twin
        self.spec = twin.spec
        self.relay_url = relay_url
        self.volume = volume if volume is not None else ReachVolume.from_workspace(twin.workspace)
        self.mode = mode
        self.hover = float(hover)
        self.touch_dz = float(touch_dz)
        self.speed = float(speed_deg_s)
        self.backoff_speed = float(backoff_speed_deg_s)
        self.max_jump = float(max_jump_deg)
        self.grasp = bool(grasp)
        self.allow_untrusted = bool(allow_untrusted)
        self.real_ok = bool(real_ok)
        self.hold_off_s = float(hold_off_s)
        self.on_state = on_state
        self.arm = tuple(j for j in self.spec.joints if j != self.spec.gripper)

        self._stop = threading.Event()
        self._net: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue[Target] = queue.Queue()
        self._ws = None
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        #: Sesja ostatniej ZAUFANEJ rejestracji (i ostatniej w ogole - dla `allow_untrusted`).
        self._session: str | None = None
        self._session_any: str | None = None
        #: Przerwanie biezacej rampy: zdarzenie + przyczyna ("retarget", "impact", "preempt", "stop").
        self._cancel = threading.Event()
        self._cause = ""
        self._phase = "idle"                  # idle | moving | backoff (ustawia wykonawca)
        self._guard: ImpactGuard | None = None
        self._guard_from = 0.0
        self._impact: Impact | None = None
        self._last_state: dict | None = None
        self._target_shown_until: float | None = None
        #: Kinematyka watku sieci (kulka TCP) - IK i kolizje maja wlasne, w watku wykonawcy.
        self._kin_net = RobotKinematics(self.spec)
        self._kin_worker = RobotKinematics(self.spec)
        self._checker: CollisionChecker | None = None
        self._checker_version = -1
        self._kin: RobotKinematics | None = None
        self._effort_ids: np.ndarray | None = None
        self._effort_limit: np.ndarray | None = None
        #: Granica predkosci nadzoru sprzed pierwszego ruchu (przywracana po nim); `_capped` - zmieniona.
        self._prev_cap: float | None = None
        self._capped = False
        self._current: Target | None = None
        self._detail = ""
        self._validated_from: dict[str, float] | None = None

    # ------------------------------------------------------------ zycie
    def start(self) -> None:
        if self._net is not None:
            return
        self._stop.clear()
        self._net = threading.Thread(target=self._net_loop, name="okulary-siec", daemon=True)
        self._worker = threading.Thread(target=self._work_loop, name="okulary-ruch", daemon=True)
        self._net.start()
        self._worker.start()

    def stop(self) -> None:
        """Konczy watki. Ramie, jesli jechalo, staje w zmierzonej pozie i jest zwalniane."""
        self._stop.set()
        self._interrupt("stop")
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:                           # pragma: no cover - zamykanie po zerwaniu
                pass
        for t in (self._worker, self._net):
            if t is not None:
                t.join(timeout=5.0)
        self._net = self._worker = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def last_state(self) -> dict | None:
        return self._last_state

    def status(self) -> dict:
        """Stan dla panelu: polaczenie, sesja, ostatni `targetState`."""
        return {"connected": self.connected, "mode": self.mode, "session": self._session,
                "phase": self._phase, "last": self._last_state}

    # ------------------------------------------------------------ siec
    def _net_loop(self) -> None:
        from websockets.sync.client import connect

        failures = 0
        while not self._stop.is_set():
            try:
                ws = connect(self.relay_url, proxy=None, open_timeout=3.0, close_timeout=1.0)
            except Exception as exc:
                (logger.warning if failures == 0 else logger.info)(
                    "Przekaznik %s niedostepny (%s) - ponowie za %.0f s", self.relay_url, exc, self.reconnect_s)
                failures += 1
                self._stop.wait(self.reconnect_s)
                continue
            failures = 0
            logger.info("Polaczono z przekaznikiem %s", self.relay_url)
            self._ws = ws
            try:
                self._send({"type": "overlay", "objects": [self.volume.overlay_object(), self._tcp_object()]})
                next_tcp = time.monotonic() + self.tcp_period_s
                while not self._stop.is_set():
                    try:
                        text = ws.recv(timeout=max(0.0, next_tcp - time.monotonic()))
                    except TimeoutError:
                        text = None
                    if text is not None:
                        self._on_message(text)
                    now = time.monotonic()
                    if now >= next_tcp:
                        next_tcp = now + self.tcp_period_s
                        self._send_tcp(now)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("Przekaznik rozlaczony (%s) - ponowie za %.0f s", exc, self.reconnect_s)
            finally:
                self._ws = None
                # Prawdziwe ramie bez przekaznika nie jedzie dalej: okulary nie widza juz stanu.
                if self.mode == "real" and self._phase == "moving" and not self._stop.is_set():
                    self._interrupt("relay")
                try:
                    ws.close()
                except Exception:                       # pragma: no cover
                    pass
            self._stop.wait(self.reconnect_s)

    def _send(self, msg: dict) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(json.dumps(msg))
            return True
        except Exception as exc:
            logger.debug("Wysylka do przekaznika nieudana: %s", exc)
            return False

    def _tcp_now(self, kin: RobotKinematics | None = None) -> list[float]:
        kin = kin or self._kin_net
        return [round(float(v), 4) for v in kin.tcp(self.twin.joints())[:3, 3]]

    def _tcp_object(self) -> dict:
        return {"id": "tcp", "kind": "sphere", "pos": self._tcp_now(), "size": [0.02], "color": list(TCP_COLOR)}

    def _send_tcp(self, now: float) -> None:
        msg: dict[str, Any] = {"type": "overlay", "objects": [self._tcp_object()]}
        with self._lock:
            until = self._target_shown_until
            if until is not None and now >= until:
                self._target_shown_until = None
                msg["remove"] = ["target"]
        self._send(msg)

    def _on_message(self, text: str | bytes) -> None:
        try:
            msg = json.loads(text)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "registration":
            session = msg.get("session")
            if session is None:
                return
            with self._lock:
                self._session_any = str(session)
                # Kazda rejestracja bez `trusted: true` (tez pozniejsza tej samej sesji) odbiera zaufanie.
                self._session = str(session) if msg.get("trusted") is True else None
        elif kind == "target":
            t = parse_target(msg, time.monotonic())
            if t is None:
                logger.warning("Zepsuty cel z okularow: %.200s", text)
                return
            self._queue.put(t)
            # Nowy cel w trakcie jazdy przerywa ja (wykonawca jedzie dalej od biezacego rozkazu).
            if self._phase == "moving":
                self._interrupt("retarget")

    def _interrupt(self, cause: str) -> None:
        with self._lock:
            if not self._cancel.is_set():
                self._cause = cause
                self._cancel.set()

    # ------------------------------------------------------------ targetState
    def _report(self, t: Target, state: str, reason: str | None = None, *, tcp: bool = True,
                detail: str = "") -> None:
        msg: dict[str, Any] = {"type": "targetState", "state": state}
        if reason:
            msg["reason"] = reason
        if detail:
            msg["detail"] = detail
        msg["pos"] = [round(float(v), 4) for v in t.pos]
        if tcp:
            try:
                msg["tcp"] = self._tcp_now(self._kin_worker)
            except Exception:                           # pragma: no cover - diagnostyka
                pass
        msg["ts"] = t.ts
        msg["mode"] = self.mode
        self._last_state = msg
        log = logger.warning if state in ("rejected", "backoff") else logger.info
        log("Cel %s %s: %s%s%s", t.source, np.round(t.pos, 3).tolist(), state, f" ({reason})" if reason else "",
            f" - {detail}" if detail else "")
        self._send(msg)
        if self.on_state is not None:
            try:
                self.on_state(msg)
            except Exception:                           # pragma: no cover
                logger.exception("on_state rzucilo wyjatek")

    def _show_target(self, p: np.ndarray) -> None:
        with self._lock:
            self._target_shown_until = None
        self._send({"type": "overlay", "objects": [{"id": "target", "kind": "sphere",
                                                     "pos": [round(float(v), 4) for v in p],
                                                     "size": [0.025], "color": list(TARGET_COLOR)}]})

    def _hide_target_later(self) -> None:
        with self._lock:
            self._target_shown_until = time.monotonic() + self.target_linger_s

    # ------------------------------------------------------------ wykonawca
    def _work_loop(self) -> None:
        next_effort = 0.0
        while not self._stop.is_set():
            try:
                t = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self.mode == "dry" and time.monotonic() >= next_effort:
                    next_effort = time.monotonic() + self.dry_effort_period_s
                    self._log_efforts("dry")
                continue
            t = self._latest(t)
            try:
                while t is not None and not self._stop.is_set():
                    t = self._execute(t)
            except Exception as exc:
                logger.exception("Kierowca okularow: blad wykonania celu")
                cur = self._current
                if self.twin.owner == OWNER:
                    try:
                        self.twin.hold_measured()       # nie dociskac do ostatniego celu rampy
                    except Exception:                   # pragma: no cover
                        logger.exception("hold_measured po bledzie")
                if cur is not None:
                    self._report(cur, "hold", "error", detail=f"{type(exc).__name__}: {exc}"[:200])
            finally:
                self._current = None
                self._phase = "idle"
                self._release()
                self._restore_speed()
        self._release()
        self._restore_speed()

    def _log_efforts(self, what: str) -> dict[str, float] | None:
        """Wysilek stawow ramienia do logu (progi straznika na A1X sa zgadywane - to daje liczby)."""
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated and self._effort_ids is None:
            self._effort_limits(False)                  # ustawia aktuatory ramienia
        eff = self._efforts()
        if eff:
            logger.info("Wysilek stawow (%s): %s", what, {k: round(float(v), 2) for k, v in eff.items()})
        return eff

    # ------------------------------------------------------------ granica predkosci
    def _cap_speed(self, deg_s: float) -> None:
        """Granica predkosci nadzoru na czas ruchu kierowcy (nie wyzej niz ustawiona w panelu)."""
        tw = self.twin
        if not self._capped:
            self._prev_cap = getattr(tw, "_speed_cap", None)
            self._capped = True
        prev = self._prev_cap
        tw.set_speed(min(prev if prev is not None else 1e9, float(deg_s)))

    def _restore_speed(self) -> None:
        if not self._capped:
            return
        self._capped = False
        try:
            self.twin.set_speed(self._prev_cap)
        except Exception:                               # pragma: no cover
            logger.exception("Przywracanie granicy predkosci")

    def _latest(self, t: Target) -> Target:
        """Z kolejki bierze najnowszy cel; starsze, nieobsluzone, dostaja "busy"."""
        while True:
            try:
                nxt = self._queue.get_nowait()
            except queue.Empty:
                return t
            self._report(t, "rejected", "busy")
            t = nxt

    def _release(self) -> None:
        try:
            self.twin.release(OWNER)
        except Exception:                               # pragma: no cover
            pass

    def _tools(self) -> tuple[RobotKinematics, CollisionChecker]:
        """IK i sprawdzacz kolizji na scenie stanowiska BEZ dodatkow (obiekty zadania, karta).

        Budowane w watku wykonawcy, od nowa po przebudowie sceny blizniaka (`twin.version`).
        """
        v = self.twin.version
        if self._checker is None or self._checker_version != v:
            scene = sc.build(self.twin.workspace.scene_config())
            self._checker = CollisionChecker(scene)
            self._kin = RobotKinematics(self.spec, scene.model, sc.PREFIX)
            self._checker_version = v
        assert self._kin is not None
        return self._kin, self._checker

    def _estopped(self) -> bool:
        from ..control.safety import SafetyState
        return self.twin.safety_state == SafetyState.ESTOP

    def _command(self) -> dict[str, float]:
        tw = self.twin
        return {k: float(v) for k, v in (tw.status.command or tw.joints()).items()}

    def _hover_point(self, pos: np.ndarray) -> np.ndarray:
        """Punkt nad obiektem: `hover` wyzej, ale najwyzej 3 cm pod sufitem pudelka
        (wyzej IK z TCP w dol czesto nie ma rozwiazania)."""
        p = np.array(pos, float)
        p[2] = min(float(pos[2]) + self.hover, float(self.volume.hi[2]) - 0.03)
        return p

    def _validate(self, t: Target) -> tuple[str | None, list[_Leg]]:
        """(powod odmowy albo None, odcinki ruchu). Kolejnosc sprawdzen jak w module.
        Szczegol odmowy (IK, kolizja) w `self._detail`."""
        tw = self.twin
        self._detail = ""
        if not tw.connected:
            return "not connected", []
        if self._estopped():
            return "stop", []
        if tw.owner not in (None, OWNER):
            return "busy", []
        if self.mode == "real" and not self.real_ok:
            return "not armed", []
        if self.mode == "sim" and not tw.status.simulated:
            return "not armed", []
        with self._lock:
            session = self._session_any if self.allow_untrusted else self._session
        if not t.trusted and not self.allow_untrusted:
            return "untrusted", []
        if session is None:
            return "untrusted", []
        if t.session != session:
            return "session", []
        if not self.volume.contains(t.pos):
            return "outside volume", []
        if t.source == "object" and not t.depth:
            return "no depth", []

        kin, checker = self._tools()
        cmd = self._command()
        self._validated_from = {k: cmd[k] for k in self.arm if k in cmd}
        legs: list[_Leg] = []
        # Ramie z mniej niz 6 stawami (SO-101) nie ustawi dowolnego obrotu: IK tylko na pozycje,
        # a os podejscia w dol sprawdzana tylko przed zejsciem (--grasp); samo "nad obiekt" bez niej.
        full_rot = len(self.arm) >= 6
        if t.source == "hand":
            points = [(t.pos, None, False, 1.0, "normal", "dlon")]
        else:
            R = down_rotation(self.spec, t.pos) if full_rot else None
            check_down = full_rot or self.grasp
            points = [(self._hover_point(t.pos), R, check_down, 1.0, "normal", "nad obiekt")]
            if self.grasp:
                points.append((t.pos + np.array([0.0, 0.0, self.touch_dz]), R, True, 0.5, "descend", "w dol"))
        prev = {k: cmd[k] for k in self.arm if k in cmd}
        for p, R, check_down, factor, profile, what in points:
            sol = kin.ik(p, rotation=R, seed={**cmd, **prev}, tol=self.ik_tol)
            if not sol.ok:
                self._detail = f"IK {what}: blad {sol.pos_err * 1000:.1f} mm"
                logger.info(self._detail)
                return "ik", []
            if check_down:
                approach = kin.tool_axes(sol.joints)[0]
                off = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
                if off > self.approach_tol:
                    self._detail = f"IK {what}: os podejscia {math.degrees(off):.0f} st. od pionu"
                    logger.info(self._detail)
                    return "ik", []
            goal = {k: float(v) for k, v in sol.joints.items() if k in self.arm}
            jump = max(abs(goal[k] - prev.get(k, goal[k])) for k in goal)
            if jump > self.max_jump:
                self._detail = f"skok {what} {jump:.0f} st. > {self.max_jump:.0f}"
                logger.info(self._detail)
                return "jump", []
            why = checker.path_clear(prev, goal)
            if why is not None:
                self._detail = f"kolizja {what}: {why}"
                logger.info(self._detail)
                return "collision", []
            legs.append(_Leg(goal, self.speed * factor, profile, what))
            prev = goal
        if t.source == "object" and self.grasp:
            # Po chwycie z powrotem nad obiekt, tym samym torem (juz sprawdzonym), z zamknietym chwytakiem.
            back = dict(legs[0].goal)
            if self.spec.gripper:
                back[self.spec.gripper] = self.grip_closed
            legs.append(_Leg(back, self.speed * 0.5, "descend", "w gore"))
        return None, legs

    def _execute(self, t: Target) -> Target | None:
        """Obsluga jednego celu; zwraca nastepny cel, gdy ten przerwano nowym."""
        self._current = t
        with self._lock:
            if not self._stop.is_set():                     # `stop()` z tej chwili nie moze przepasc
                self._cancel.clear()
                self._cause = ""
        # "moving" juz od sprawdzenia (IK + kolizje trwaja): nowy cel w tym czasie zastepuje ten,
        # zamiast czekac w kolejce i ruszyc ramieniem po nim.
        self._phase = "moving"
        if not self._queue.empty():
            self._interrupt("retarget")
        reason, legs = self._validate(t)
        if reason is not None:
            self._report(t, "rejected", reason, detail=self._detail)
            self._release()
            return None
        self._report(t, "accepted")
        self._show_target(t.pos if t.source == "hand" else self._hover_point(t.pos))
        if self.mode == "dry":
            self._report(t, "hold")
            return None
        if self._cancel.is_set():
            return self._after_cancel(t)
        if self.mode == "real" and t.source == "hand" and self.hold_off_s > 0:
            if self._cancel.wait(self.hold_off_s):          # nowy cel w trakcie odczekania go zastepuje
                return self._after_cancel(t)
        try:
            gen = self._take()
            self._recheck(legs)
        except _Abort as exc:
            self._report(t, exc.state, exc.reason, detail=self._detail)
            self._release()
            return None
        self._cap_speed(1.5 * self.speed + 5.0)            # smoothstep: szczyt 1,5 x sredniej
        self._guard = ImpactGuard(self.arm)
        self._report(t, "moving")
        self._detail = ""
        for k, leg in enumerate(legs):
            try:
                if leg.what == "w gore" and self.spec.gripper:
                    self._close_gripper(gen)
                self._tune_guard(self._guard, leg.profile)
                done = self._ramp(leg.goal, leg.speed, gen, profile=leg.profile)
            except _Abort as exc:
                self._report(t, exc.state, exc.reason, detail=self._detail)
                self._hide_target_later()
                self._release()
                return None
            if not done:
                return self._after_cancel(t)
            if k + 1 < len(legs):
                self._report(t, "moving")
        self._report(t, "reached")
        self._hide_target_later()
        self._release()                                     # sprzeglo zostaje, cel trzymany
        return None

    def _after_cancel(self, t: Target) -> Target | None:
        with self._lock:
            cause = self._cause
        if cause == "retarget":
            try:
                nxt = self._latest(self._queue.get_nowait())
            except queue.Empty:
                nxt = None
            self._report(t, "hold", "retargeted")
            return nxt
        if cause == "impact":
            self._backoff(t)
            return None
        if cause == "relay":
            reason = "relay"
        else:
            reason = "stop" if (cause == "stop" and self._stop.is_set()) or self._estopped() else "preempted"
        self._report(t, "hold", reason)
        self._hide_target_later()
        self._release()
        return None

    def _take(self) -> int:
        """Ramie dla kierowcy jak w `Twin.move`: czekanie na dom, `claim` z `gen`, sprzeglo."""
        tw = self.twin
        try:
            tw._wait_for_homing()
        except RuntimeError:
            raise _Abort("rejected", "busy") from None
        gen = tw.preempt_gen
        try:
            tw.claim(OWNER, preempt=lambda: self._interrupt("preempt"), gen=gen)
        except RuntimeError:
            raise _Abort("rejected", "stop" if self._estopped() else "busy") from None
        if not tw.set_engaged(True, owner=OWNER):
            self._release()
            raise _Abort("rejected", "stop" if self._estopped() else "busy")
        return gen

    def _recheck(self, legs: list[_Leg]) -> None:
        """Po wzieciu ramienia: rozkaz mogl sie zmienic od sprawdzenia (suwaki w czasie odczekania).
        Skok i droga do pierwszego celu jeszcze raz, od biezacego rozkazu; inaczej hold "moved"."""
        if not legs:
            return
        cmd = self._command()
        first = legs[0].goal
        start = {k: cmd[k] for k in self.arm if k in cmd}
        validated = self._validated_from
        if validated is not None and all(abs(start[k] - validated.get(k, math.inf)) < 0.5 for k in start):
            return
        jump = max((abs(first[k] - start.get(k, first[k])) for k in first if k in self.arm), default=0.0)
        if jump > self.max_jump:
            self._detail = f"rozkaz zmieniony w czasie odczekania: skok {jump:.0f} st."
            raise _Abort("hold", "moved")
        _, checker = self._tools()
        why = checker.path_clear(start, {k: first[k] for k in first if k in self.arm})
        if why is not None:
            self._detail = f"rozkaz zmieniony w czasie odczekania: kolizja {why}"
            raise _Abort("hold", "moved")

    def center_pose(self) -> dict[str, float] | None:
        """Stawy ramienia z TCP nad srodkiem pudelka zasiegu (IK od pozy domowej); None - brak IK."""
        kin, _ = self._tools()
        sol = kin.ik(self.volume.center, seed=self.spec.home, tol=self.ik_tol)
        if not sol.ok:
            return None
        return {k: float(v) for k, v in sol.joints.items() if k in self.arm}

    def start_center(self, speed_deg_s: float = 10.0) -> str | None:
        """Przed startem kierowcy: etapami (`spec.home_stages`) nad srodek pudelka, pod tym samym
        straznikiem uderzen. Z pozy domowej kazdy punkt pudelka to skok > 45 st. (odmowa "jump").

        Wywolywac PRZED `start()` (z watku, ktory potem uruchomi kierowce). Uderzenie = trzymanie
        zmierzonej pozy, bez wycofania. None - dojechal, inaczej powod.
        """
        tw = self.twin
        if not tw.connected:
            return "not connected"
        if self._estopped():
            return "stop"
        if not (tw.status.simulated or (self.mode == "real" and self.real_ok)):
            return "not armed"
        goal = self.center_pose()
        if goal is None:
            return "ik"
        _, checker = self._tools()
        start = {k: v for k, v in self._command().items() if k in self.arm}
        steps = self._stage_plan(start, goal, checker)
        if isinstance(steps, str):
            logger.warning("Start nad srodek pudelka: %s", steps)
            return "collision"
        with self._lock:
            self._cancel.clear()
            self._cause = ""
        try:
            gen = self._take()
        except _Abort as exc:
            return exc.reason
        self._cap_speed(1.5 * speed_deg_s + 5.0)
        self._guard = ImpactGuard(self.arm)
        try:
            for step in steps:
                self._tune_guard(self._guard, "normal")
                if not self._ramp(step, speed_deg_s, gen):
                    with self._lock:
                        cause = self._cause
                    tw.hold_measured()
                    hit = self._impact
                    self._impact = None
                    if cause == "impact" and hit is not None:
                        logger.warning("Start nad srodek pudelka: uderzenie %s %s - trzymanie", hit.kind, hit.joint)
                        return f"impact {hit.kind} {hit.joint}"
                    return cause or "preempted"
            return None
        except _Abort as exc:
            return exc.reason
        finally:
            self._guard = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _stage_plan(self, start: dict[str, float], goal: dict[str, float],
                    checker: CollisionChecker) -> list[dict[str, float]] | str:
        """Pozy posrednie jak w domu etapami: grupy stawow po kolei (tez odwrotnie), a na koniec
        caly cel; pierwszy plan z czystymi drogami wygrywa, bez etapow - prosto. Albo opis kolizji."""
        stages = [tuple(j for j in st if j in self.arm) for st in self.spec.home_stages]
        stages = [st for st in stages if st]
        orders = [stages, stages[::-1]] if stages else []
        why = ""
        for order in [*orders, []]:
            cur, out = dict(start), []
            bad = None
            for stage in [*order, tuple(self.arm)]:
                nxt = dict(cur)
                for j in stage:
                    nxt[j] = goal[j]
                if max((abs(nxt[j] - cur[j]) for j in nxt), default=0.0) < 1e-3:
                    continue
                bad = checker.path_clear(cur, nxt)
                if bad is not None:
                    break
                out.append(nxt)
                cur = nxt
            if bad is None:
                return out
            why = why or bad
        return why

    def _close_gripper(self, gen: int) -> None:
        g = self.spec.gripper
        self.twin.set_target({g: self.grip_closed}, owner=OWNER)
        if self._cancel.wait(self.grip_wait_s):
            return
        if self.twin.preempt_gen != gen:
            raise _Abort("hold", "stop" if self._estopped() else "preempted")

    # ------------------------------------------------------------ rampa
    def _ramp(self, goal: Mapping[str, float], speed: float, gen: int, *, guarded: bool = True,
              profile: str = "normal") -> bool:
        """Rampa smoothstep do `goal` i czekanie na dojechanie. True = dojechal, False = przerwana
        (`self._cause`); `_Abort` - ramie odebrane albo nie dojechalo (wtedy trzyma zmierzona poze)."""
        tw = self.twin
        start_all = self._command()
        start = {k: start_all.get(k, float(v)) for k, v in goal.items()}
        goal = {k: float(v) for k, v in goal.items()}
        jump = max((abs(goal[k] - start[k]) for k in goal if k in self.arm), default=0.0)
        duration = max(0.5, jump / max(speed, 1e-3))
        handle = tw.add_tick_listener(self._on_tick) if guarded else None
        self._phase = "moving" if guarded else "backoff"
        frozen = getattr(tw, "_frozen", None)
        try:
            t0 = t_prev = time.monotonic()
            paused = 0.0
            while True:
                if self._cancel.is_set():
                    return False
                self._check(gen)
                now = time.monotonic()
                if frozen is not None and frozen():
                    paused += now - t_prev
                t_prev = now
                s = smoothstep((now - t0 - paused) / duration)
                target = {k: start[k] + s * (goal[k] - start[k]) for k in goal}
                try:
                    tw.set_target(target, owner=OWNER)
                except RuntimeError:
                    raise _Abort("hold", "stop" if self._estopped() else "preempted") from None
                if s >= 1.0:
                    break
                if self._cancel.wait(self.tick_s):
                    return False
            guard = self._guard
            if guard is not None:
                # Staw zablokowany tuz przy celu: wycofanie, zanim backend uzna go za "gluchy".
                guard.lag_deg, guard.lag_s = self.arrive_lag_deg, self.arrive_lag_s
            # Dojechanie: kazdy staw ramienia najwyzej `arrive_deg` od celu i stoi.
            deadline = time.monotonic() + self.arrive_timeout_s
            prev: tuple[float, dict[str, float]] | None = None
            while True:
                if self._cancel.wait(0.05):
                    return False
                self._check(gen)
                st = tw.status
                meas = dict(st.measured)
                far = [k for k in goal if k in self.arm and abs(meas.get(k, goal[k]) - goal[k]) > self.arrive_deg]
                still = False
                if prev is not None and st.measured_t > prev[0]:
                    dt = st.measured_t - prev[0]
                    still = all(abs(meas[k] - prev[1].get(k, meas[k])) / dt < self.still_deg_s
                                for k in self.arm if k in meas)
                if prev is None or st.measured_t > prev[0]:
                    prev = (st.measured_t, meas)
                if not far and still:
                    if profile == "descend":
                        return True                     # przy zejsciu bez docinania: docisk do obiektu
                    return self._trim(goal, gen)
                if time.monotonic() > deadline:
                    # Staw zablokowany nie naciska dalej na przeszkode celem, ktorego nie osiagnal.
                    tw.hold_measured()
                    self._detail = ", ".join(f"{k} {meas.get(k, math.nan) - goal[k]:+.1f} st." for k in far)
                    raise _Abort("hold", "not arrived")
        finally:
            if handle is not None:
                tw.remove_tick_listener(handle)

    def _trim(self, goal: Mapping[str, float], gen: int) -> bool:
        """Rozkaz = cel + uchyb (pomiar ma trafic w cel IK), kilka krokow. False = przerwane."""
        tw = self.twin
        cmd = {k: float(v) for k, v in goal.items()}
        guard = self._guard
        if guard is not None and guard.effort_abs:
            eff = self._efforts() or {}
            loaded = [j for j, lim in guard.effort_abs.items()
                      if j in eff and abs(eff[j]) > self.effort_rest_frac * lim]
            if loaded:
                logger.info("Bez docinania: wysilek %s ponad polowe progu", loaded)
                return True
        for _ in range(self.trim_steps):
            meas = dict(tw.status.measured)
            off = {k: goal[k] - meas[k] for k in goal if k in self.arm and k in meas}
            if not off or max(abs(v) for v in off.values()) < self.trim_deg:
                break
            for k, v in off.items():
                cmd[k] = float(np.clip(cmd[k] + v, goal[k] - self.trim_max_deg, goal[k] + self.trim_max_deg))
            try:
                tw.set_target(cmd, owner=OWNER)
            except RuntimeError:
                raise _Abort("hold", "stop" if self._estopped() else "preempted") from None
            if self._cancel.wait(0.4):
                return False
            self._check(gen)
        return True

    def _check(self, gen: int) -> None:
        tw = self.twin
        if self._estopped():
            raise _Abort("hold", "stop")
        if tw.preempt_gen != gen:
            raise _Abort("hold", "preempted")
        if not tw.connected:
            raise _Abort("hold", "not connected")

    # ------------------------------------------------------------ uderzenie
    def _tune_guard(self, guard: ImpactGuard, profile: str) -> None:
        """Progi straznika dla odcinka: `descend` ciasniejszy (opoznienie 3 st., mniejszy wysilek).
        Jeden straznik na caly cel - bufor dobrych probek (poza wycofania) przechodzi przez odcinki."""
        descend = profile == "descend"
        guard.lag_deg = 3.0 if descend else 6.0
        guard.lag_s = 0.15
        limits = self._effort_limits(descend) or {}
        # Staw, ktory juz w spoczynku ma wysilek ponad polowe progu (ciezar, sprezyna), wyzwalalby
        # przy kazdym ruchu: dla niego w tym odcinku tylko regula opoznienia.
        rest = self._log_efforts("start odcinka") if limits else None
        if rest:
            for j in [j for j, lim in limits.items() if j in rest and abs(rest[j]) > self.effort_rest_frac * lim]:
                logger.warning("Wysilek %s w spoczynku %.1f > %.0f%% progu %.1f - regula wysilku wylaczona "
                               "dla tego stawu w tym ruchu", j, rest[j], 100 * self.effort_rest_frac, limits[j])
                limits.pop(j)
        guard.effort_abs = limits

    def _effort_limits(self, descend: bool) -> dict[str, float] | None:
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated:
            s = self.twin.scene
            frac = self.sim_effort_frac_descend if descend else self.sim_effort_frac
            ids = [s.model.actuator(sc.PREFIX + j).id for j in self.arm]
            rng = s.model.actuator_forcerange[ids]
            limited = s.model.actuator_forcelimited[ids]
            self._effort_ids = np.array(ids)
            return {j: frac * float(max(abs(rng[i, 0]), abs(rng[i, 1])))
                    for i, j in enumerate(self.arm) if limited[i] and rng[i, 1] > rng[i, 0]}
        try:
            probe = b.efforts()
        except Exception:
            probe = None
        if probe is None:
            return None                                 # tylko regula opoznienia
        return {j: self.real_effort_abs_descend if descend else self.real_effort_abs for j in self.arm}

    def _efforts(self) -> dict[str, float] | None:
        """Wysilek stawow: backend, a w symulacji `actuator_force` aktuatorow ramienia [N m]."""
        tw = self.twin
        b = tw.backend
        if b is None:
            return None
        if not b.info.simulated:
            try:
                return b.efforts()
            except Exception:
                return None
        ids = self._effort_ids
        if ids is None or not tw.lock.acquire(timeout=0.005):
            return None
        try:
            f = tw.scene.data.actuator_force[ids]
            return {j: float(f[i]) for i, j in enumerate(self.arm)}
        finally:
            tw.lock.release()

    def _on_tick(self, sample) -> None:
        """Sluchacz taktow (watek petli): tylko zapis uderzenia i przerwanie rampy."""
        guard = self._guard
        if guard is None or self._cancel.is_set() or time.monotonic() < self._guard_from:
            return
        b = self.twin.backend
        vel = None
        if b is not None and not b.info.simulated:
            try:
                vel = b.velocities()
            except Exception:
                vel = None
        hit = guard.feed(sample.t, sample.sent, sample.measured, sample.fresh,
                         efforts=self._efforts(), velocities=vel)
        if hit is not None:
            self._impact = hit
            self._interrupt("impact")

    def _backoff(self, t: Target) -> None:
        """Po uderzeniu: trzymanie zmierzonej pozy, potem wolno do pozy wycofania.
        Drugie uderzenie w trakcie wycofania = tylko trzymanie ("hold")."""
        tw = self.twin
        hit = self._impact
        self._impact = None
        # Czysty pomiar, bez drogi hamowania: staw oparty o przeszkode przestaje na nia naciskac.
        tw.hold_measured(dict(tw.status.measured) or None)
        why = f"impact {hit.kind} {hit.joint}" if hit is not None else "impact"
        logger.warning("Uderzenie: %s (%.1f) - wycofanie", why, hit.value if hit is not None else float("nan"))
        pose = dict(hit.backoff_pose or {}) if hit is not None else {}
        gen = tw.preempt_gen
        if not pose or tw.owner != OWNER:
            self._report(t, "hold", why)
            self._hide_target_later()
            self._release()
            return
        with self._lock:
            if not self._stop.is_set():
                self._cancel.clear()
                self._cause = ""
        self._cap_speed(self.backoff_cap_deg_s)
        self._detail = ""
        self._guard = ImpactGuard(self.arm, effort_abs=self._effort_limits(False))
        self._guard_from = time.monotonic() + self.backoff_grace_s
        handle = tw.add_tick_listener(self._on_tick)
        try:
            if self._cancel.wait(0.06):                 # rampa od rozkazu PO trzymaniu (nastepny takt petli)
                raise _Abort("hold", self._cause or "preempted")
            done = self._ramp(pose, self.backoff_speed, gen, guarded=False)
        except _Abort as exc:
            self._report(t, "hold", f"{why}; {exc.reason}", detail=self._detail)
            self._hide_target_later()
            self._release()
            return
        finally:
            tw.remove_tick_listener(handle)
            self._guard = None
        if not done:
            with self._lock:
                cause = self._cause
            if cause == "impact":
                tw.hold_measured()                      # drugie uderzenie: bez kolejnego wycofania
                self._impact = None
            self._report(t, "hold", why if cause == "impact" else cause)
        else:
            self._report(t, "backoff", why)
        # Cele z czasu wycofania nie jada - okulary pokaza "busy", trzeba wskazac od nowa.
        while True:
            try:
                self._report(self._queue.get_nowait(), "rejected", "busy")
            except queue.Empty:
                break
        self._hide_target_later()
        self._release()
