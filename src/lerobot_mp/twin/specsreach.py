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
session, outside volume, no depth, stale, ik, jump, collision.

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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args

import numpy as np

from . import scene as sc
from .collision import CollisionChecker
from .impact import Impact, ImpactGuard
from .kinematics import RobotKinematics
from .reach import ReachVolume
from .runtime import Twin, smoothstep

logger = logging.getLogger(__name__)

Mode = Literal["dry", "sim", "real"]
Source = Literal["hand", "object"]
#: Stan celu w `targetState` (umowa z okularami - wartosci sie nie zmieniaja).
State = Literal["accepted", "rejected", "moving", "reached", "backoff", "hold"]
#: Profil odcinka: `descend` - wolniej i z ciasniejszym straznikiem (zejscie do obiektu i powrot).
Profile = Literal["normal", "descend"]
#: Co robi wykonawca - czytane z innych watkow (siec, panel).
Phase = Literal["idle", "moving", "backoff"]
#: Dlaczego przerwano rampe; "" - nie przerwano.
Cause = Literal["", "retarget", "impact", "preempt", "stop", "mode", "relay"]

MODES: tuple[Mode, ...] = get_args(Mode)
SOURCES: tuple[Source, ...] = get_args(Source)
OWNER = "specs"

#: Kolory nakladki (RGB 0..1).
TCP_COLOR = (1.0, 0.8, 0.1)
TARGET_COLOR = (0.2, 0.9, 0.35)

Joints = dict[str, float]


@dataclass
class Target:
    """Cel z okularow, tak jak przyszedl (po sprawdzeniu typow)."""

    source: Source
    pos: np.ndarray                   # uklad podstawy [m]
    session: str
    trusted: bool
    ts: Any                           # stempel okularow - odsylany w targetState, nie porownywany z zegarem
    depth: bool = False               # obiekt: glebia zmierzona (nie domyslne 150 cm)
    label: str = ""
    t_rx: float = 0.0                 # chwila odbioru (zegar blizniaka)
    #: Obiekt: ile sekund temu okulary naprawde widzialy wykrycie; None - nie podano (atrapy okularow).
    age: float | None = None


@dataclass(frozen=True, eq=False)
class _Waypoint:
    """Punkt, do ktorego ramie ma dojechac, zanim zostanie przeliczony na stawy (IK)."""

    pos: np.ndarray
    what: str                         # do logu i odmowy: "dlon", "nad obiekt", "w dol"
    rotation: np.ndarray | None = None
    #: Sprawdzic, czy os podejscia chwytaka jest pionowa (w dol).
    check_down: bool = False
    speed_factor: float = 1.0
    profile: Profile = "normal"


@dataclass
class _Leg:
    """Odcinek ruchu, juz w stawach: jedna rampa."""

    goal: Joints
    speed: float                      # st./s najwiekszego stawu
    profile: Profile
    what: str
    close_gripper: bool = False       # najpierw zamknac chwytak (powrot po chwycie)


class _Sample(NamedTuple):
    t: float
    joints: Joints


class _Abort(Exception):
    """Ruch przerwany z powodu, ktory idzie do `targetState`."""

    def __init__(self, state: State, reason: str, detail: str = ""):
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.detail = detail


class _Rejected(_Abort):
    """Cel odrzucony przed ruchem."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__("rejected", reason, detail)


class _LoopClock:
    """Czas rampy, ktory stoi, gdy stoi petla blizniaka."""

    def __init__(self, twin: Twin):
        # `Twin._frozen` nie jest publiczne; atrapy blizniaka w testach go nie maja.
        self._frozen: Callable[[], bool] | None = getattr(twin, "_frozen", None)
        self._start = self._last = time.monotonic()
        self._paused = 0.0

    def elapsed(self) -> float:
        now = time.monotonic()
        if self._frozen is not None and self._frozen():
            self._paused += now - self._last
        self._last = now
        return now - self._start - self._paused


def parse_target(msg: Mapping[str, Any], now: float) -> Target | None:
    """`target` z przekaznika -> `Target`; None, gdy wiadomosc jest zepsuta."""
    try:
        pos = np.asarray(msg["pos"], float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None

    source = msg.get("source")
    if source not in SOURCES:
        return None

    age = msg.get("age")
    if age is not None:
        # Zepsuty wiek = nieznany, ale podany: NaN, czyli odmowa "stale" (nie cisza bez targetState).
        is_number = isinstance(age, (int, float)) and not isinstance(age, bool)
        age = float(age) if is_number else math.nan

    return Target(
        source=source,
        pos=pos,
        session=str(msg.get("session") or ""),
        trusted=msg.get("trusted") is True,
        ts=msg.get("ts"),
        depth=msg.get("depthMeasured") is True,
        label=str(msg.get("label") or ""),
        t_rx=now,
        age=age,
    )


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


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"tryb {mode!r}: dozwolone {', '.join(MODES)}")


def _rounded(v: np.ndarray | list[float]) -> list[float]:
    return [round(float(x), 4) for x in v]


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

    #: Regula opoznienia straznika w rampie [st., s]; przy zejsciu ciasniej.
    lag_deg = 6.0
    lag_deg_descend = 3.0
    lag_s = 0.15

    #: Po koncu rampy straznik ciasniejszy: staw zablokowany przy celu wycofuje sie, zanim
    #: backend A1X zglosi "gluchy" staw (3 st. przez 1 s) i STOP.
    arrive_lag_deg = 2.5
    arrive_lag_s = 0.3

    #: `start_center` (10 st./s): ciasniej - staw zablokowany z 3-6 st. rampy przed soba nie
    #: przekracza 6 st., a uchyb ponad 3 st. liczy juz "gluchy" zegar backendu. Opoznienie 4 st.
    #: w rampie i 0,2 s przy dojezdzaniu - wyzwolenie co najmniej 0,3 s przed bledem backendu.
    center_lag_deg = 4.0
    center_arrive_lag_s = 0.2

    #: Co ile sekund kulka TCP w nakladce; po ilu sekundach znika kulka celu.
    tcp_period_s = 0.2
    target_linger_s = 2.0
    reconnect_s = 2.0

    #: Tolerancja IK [m] i najwiekszy blad osi podejscia [rad] dla celu obiektu.
    ik_tol = 0.003
    approach_tol = 0.26
    #: Punkt nad obiektem najwyzej tyle [m] pod sufitem pudelka (wyzej IK z TCP w dol czesto zawodzi).
    hover_ceiling_gap = 0.03
    #: Rozkaz ruszony od sprawdzenia o mniej niz tyle st. = ten sam (bez ponownego sprawdzenia).
    unchanged_deg = 0.5

    #: Cel obiektu z wykrycia starszego niz tyle sekund (pole `age` z okularow) = odmowa "stale".
    max_object_age_s = 0.5

    #: Sila chwytu: rozkaz chwytaka i czas na zamkniecie.
    grip_closed = 0.0
    grip_wait_s = 1.2

    #: Symulacja: prog wysilku jako czesc `forcerange` aktuatora (zwykly ruch / opuszczanie).
    sim_effort_frac = 0.8
    sim_effort_frac_descend = 0.5
    #: Roboty (`spec.name`), dla ktorych w symulacji nie ma reguly wysilku - tylko opoznienie.
    #: SO-101: kp 998 N m/rad przy forcerange 2,94 N m - naped nasyca sie przy 0,17 st. uchybu,
    #: wiec w zwyklym ruchu |sila| stoi na forcerange (zmierzone: 4 z 5 stawow na 2,94, wrist_roll
    #: 0,6). Sily ponad forcerange nie ma, wiec zaden prog nie ma zapasu 1,5x nad zwyklym ruchem.
    #: Sciane w symulacji SO-101 lapie regula opoznienia (test).
    sim_effort_off = frozenset({"so101"})
    #: A1X: prog wysilku w surowej skali ramki 0x052 (zgadywane, patrz docs/TWIN.md).
    real_effort_abs = 20.0
    real_effort_abs_descend = 12.0
    #: Wysilek stawu w spoczynku ponad taka czesc progu = regula wysilku wylaczona dla stawu w tym ruchu.
    effort_rest_frac = 0.5

    #: Wycofanie: straznik slucha dopiero po tylu sekundach - staw oparty o przeszkode ma jeszcze
    #: przez chwile wysilek ponad prog (symulacja: 27 N m na J1 zaraz po uderzeniu).
    backoff_grace_s = 0.3
    #: Granica predkosci nadzoru na czas wycofania [st./s].
    backoff_cap_deg_s = 15.0

    #: Tryb dry: co ile sekund wysilek stawow do logu (liczby do progow na pierwszej sesji).
    dry_effort_period_s = 1.0

    def __init__(self, twin: Twin, *, relay_url: str, volume: ReachVolume | None = None, mode: Mode = "dry",
                 hover: float = 0.08, touch_dz: float = 0.01, speed_deg_s: float = 20.0,
                 backoff_speed_deg_s: float = 8.0, max_jump_deg: float = 45.0, grasp: bool = False,
                 allow_untrusted: bool = False, real_ok: bool = False, hold_off_s: float = 1.0,
                 on_state: Callable[[dict], None] | None = None):
        _check_mode(mode)
        self.twin = twin
        self.spec = twin.spec
        self.arm = tuple(j for j in self.spec.joints if j != self.spec.gripper)
        self.relay_url = relay_url
        self.volume = volume if volume is not None else ReachVolume.from_workspace(twin.workspace)
        self.mode: Mode = mode
        self.real_ok = bool(real_ok)
        self.hover = float(hover)
        self.touch_dz = float(touch_dz)
        self.speed = float(speed_deg_s)
        self.backoff_speed = float(backoff_speed_deg_s)
        self.max_jump = float(max_jump_deg)
        self.grasp = bool(grasp)
        self.allow_untrusted = bool(allow_untrusted)
        self.hold_off_s = float(hold_off_s)
        self.on_state = on_state

        # Watki i lacze
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

        # Przerwanie biezacej rampy: zdarzenie + przyczyna. Faze ustawia wykonawca.
        self._cancel = threading.Event()
        self._cause: Cause = ""
        self._phase: Phase = "idle"

        # Straznik uderzen (czytany z watku petli w `_on_tick`)
        self._guard: ImpactGuard | None = None
        self._guard_from = 0.0
        self._impact: Impact | None = None

        # Nakladka i stan dla panelu
        self._last_state: dict | None = None
        self._target_shown_until: float | None = None

        #: Kinematyka watku sieci (kulka TCP) - IK i kolizje maja wlasne, w watku wykonawcy.
        self._kin_net = RobotKinematics(self.spec)
        self._kin_worker = RobotKinematics(self.spec)
        self._checker: CollisionChecker | None = None
        self._checker_version = -1
        self._kin: RobotKinematics | None = None
        self._effort_ids: np.ndarray | None = None

        #: Granica predkosci nadzoru sprzed pierwszego ruchu (przywracana po nim); `_capped` - zmieniona.
        self._prev_cap: float | None = None
        self._capped = False

        self._current: Target | None = None
        #: Stawy rozkazu, od ktorych sprawdzono cel (`_recheck` porownuje z nimi).
        self._validated_from: Joints | None = None

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
            with suppress(Exception):                   # zamykanie po zerwaniu
                ws.close()

        for thread in (self._worker, self._net):
            if thread is not None:
                thread.join(timeout=5.0)
        self._net = self._worker = None

    def set_mode(self, mode: Mode, real_ok: bool) -> None:
        """Zmiana trybu z panelu. Ruch w toku (tez odczekanie przed nim) staje: hold "mode"."""
        _check_mode(mode)
        changed = (mode, bool(real_ok)) != (self.mode, self.real_ok)
        self.mode, self.real_ok = mode, bool(real_ok)
        if changed and self._phase == "moving":
            self._interrupt("mode")

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def last_state(self) -> dict | None:
        return self._last_state

    def status(self) -> dict:
        """Stan dla panelu: polaczenie, sesja, ostatni `targetState`."""
        return {
            "connected": self.connected,
            "mode": self.mode,
            "session": self._session,
            "phase": self._phase,
            "last": self._last_state,
        }

    # ------------------------------------------------------------ siec
    def _net_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ws = self._connect(quiet=failures > 0)
            if ws is None:
                failures += 1
            else:
                failures = 0
                self._serve(ws)
            self._stop.wait(self.reconnect_s)

    def _connect(self, *, quiet: bool):
        from websockets.sync.client import connect

        try:
            ws = connect(self.relay_url, proxy=None, open_timeout=3.0, close_timeout=1.0)
        except Exception as exc:
            log = logger.info if quiet else logger.warning
            log("Przekaznik %s niedostepny (%s) - ponowie za %.0f s", self.relay_url, exc, self.reconnect_s)
            return None
        logger.info("Polaczono z przekaznikiem %s", self.relay_url)
        return ws

    def _serve(self, ws) -> None:
        """Jedno polaczenie z przekaznikiem, az do zerwania albo `stop()`."""
        self._ws = ws
        try:
            self._send({"type": "overlay", "objects": [self.volume.overlay_object(), self._tcp_object()]})
            self._pump(ws)
        except Exception as exc:
            if not self._stop.is_set():
                logger.warning("Przekaznik rozlaczony (%s) - ponowie za %.0f s", exc, self.reconnect_s)
        finally:
            self._ws = None
            # Prawdziwe ramie bez przekaznika nie jedzie dalej: okulary nie widza juz stanu.
            if self.mode == "real" and self._phase == "moving" and not self._stop.is_set():
                self._interrupt("relay")
            with suppress(Exception):
                ws.close()

    def _pump(self, ws) -> None:
        """Odbior wiadomosci, a miedzy nimi kulka TCP co `tcp_period_s`."""
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

    def _send(self, msg: dict) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("Wysylka do przekaznika nieudana: %s", exc)
            return False
        return True

    def _tcp_now(self, kin: RobotKinematics | None = None) -> list[float]:
        kin = kin or self._kin_net
        return _rounded(kin.tcp(self.twin.joints())[:3, 3])

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
            self._on_registration(msg)
        elif kind == "target":
            self._on_target(msg, text)

    def _on_registration(self, msg: dict) -> None:
        session = msg.get("session")
        if session is None:
            return
        with self._lock:
            self._session_any = str(session)
            # Kazda rejestracja bez `trusted: true` (tez pozniejsza tej samej sesji) odbiera zaufanie.
            self._session = str(session) if msg.get("trusted") is True else None

    def _on_target(self, msg: dict, text: str | bytes) -> None:
        t = parse_target(msg, time.monotonic())
        if t is None:
            logger.warning("Zepsuty cel z okularow: %.200s", text)
            return
        self._queue.put(t)
        # Nowy cel w trakcie jazdy przerywa ja (wykonawca jedzie dalej od biezacego rozkazu).
        if self._phase == "moving":
            self._interrupt("retarget")

    # ------------------------------------------------------------ przerwanie
    def _interrupt(self, cause: Cause) -> None:
        with self._lock:
            if not self._cancel.is_set():
                self._cause = cause
                self._cancel.set()

    def _reset_cancel(self) -> None:
        with self._lock:
            if not self._stop.is_set():                 # `stop()` z tej chwili nie moze przepasc
                self._cancel.clear()
                self._cause = ""

    @property
    def _cancel_cause(self) -> Cause:
        with self._lock:
            return self._cause

    def _lost_reason(self) -> str:
        """Powod, gdy ramie odebrano w trakcie ruchu."""
        return "stop" if self._estopped() else "preempted"

    # ------------------------------------------------------------ targetState
    def _report(self, t: Target, state: State, reason: str | None = None, *, detail: str = "") -> None:
        msg: dict[str, Any] = {"type": "targetState", "state": state}
        if reason:
            msg["reason"] = reason
        if detail:
            msg["detail"] = detail
        msg["pos"] = _rounded(t.pos)
        with suppress(Exception):                       # tylko diagnostyka
            msg["tcp"] = self._tcp_now(self._kin_worker)
        msg["ts"] = t.ts
        msg["mode"] = self.mode
        self._last_state = msg

        log = logger.warning if state in ("rejected", "backoff") else logger.info
        because = f" ({reason})" if reason else ""
        more = f" - {detail}" if detail else ""
        log("Cel %s %s: %s%s%s", t.source, np.round(t.pos, 3).tolist(), state, because, more)

        self._send(msg)
        if self.on_state is not None:
            try:
                self.on_state(msg)
            except Exception:
                logger.exception("on_state rzucilo wyjatek")

    def _finish(self, t: Target, state: State, reason: str | None = None, detail: str = "") -> None:
        """Ostatni stan celu: raport, kulka celu gasnie po chwili, ramie zwolnione."""
        self._report(t, state, reason, detail=detail)
        self._let_go()

    def _let_go(self) -> None:
        self._hide_target_later()
        self._release()

    def _show_target(self, p: np.ndarray) -> None:
        with self._lock:
            self._target_shown_until = None
        sphere = {"id": "target", "kind": "sphere", "pos": _rounded(p), "size": [0.025], "color": list(TARGET_COLOR)}
        self._send({"type": "overlay", "objects": [sphere]})

    def _hide_target_later(self) -> None:
        with self._lock:
            self._target_shown_until = time.monotonic() + self.target_linger_s

    # ------------------------------------------------------------ wykonawca
    def _work_loop(self) -> None:
        next_effort_log = 0.0
        while not self._stop.is_set():
            try:
                t = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self.mode == "dry" and time.monotonic() >= next_effort_log:
                    next_effort_log = time.monotonic() + self.dry_effort_period_s
                    self._log_efforts("dry")
                continue
            self._run(self._latest(t))
        self._release()
        self._restore_speed()

    def _run(self, t: Target | None) -> None:
        """Cel i te, ktore go po kolei zastapily. Blad = ramie trzyma zmierzona poze."""
        try:
            while t is not None and not self._stop.is_set():
                t = self._execute(t)
        except Exception as exc:
            logger.exception("Kierowca okularow: blad wykonania celu")
            self._hold_after_error(exc)
        finally:
            self._current = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _hold_after_error(self, exc: Exception) -> None:
        if self.twin.owner == OWNER:
            try:
                self.twin.hold_measured()               # nie dociskac do ostatniego celu rampy
            except Exception:
                logger.exception("hold_measured po bledzie")
        if self._current is not None:
            self._report(self._current, "hold", "error", detail=f"{type(exc).__name__}: {exc}"[:200])

    def _latest(self, t: Target) -> Target:
        """Z kolejki bierze najnowszy cel; starsze, nieobsluzone, dostaja "busy"."""
        while True:
            try:
                newer = self._queue.get_nowait()
            except queue.Empty:
                return t
            self._report(t, "rejected", "busy")
            t = newer

    def _reject_queued(self) -> None:
        """Wszystkie cele z kolejki dostaja "busy" - okulary pokaza, ze trzeba wskazac od nowa."""
        while True:
            try:
                t = self._queue.get_nowait()
            except queue.Empty:
                return
            self._report(t, "rejected", "busy")

    def _execute(self, t: Target) -> Target | None:
        """Obsluga jednego celu; zwraca nastepny cel, gdy ten przerwano nowym."""
        self._current = t
        self._reset_cancel()
        # "moving" juz od sprawdzenia (IK + kolizje trwaja): nowy cel w tym czasie zastepuje ten,
        # zamiast czekac w kolejce i ruszyc ramieniem po nim.
        self._phase = "moving"
        if not self._queue.empty():
            self._interrupt("retarget")

        try:
            legs = self._plan(t)
        except _Abort as exc:
            self._report(t, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._report(t, "accepted")
        self._show_target(t.pos if t.source == "hand" else self._hover_point(t.pos))
        if self.mode == "dry":
            self._report(t, "hold")
            return None
        if self._cancel.is_set() or self._held_off(t):
            return self._after_cancel(t)

        try:
            gen = self._take()
            self._recheck(legs)
        except _Abort as exc:
            # Ramie jeszcze nie ruszylo: kulka celu zostaje (jak przed refaktorem), tylko zwolnienie.
            self._report(t, exc.state, exc.reason, detail=exc.detail)
            self._release()
            return None

        self._cap_speed(1.5 * self.speed + 5.0)         # smoothstep: szczyt 1,5 x sredniej
        self._guard = ImpactGuard(self.arm)
        self._report(t, "moving")
        try:
            arrived = self._drive(t, legs, gen)
        except _Abort as exc:
            self._finish(t, exc.state, exc.reason, exc.detail)
            return None
        if not arrived:
            return self._after_cancel(t)

        self._finish(t, "reached")                      # sprzeglo zostaje, cel trzymany
        return None

    def _held_off(self, t: Target) -> bool:
        """W `real` cel dloni czeka `hold_off_s` przed ruchem. True = w tym czasie przerwano."""
        if self.mode == "real" and t.source == "hand" and self.hold_off_s > 0:
            return self._cancel.wait(self.hold_off_s)
        return False

    def _drive(self, t: Target, legs: list[_Leg], gen: int) -> bool:
        """Odcinki po kolei. False = przerwane (`self._cause`)."""
        for i, leg in enumerate(legs):
            if i > 0:
                self._report(t, "moving")
            if leg.close_gripper:
                self._close_gripper(gen)
            self._tune_guard(self._guard, leg.profile)
            if not self._ramp(leg.goal, leg.speed, gen, profile=leg.profile):
                return False
        return True

    def _after_cancel(self, t: Target) -> Target | None:
        """Rampa przerwana: nowy cel jedzie dalej, uderzenie - wycofanie, reszta - trzymanie."""
        cause = self._cancel_cause
        if cause == "retarget":
            try:
                newer = self._latest(self._queue.get_nowait())
            except queue.Empty:
                newer = None
            self._report(t, "hold", "retargeted")
            return newer

        if cause == "impact":
            self._backoff(t)
            return None

        if cause in ("relay", "mode"):
            reason = cause
        elif cause == "stop" and self._stop.is_set():
            reason = "stop"
        else:
            reason = self._lost_reason()
        self._finish(t, "hold", reason)
        return None

    # ------------------------------------------------------------ sprawdzenie celu
    def _plan(self, t: Target) -> list[_Leg]:
        """Odcinki ruchu do celu. Odmowa = `_Rejected`, w kolejnosci z opisu modulu."""
        self._gate(t)
        return self._legs(t)

    def _gate(self, t: Target) -> None:
        """Sprawdzenia bez kinematyki: polaczenie, wlasnosc, tryb, zaufanie, pudelko, obiekt."""
        tw = self.twin
        if not tw.connected:
            raise _Rejected("not connected")
        if self._estopped():
            raise _Rejected("stop")
        if tw.owner not in (None, OWNER):
            raise _Rejected("busy")
        if self.mode == "real" and not self.real_ok:
            raise _Rejected("not armed")
        if self.mode == "sim" and not tw.status.simulated:
            raise _Rejected("not armed")

        with self._lock:
            session = self._session_any if self.allow_untrusted else self._session
        if not (t.trusted or self.allow_untrusted) or session is None:
            raise _Rejected("untrusted")
        if t.session != session:
            raise _Rejected("session")

        if not self.volume.contains(t.pos):
            raise _Rejected("outside volume")
        if t.source == "object" and not t.depth:
            raise _Rejected("no depth")
        if t.source == "object" and self._is_stale(t):
            raise _Rejected("stale", f"wykrycie sprzed {t.age:.2f} s > {self.max_object_age_s:.1f} s")

    def _is_stale(self, t: Target) -> bool:
        """Wykrycie za stare; zepsuty wiek (NaN) tez."""
        return t.age is not None and (math.isnan(t.age) or t.age > self.max_object_age_s)

    def _legs(self, t: Target) -> list[_Leg]:
        """Punkty celu -> stawy (IK) -> odcinki; kazdy sprawdzony na skok i kolizje."""
        kin, checker = self._tools()
        command = self._command()
        self._validated_from = self._arm_joints(command)
        prev = self._validated_from

        legs: list[_Leg] = []
        for wp in self._waypoints(t):
            goal = self._solve(kin, wp, seed={**command, **prev})
            self._check_leg(checker, prev, goal, wp.what)
            legs.append(_Leg(goal, self.speed * wp.speed_factor, wp.profile, wp.what))
            prev = goal

        if t.source == "object" and self.grasp:
            legs.append(self._lift_leg(legs[0].goal))
        return legs

    def _waypoints(self, t: Target) -> list[_Waypoint]:
        if t.source == "hand":
            return [_Waypoint(t.pos, "dlon")]

        # Ramie z mniej niz 6 stawami (SO-101) nie ustawi dowolnego obrotu: IK tylko na pozycje,
        # a os podejscia w dol sprawdzana tylko przed zejsciem (--grasp); samo "nad obiekt" bez niej.
        full_rotation = len(self.arm) >= 6
        rotation = down_rotation(self.spec, t.pos) if full_rotation else None
        hover = _Waypoint(self._hover_point(t.pos), "nad obiekt", rotation,
                          check_down=full_rotation or self.grasp)
        if not self.grasp:
            return [hover]

        touch = _Waypoint(t.pos + np.array([0.0, 0.0, self.touch_dz]), "w dol", rotation,
                          check_down=True, speed_factor=0.5, profile="descend")
        return [hover, touch]

    def _lift_leg(self, hover_goal: Joints) -> _Leg:
        """Po chwycie z powrotem nad obiekt, tym samym torem (juz sprawdzonym), z zamknietym chwytakiem."""
        back = dict(hover_goal)
        gripper = self.spec.gripper
        if gripper:
            back[gripper] = self.grip_closed
        return _Leg(back, self.speed * 0.5, "descend", "w gore", close_gripper=bool(gripper))

    def _solve(self, kin: RobotKinematics, wp: _Waypoint, seed: Joints) -> Joints:
        """IK punktu; stawy ramienia albo `_Rejected("ik")`."""
        sol = kin.ik(wp.pos, rotation=wp.rotation, seed=seed, tol=self.ik_tol)
        if not sol.ok:
            raise _Rejected("ik", f"IK {wp.what}: blad {sol.pos_err * 1000:.1f} mm")

        if wp.check_down:
            approach = kin.tool_axes(sol.joints)[0]
            tilt = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
            if tilt > self.approach_tol:
                raise _Rejected("ik", f"IK {wp.what}: os podejscia {math.degrees(tilt):.0f} st. od pionu")

        return {k: float(v) for k, v in sol.joints.items() if k in self.arm}

    def _check_leg(self, checker: CollisionChecker, start: Joints, goal: Joints, what: str) -> None:
        jump = self._largest_step(start, goal)
        if jump > self.max_jump:
            raise _Rejected("jump", f"skok {what} {jump:.0f} st. > {self.max_jump:.0f}")
        collision = checker.path_clear(start, goal)
        if collision is not None:
            raise _Rejected("collision", f"kolizja {what}: {collision}")

    def _recheck(self, legs: list[_Leg]) -> None:
        """Po wzieciu ramienia: rozkaz mogl sie zmienic od sprawdzenia (suwaki w czasie odczekania).
        Skok i droga do pierwszego celu jeszcze raz, od biezacego rozkazu; inaczej hold "moved"."""
        if not legs:
            return
        start = self._arm_joints(self._command())
        validated = self._validated_from
        if validated is not None and all(abs(v - validated.get(k, math.inf)) < self.unchanged_deg
                                         for k, v in start.items()):
            return

        first = self._arm_joints(legs[0].goal)
        jump = self._largest_step(start, first)
        if jump > self.max_jump:
            raise _Abort("hold", "moved", f"rozkaz zmieniony w czasie odczekania: skok {jump:.0f} st.")
        _, checker = self._tools()
        collision = checker.path_clear(start, first)
        if collision is not None:
            raise _Abort("hold", "moved", f"rozkaz zmieniony w czasie odczekania: kolizja {collision}")

    def _hover_point(self, pos: np.ndarray) -> np.ndarray:
        """Punkt `hover` nad obiektem, ale najwyzej `hover_ceiling_gap` pod sufitem pudelka."""
        p = np.array(pos, float)
        p[2] = min(float(pos[2]) + self.hover, float(self.volume.hi[2]) - self.hover_ceiling_gap)
        return p

    # ------------------------------------------------------------ ramie
    def _take(self) -> int:
        """Ramie dla kierowcy jak w `Twin.move`: czekanie na dom, `claim` z `gen`, sprzeglo."""
        tw = self.twin
        try:
            tw._wait_for_homing()
        except RuntimeError:
            raise _Rejected("busy") from None

        gen = tw.preempt_gen
        try:
            tw.claim(OWNER, preempt=lambda: self._interrupt("preempt"), gen=gen)
        except RuntimeError:
            raise _Rejected("stop" if self._estopped() else "busy") from None
        if not tw.set_engaged(True, owner=OWNER):
            self._release()
            raise _Rejected("stop" if self._estopped() else "busy")
        return gen

    def _release(self) -> None:
        with suppress(Exception):
            self.twin.release(OWNER)

    def _command_arm(self, target: Joints) -> None:
        """`set_target` jako wlasciciel; odmowa = ramie odebrane (STOP albo inny wlasciciel)."""
        try:
            self.twin.set_target(target, owner=OWNER)
        except RuntimeError:
            raise _Abort("hold", self._lost_reason()) from None

    def _check(self, gen: int) -> None:
        tw = self.twin
        if self._estopped():
            raise _Abort("hold", "stop")
        if tw.preempt_gen != gen:
            raise _Abort("hold", "preempted")
        if not tw.connected:
            raise _Abort("hold", "not connected")

    def _close_gripper(self, gen: int) -> None:
        self.twin.set_target({self.spec.gripper: self.grip_closed}, owner=OWNER)
        if self._cancel.wait(self.grip_wait_s):
            return
        if self.twin.preempt_gen != gen:
            raise _Abort("hold", self._lost_reason())

    def _cap_speed(self, deg_s: float) -> None:
        """Granica predkosci nadzoru na czas ruchu kierowcy (nie wyzej niz ustawiona w panelu)."""
        if not self._capped:
            # `Twin` nie ma publicznego odczytu granicy; atrapy w testach nie maja jej wcale.
            self._prev_cap = getattr(self.twin, "_speed_cap", None)
            self._capped = True
        panel_cap = self._prev_cap if self._prev_cap is not None else math.inf
        self.twin.set_speed(min(panel_cap, float(deg_s)))

    def _restore_speed(self) -> None:
        if not self._capped:
            return
        self._capped = False
        try:
            self.twin.set_speed(self._prev_cap)
        except Exception:
            logger.exception("Przywracanie granicy predkosci")

    def _tools(self) -> tuple[RobotKinematics, CollisionChecker]:
        """IK i sprawdzacz kolizji na scenie stanowiska BEZ dodatkow (obiekty zadania, karta).

        Budowane w watku wykonawcy, od nowa po przebudowie sceny blizniaka (`twin.version`).
        """
        version = self.twin.version
        if self._checker is None or self._checker_version != version:
            scene = sc.build(self.twin.workspace.scene_config())
            self._checker = CollisionChecker(scene)
            self._kin = RobotKinematics(self.spec, scene.model, sc.PREFIX)
            self._checker_version = version
        assert self._kin is not None
        return self._kin, self._checker

    def _estopped(self) -> bool:
        from ..control.safety import SafetyState

        return self.twin.safety_state == SafetyState.ESTOP

    def _command(self) -> Joints:
        tw = self.twin
        return {k: float(v) for k, v in (tw.status.command or tw.joints()).items()}

    def _arm_joints(self, joints: Mapping[str, float]) -> Joints:
        """Tylko stawy ramienia (bez chwytaka)."""
        return {k: float(joints[k]) for k in self.arm if k in joints}

    def _largest_step(self, start: Mapping[str, float], goal: Mapping[str, float]) -> float:
        """Najwiekszy ruch stawu ramienia miedzy pozami [st.]."""
        return max((abs(v - start.get(k, v)) for k, v in goal.items() if k in self.arm), default=0.0)

    # ------------------------------------------------------------ start nad srodek
    def center_pose(self) -> Joints | None:
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
        steps, collision = self._stage_plan(self._arm_joints(self._command()), goal, checker)
        if collision is not None:
            logger.warning("Start nad srodek pudelka: %s", collision)
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
                self._guard.lag_deg = self.center_lag_deg
                if not self._ramp(step, speed_deg_s, gen, arrive_lag_s=self.center_arrive_lag_s):
                    return self._center_interrupted()
            return None
        except _Abort as exc:
            return exc.reason
        finally:
            self._guard = None
            self._phase = "idle"
            self._release()
            self._restore_speed()

    def _center_interrupted(self) -> str:
        cause = self._cancel_cause
        self.twin.hold_measured()
        hit, self._impact = self._impact, None
        if cause == "impact" and hit is not None:
            logger.warning("Start nad srodek pudelka: uderzenie %s %s - trzymanie", hit.kind, hit.joint)
            return f"impact {hit.kind} {hit.joint}"
        return cause or "preempted"

    def _stage_plan(self, start: Joints, goal: Joints,
                    checker: CollisionChecker) -> tuple[list[Joints], str | None]:
        """Pozy posrednie jak w domu etapami: grupy stawow po kolei (tez odwrotnie), a na koniec
        caly cel; pierwszy plan z czystymi drogami wygrywa, bez etapow - prosto.
        Zwraca (pozy, None) albo ([], opis pierwszej kolizji)."""
        stages = [tuple(j for j in stage if j in self.arm) for stage in self.spec.home_stages]
        stages = [stage for stage in stages if stage]
        orders = [stages, stages[::-1]] if stages else []

        first_collision: str | None = None
        for order in [*orders, []]:
            poses, collision = self._walk_stages(start, goal, [*order, self.arm], checker)
            if collision is None:
                return poses, None
            first_collision = first_collision or collision
        return [], first_collision

    def _walk_stages(self, start: Joints, goal: Joints, stages: list[tuple[str, ...]],
                     checker: CollisionChecker) -> tuple[list[Joints], str | None]:
        """Etap po etapie stawy etapu jada do celu; pierwsza kolizja przerywa."""
        pose = dict(start)
        poses: list[Joints] = []
        for stage in stages:
            nxt = {**pose, **{j: goal[j] for j in stage}}
            if self._largest_step(pose, nxt) < 1e-3:
                continue
            collision = checker.path_clear(pose, nxt)
            if collision is not None:
                return poses, collision
            poses.append(nxt)
            pose = nxt
        return poses, None

    # ------------------------------------------------------------ rampa
    def _ramp(self, goal: Mapping[str, float], speed: float, gen: int, *, guarded: bool = True,
              profile: Profile = "normal", arrive_lag_s: float | None = None) -> bool:
        """Rampa smoothstep do `goal` i czekanie na dojechanie. True = dojechal, False = przerwana
        (`self._cause`); `_Abort` - ramie odebrane albo nie dojechalo (wtedy trzyma zmierzona poze).
        `arrive_lag_s` - czas reguly opoznienia przy dojezdzaniu (domyslnie `arrive_lag_s`)."""
        command = self._command()
        start = {k: command.get(k, float(v)) for k, v in goal.items()}
        goal = {k: float(v) for k, v in goal.items()}
        duration = max(0.5, self._largest_step(start, goal) / max(speed, 1e-3))
        with self._listening(guarded):
            self._phase = "moving" if guarded else "backoff"
            if not self._sweep(start, goal, duration, gen):
                return False
            self._tighten_guard_for_arrival(arrive_lag_s)
            return self._settle(goal, gen, profile)

    def _sweep(self, start: Joints, goal: Joints, duration: float, gen: int) -> bool:
        """Rozkaz od `start` do `goal` po krzywej smoothstep, co `tick_s`. False = przerwana."""
        clock = _LoopClock(self.twin)
        while True:
            if self._cancel.is_set():
                return False
            self._check(gen)

            s = smoothstep(clock.elapsed() / duration)
            self._command_arm({k: start[k] + s * (goal[k] - start[k]) for k in goal})
            if s >= 1.0:
                return True

            if self._cancel.wait(self.tick_s):
                return False

    def _tighten_guard_for_arrival(self, arrive_lag_s: float | None) -> None:
        """Staw zablokowany tuz przy celu: wycofanie, zanim backend uzna go za "gluchy"."""
        guard = self._guard
        if guard is None:
            return
        guard.lag_deg = self.arrive_lag_deg
        guard.lag_s = self.arrive_lag_s if arrive_lag_s is None else float(arrive_lag_s)

    def _settle(self, goal: Joints, gen: int, profile: Profile) -> bool:
        """Czekanie, az kazdy staw ramienia bedzie najwyzej `arrive_deg` od celu i stanie.
        Po `arrive_timeout_s` bez tego: trzymanie zmierzonej pozy i `_Abort("hold", "not arrived")`."""
        deadline = time.monotonic() + self.arrive_timeout_s
        previous: _Sample | None = None
        while True:
            if self._cancel.wait(0.05):
                return False
            self._check(gen)

            status = self.twin.status
            sample = _Sample(status.measured_t, dict(status.measured))
            still = False
            if previous is None or sample.t > previous.t:
                still = previous is not None and self._stood_still(previous, sample)
                previous = sample

            far = self._far_joints(goal, sample.joints)
            if not far and still:
                if profile == "descend":
                    return True                         # przy zejsciu bez docinania: docisk do obiektu
                return self._trim(goal, gen)

            if time.monotonic() > deadline:
                # Staw zablokowany nie naciska dalej na przeszkode celem, ktorego nie osiagnal.
                self.twin.hold_measured()
                misses = ", ".join(f"{k} {sample.joints.get(k, math.nan) - goal[k]:+.1f} st." for k in far)
                raise _Abort("hold", "not arrived", misses)

    def _far_joints(self, goal: Joints, measured: Joints) -> list[str]:
        return [k for k in goal if k in self.arm and abs(measured.get(k, goal[k]) - goal[k]) > self.arrive_deg]

    def _stood_still(self, before: _Sample, after: _Sample) -> bool:
        dt = after.t - before.t
        return all(abs(v - before.joints.get(k, v)) / dt < self.still_deg_s
                   for k, v in after.joints.items() if k in self.arm)

    def _trim(self, goal: Joints, gen: int) -> bool:
        """Rozkaz = cel + uchyb (pomiar ma trafic w cel IK), kilka krokow. False = przerwane."""
        guard = self._guard
        if guard is not None and guard.effort_abs:
            loaded = self._loaded_joints(self._efforts() or {}, guard.effort_abs)
            if loaded:
                logger.info("Bez docinania: wysilek %s ponad polowe progu", loaded)
                return True

        command = dict(goal)
        for _ in range(self.trim_steps):
            measured = dict(self.twin.status.measured)
            error = {k: goal[k] - measured[k] for k in goal if k in self.arm and k in measured}
            if not error or max(abs(v) for v in error.values()) < self.trim_deg:
                break
            for k, v in error.items():
                command[k] = float(np.clip(command[k] + v, goal[k] - self.trim_max_deg, goal[k] + self.trim_max_deg))
            self._command_arm(command)
            if self._cancel.wait(0.4):
                return False
            self._check(gen)
        return True

    # ------------------------------------------------------------ uderzenie
    @contextmanager
    def _listening(self, enabled: bool = True) -> Iterator[None]:
        """Straznik uderzen slucha taktow petli blizniaka na czas bloku."""
        handle = self.twin.add_tick_listener(self._on_tick) if enabled else None
        try:
            yield
        finally:
            if handle is not None:
                self.twin.remove_tick_listener(handle)

    def _on_tick(self, sample) -> None:
        """Sluchacz taktow (watek petli): tylko zapis uderzenia i przerwanie rampy."""
        guard = self._guard
        if guard is None or self._cancel.is_set() or time.monotonic() < self._guard_from:
            return
        hit = guard.feed(sample.t, sample.sent, sample.measured, sample.fresh,
                         efforts=self._efforts(), velocities=self._velocities())
        if hit is not None:
            self._impact = hit
            self._interrupt("impact")

    def _tune_guard(self, guard: ImpactGuard, profile: Profile) -> None:
        """Progi straznika dla odcinka: `descend` ciasniejszy (opoznienie 3 st., mniejszy wysilek).
        Jeden straznik na caly cel - bufor dobrych probek (poza wycofania) przechodzi przez odcinki."""
        descend = profile == "descend"
        guard.lag_deg = self.lag_deg_descend if descend else self.lag_deg
        guard.lag_s = self.lag_s

        limits = self._effort_limits(descend) or {}
        # Staw, ktory juz w spoczynku ma wysilek ponad polowe progu (ciezar, sprezyna), wyzwalalby
        # przy kazdym ruchu: dla niego w tym odcinku tylko regula opoznienia.
        at_rest = (self._log_efforts("start odcinka") or {}) if limits else {}
        for joint in self._loaded_joints(at_rest, limits):
            logger.warning("Wysilek %s w spoczynku %.1f > %.0f%% progu %.1f - regula wysilku wylaczona "
                           "dla tego stawu w tym ruchu", joint, at_rest[joint], 100 * self.effort_rest_frac,
                           limits[joint])
            del limits[joint]
        guard.effort_abs = limits

    def _loaded_joints(self, efforts: Mapping[str, float], limits: Mapping[str, float]) -> list[str]:
        """Stawy z wysilkiem ponad `effort_rest_frac` progu."""
        return [j for j, limit in limits.items() if j in efforts and abs(efforts[j]) > self.effort_rest_frac * limit]

    def _backoff(self, t: Target) -> None:
        """Po uderzeniu: trzymanie zmierzonej pozy, potem wolno do pozy wycofania.
        Drugie uderzenie w trakcie wycofania = tylko trzymanie ("hold")."""
        tw = self.twin
        hit, self._impact = self._impact, None
        # Czysty pomiar, bez drogi hamowania: staw oparty o przeszkode przestaje na nia naciskac.
        tw.hold_measured(dict(tw.status.measured) or None)

        why = f"impact {hit.kind} {hit.joint}" if hit is not None else "impact"
        logger.warning("Uderzenie: %s (%.1f) - wycofanie", why, hit.value if hit is not None else math.nan)
        pose = dict(hit.backoff_pose or {}) if hit is not None else {}
        gen = tw.preempt_gen
        if not pose or tw.owner != OWNER:
            self._finish(t, "hold", why)
            return

        self._reset_cancel()
        self._cap_speed(self.backoff_cap_deg_s)
        try:
            arrived = self._retreat(pose, gen)
        except _Abort as exc:
            self._finish(t, "hold", f"{why}; {exc.reason}", exc.detail)
            return

        cause = self._cancel_cause
        if arrived:
            self._report(t, "backoff", why)
        elif cause == "impact":
            tw.hold_measured()                          # drugie uderzenie: bez kolejnego wycofania
            self._impact = None
            self._report(t, "hold", why)
        else:
            self._report(t, "hold", cause)
        self._reject_queued()
        self._let_go()

    def _retreat(self, pose: Joints, gen: int) -> bool:
        """Rampa do pozy wycofania pod nowym straznikiem (z opoznionym startem). False = przerwana."""
        self._guard = ImpactGuard(self.arm, effort_abs=self._effort_limits(False))
        self._guard_from = time.monotonic() + self.backoff_grace_s
        try:
            with self._listening():
                if self._cancel.wait(0.06):             # rampa od rozkazu PO trzymaniu (nastepny takt petli)
                    raise _Abort("hold", self._cancel_cause or "preempted")
                return self._ramp(pose, self.backoff_speed, gen, guarded=False)
        finally:
            self._guard = None

    # ------------------------------------------------------------ wysilek
    def _effort_limits(self, descend: bool) -> Joints | None:
        """Progi reguly wysilku na staw; {} - bez reguly wysilku, None - backend nie mierzy wysilku."""
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated:
            return self._sim_effort_limits(descend)
        try:
            probe = b.efforts()
        except Exception:
            probe = None
        if probe is None:
            return None                                 # tylko regula opoznienia
        limit = self.real_effort_abs_descend if descend else self.real_effort_abs
        return dict.fromkeys(self.arm, limit)

    def _sim_effort_limits(self, descend: bool) -> Joints:
        model = self.twin.scene.model
        ids = [model.actuator(sc.PREFIX + j).id for j in self.arm]
        self._effort_ids = np.array(ids)
        if self.spec.name in self.sim_effort_off:
            return {}                                   # naped nasycony w zwyklym ruchu - tylko opoznienie

        frac = self.sim_effort_frac_descend if descend else self.sim_effort_frac
        ranges = model.actuator_forcerange[ids]
        limited = model.actuator_forcelimited[ids]
        return {j: frac * float(max(abs(lo), abs(hi)))
                for j, (lo, hi), is_limited in zip(self.arm, ranges, limited, strict=True)
                if is_limited and hi > lo}

    def _efforts(self) -> Joints | None:
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
            forces = tw.scene.data.actuator_force[ids]
            return {j: float(f) for j, f in zip(self.arm, forces, strict=True)}
        finally:
            tw.lock.release()

    def _velocities(self) -> dict[str, float] | None:
        """Predkosci stawow z prawdziwego backendu (symulacja - None)."""
        b = self.twin.backend
        if b is None or b.info.simulated:
            return None
        try:
            return b.velocities()
        except Exception:
            return None

    def _log_efforts(self, what: str) -> Joints | None:
        """Wysilek stawow ramienia do logu (progi straznika na A1X sa zgadywane - to daje liczby)."""
        b = self.twin.backend
        if b is None:
            return None
        if b.info.simulated and self._effort_ids is None:
            self._effort_limits(False)                  # ustawia aktuatory ramienia
        efforts = self._efforts()
        if efforts:
            logger.info("Wysilek stawow (%s): %s", what, {k: round(float(v), 2) for k, v in efforts.items()})
        return efforts

