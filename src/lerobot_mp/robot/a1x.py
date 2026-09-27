"""Galaxea A1X po CAN-FD - backend `a1x` blizniaka.

Protokol (ramki 0x050-0x055, skale, kodery, dekoder) i lacze CAN (SocketCAN,
XCAN na macOS, python-can) sa w pakiecie `galaxeo` (repozytorium
galaxeo-manipulators): jedno zrodlo prawdy dla narzedzi galaxeo i dla blizniaka.
Importujemy go leniwie - blizniak dziala bez niego, dopoki nikt nie wybierze `a1x`.

Jednostki: jak w calej aplikacji - stawy ramienia w stopniach, chwytak 0..100
(0 = zamkniety). Przeliczenie aplikacja <-> CAN jest tylko tutaj:

* staw: ``rad = radians(st. + a1x.joint_offsets_deg[i])`` (przesuniecie zera, J3 ~ +1,49 st.),
* chwytak: p_des liniowo ``closed + v/100 * (open - closed)`` (bardziej dodatnie = ciasniej);
  pomiar z grupy 7 ramki 0x052 przez dwa punkty `gripper_fb_closed/open`, gdy zmierzone,
  inaczej petla otwarta (ostatni rozkaz; ostrzezenie w `calibration_warnings`). Przed pierwszym
  rozkazem petla otwarta zglasza `gripper_start` (0 = zamkniety), a "Ustaw dom = tu" blizniaka
  potwierdza to przez `assume_gripper(0)` - umowa stanowiska: dom = szczeki zamkniete. Dopoki
  rozwarcie jest tylko zgadniete, pierwsze otwieranie idzie od razu na cel, zamykanie rampa.
* Chwytak przez XCAN (macOS) nie rusza: przejsciowka wysyla 10-bajtowa 0x051 jako 12 bajtow,
  a chwytak ja ignoruje (galaxeo `bus.py`). Tylko SocketCAN (Linux) - `gripper_drivable()`.

Zasady bezpieczenstwa (galaxeo docs/SAFETY.md, docs/PROTOCOL.md):

* Nic nie idzie na magistrale przed pierwszym `send_joints` - niesterowane A1X trzyma
  poze, w ktorej stoi; blizniak tez nie wysyla nic do pierwszego ruchu.
* Strumien p_des startuje od ZMIERZONEJ pozy i jedzie najwyzej `max_step_deg_s`,
  wiec zly pierwszy rozkaz nie skacze. Strumien leci z `stream_hz` i powtarza ostatni cel.
* Stara telemetria (> `stale_ms`) albo zamrozona (FF 2) = wysylka stoi, `link_silent`
  i blad w `faults()` - blizniak robi STOP i startuje potem od pomiaru. Cele ramienia
  i chwytaka przepadaja: po powrocie ramek nic nie idzie do nastepnego `send_joints`.
* Nigdy FF 2 (zwolnienie: ramie bez hamulcow opada), nigdy kp 0 (ramie gluche),
  nigdy heartbeat 0x023. Wlaczenie FF 1 -> 5 -> 6 tylko na zyczenie (`enable`),
  z p_des = pomiar przed, miedzy i po kazdym kodzie.
* `disconnect` konczy strumien i zamyka lacze - silnikow nie zwalnia.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

from ..config import GRIPPER, AppConfig
from .base import RobotBackend, RobotInfo

logger = logging.getLogger(__name__)

#: Rozkaz chwytaka [jednostki aplikacji] rozny od zgloszonego o wiecej niz tyle
#: uruchamia strumien 0x051 - utrzymanie zgloszonej wartosci po Polacz nic nie wysyla.
GRIP_START_DELTA = 0.5
#: Staw "stoi" przy |predkosci| ponizej tego progu [st./s] (wykrywanie gluchego ramienia).
STILL_DEG_S = 0.3
#: Ile czekac przy Polacz na pierwsza ramke 0x052 [s].
CONNECT_TIMEOUT_S = 3.0
#: Ile po pierwszej ramce zbierac kolejne, zeby rozpoznac zamrozona telemetrie [s] (co najmniej;
#: przy duzym `frozen_frames` okno rosnie do dwoch okresow 0x052 na ramke).
SETTLE_S = 0.4
#: Okres ramek 0x052 [s] (ramie odpowiada z 200 Hz).
FB_PERIOD_S = 0.005
#: Czas miedzy kodami sekwencji wlaczenia [s] - caly czas strumien p_des = pomiar.
ENABLE_STEP_S = 0.3
#: Spoczynkowy |effort| chwytaka: mediana z tylu taktow strumienia na poczatku kazdego zamykania
#: (0,1 s przy 200 Hz); w tym oknie cel nie zamarza. Kontakt = |effort| > spoczynek + `gripper_force`.
GRIP_REST_STEPS = 20
#: Otwarcie [p_des] od najciasniejszego punktu zamykania, po ktorym nastepne zamykanie mierzy
#: spoczynek od nowa. Mniejsze uchylenie (szczeki dalej na kostce) zostawia stary spoczynek.
GRIP_REOPEN_PDES = 0.3
#: Zapas na czas pelnego skoku chwytaka [s] (`gripper_travel_s`).
GRIP_TRAVEL_MARGIN_S = 0.2

INSTALL_HINT = 'pip install -e ".[a1x]"  (pakiet galaxeo z repozytorium galaxeo-manipulators)'


def _galaxeo():
    """(protocol, bus) z pakietu galaxeo; ImportError z komenda instalacji, gdy go brak."""
    try:
        from galaxeo import bus, protocol
    except ImportError as exc:
        raise ImportError(f"backend a1x potrzebuje pakietu galaxeo: {INSTALL_HINT} ({exc})") from exc
    return protocol, bus


def open_bus(iface: str) -> Any:
    """Lacze CAN z galaxeo (`can0`, `xcan`, `xcan:6`, `PCAN_USBBUS1`, `virtual:x`)."""
    _protocol, bus = _galaxeo()
    try:
        return bus.open_bus(iface)
    except ImportError as exc:                 # pyusb / python-can
        raise ImportError(f"lacze {iface!r}: {exc}. Pelny zestaw: {INSTALL_HINT}") from exc


def default_iface() -> str:
    """can0; na macOS xcan (tam nie ma SocketCAN)."""
    return _galaxeo()[1].default_iface()


class A1XArm(RobotBackend):
    """Galaxea A1X: 6 stawow ramienia + chwytak po CAN-FD (protokol z galaxeo).

    `threaded=False` (testy): bez watku IO - takty wymiany robi `io_step(now)`,
    a `enable()` odmierza czas przez `sleep` i sam wola `io_step`.
    """

    def __init__(self, cfg: AppConfig, *, bus: Any = None, clock: Callable[[], float] = time.monotonic,
                 threaded: bool = True, sleep: Callable[[float], None] = time.sleep,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S):
        self.P, self._B = _galaxeo()
        self.cfg = cfg                         # te same JointConfig co nadzor blizniaka (`_sync_backend_limits`)
        self.acfg = cfg.a1x
        self.arm = tuple(n for n in cfg.joints if n != GRIPPER)
        if len(self.arm) != self.P.N_JOINTS or GRIPPER not in cfg.joints:
            raise ValueError("backend a1x potrzebuje 6 stawow ramienia + 'gripper' w sekcji joints - "
                             "stanowisko robot: a1x")
        self.iface = cfg.robot.port or None
        self._bus = bus
        self._own_bus = bus is None
        self._clock, self._sleep, self._threaded = clock, sleep, threaded
        self._connect_timeout = connect_timeout_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._off = tuple(math.radians(x) for x in self.acfg.joint_offsets_deg)
        self._reset_state()
        self.info = RobotInfo(name="Galaxea A1X (CAN)", description=f"lacze {self.iface or 'CAN'}",
                              simulated=False)

    def _reset_state(self) -> None:
        self._fb = None                        # ostatnia zdekodowana ramka 0x052 (galaxeo Feedback)
        self._fb_t = -math.inf                 # czas jej odbioru (zegar backendu)
        self._fb_raw: bytes | None = None
        self._identical = 0                    # kolejne identyczne ramki (zamrozona telemetria)
        self._n_fb = 0
        self._start_rad: tuple[float, ...] | None = None
        self._bounds: tuple[tuple[float, float], ...] = tuple(self.P.URDF_LIMITS_RAD)
        self._p_target: list[float] | None = None    # cel ramienia [rad CAN]; None = nic nie wysylamy
        self._p_stream: list[float] | None = None    # to, co naprawde idzie w 0x050 (slew od pomiaru)
        self._g_target: float | None = None          # cel p_des chwytaka; None = 0x051 nie idzie
        self._g_stream: float | None = None
        self._g_frozen = False                       # kontakt przy zamykaniu - cel stoi
        self._g_cmd: float | None = None             # ostatni rozkaz chwytaka [0..100] albo `assume_gripper`
        self._g_assumed = False                      # `_g_cmd` z `assume_gripper`, strumien jeszcze nie ruszyl
        self._g_rest_eff: list[float] = []           # |effort| z poczatku zamykania (spoczynek)
        self._g_rest: float | None = None            # ich mediana; None = zamykanie nie trwa albo okno zbiera
        self._g_hi: float | None = None              # najciasniejszy p_des tego zamykania (odniesienie otwarcia)
        self._goal: dict[str, float] = {}            # ostatnie cele w jednostkach aplikacji
        self._pinned = False                         # sekwencja wlaczenia: p_des = pomiar
        self._last_io: float | None = None
        self._deaf_since: dict[int, float] = {}
        self._deaf: str = ""
        self._bus_error: str = ""
        self._link_silent = False
        self._stale: list[str] = []
        self.tx_frames = 0

    # ------------------------------------------------------------ przeliczenia
    def _pdes(self, v: float) -> float:
        """Chwytak 0..100 -> p_des (liniowo miedzy zamknietym i otwartym)."""
        a = self.acfg
        return a.gripper_closed_pdes + v / 100.0 * (a.gripper_open_pdes - a.gripper_closed_pdes)

    def _grip_app(self) -> float:
        """Chwytak w 0..100: pomiar przez dwa punkty albo (bez nich) ostatni rozkaz / zalozenie."""
        a = self.acfg
        if a.gripper_fb_closed is not None and a.gripper_fb_open is not None and self._fb is not None:
            span = a.gripper_fb_open - a.gripper_fb_closed
            return (self._fb.pos[6] - a.gripper_fb_closed) * 100.0 / span
        return a.gripper_start if self._g_cmd is None else self._g_cmd

    def _measured_rad(self) -> list[float]:
        assert self._fb is not None
        return list(self._fb.pos[:6])

    # ---------------------------------------------------------------- polaczenie
    def connect(self) -> None:
        if self._connected:
            return
        self._reset_state()
        if self._bus is None:
            self.iface = self.iface or default_iface()
            self.info.description = f"lacze {self.iface}"
            self._bus = open_bus(self.iface)
        try:
            self._await_feedback()
        except Exception:
            self._close_bus()
            raise
        self._start_rad = tuple(self._fb.pos[:6])
        self._bounds = tuple((min(lo, s), max(hi, s))
                             for (lo, hi), s in zip(self.P.URDF_LIMITS_RAD, self._start_rad))
        self._link_silent, self._stale = False, []
        self._connected = True
        logger.info("A1X na %s: poza startowa %s", self.iface or "podanym laczu",
                    ", ".join(f"{n}={v:.1f}" for n, v in self.read_joints().items()))
        self._start_thread()
        if self.acfg.enable_on_connect:
            self.enable()

    def _await_feedback(self) -> None:
        """Pierwsza ramka 0x052 w `connect_timeout_s`, potem chwila zbierania na test zamrozenia."""
        where = self.iface or "laczu"
        t0 = time.monotonic()
        while self._n_fb == 0:
            self._drain(self._clock())
            if self._n_fb:
                break
            if time.monotonic() - t0 > self._connect_timeout:
                raise RuntimeError(
                    f"brak ramek 0x052 na {where} - lacze CAN stoi (Linux: ./can_up.sh albo "
                    + self._B.IP_LINK_UP.format(iface=where) + ") albo ramie bez zasilania")
            time.sleep(0.002)
        # `_identical` liczy POWTORZENIA, wiec z n ramek wychodzi najwyzej n - 1: werdykt "zamrozona"
        # (> frozen_frames) wymaga frozen_frames + 2 ramek. Na sprzecie przychodza pojedynczo (co 5 ms).
        frozen = self.acfg.frozen_frames
        settle = max(SETTLE_S, 2.0 * (frozen + 2) * FB_PERIOD_S)
        t1 = time.monotonic()
        while self._n_fb < frozen + 2 and time.monotonic() - t1 < settle:
            if not self._drain(self._clock()):
                time.sleep(0.002)
        if self._identical > frozen:
            raise RuntimeError("telemetria zamrozona - ramie zwolnione (FF 2), wylacz i wlacz zasilanie")

    def _start_thread(self) -> None:
        if not self._threaded:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="a1x-can", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        """Konczy strumien i zamyka lacze. Silnikow NIE zwalnia (FF 2): A1X nie ma hamulcow."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._connected and self.cfg.robot.torque_off_on_exit:
            logger.warning("A1X nie ma hamulcow - momentu nie wylaczam (robot.torque_off_on_exit ignorowane)")
        with self._lock:
            self._p_target = self._g_target = None
            self._connected = False
        self._close_bus()

    def _close_bus(self) -> None:
        bus = self._bus
        if bus is None:
            return
        if self._own_bus:
            self._bus = None                   # nastepne connect otwiera lacze od nowa
        try:
            bus.close()
        except Exception:                      # pragma: no cover - rozlaczenie nie moze wysypac aplikacji
            logger.exception("Blad przy zamykaniu lacza CAN")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---------------------------------------------------------------- petla IO
    def _loop(self) -> None:
        period = 1.0 / self.acfg.stream_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.io_step(self._clock())
                self._bus_error = ""
            except Exception as exc:           # adapter wyjety, lacze padlo - blad dla blizniaka
                if not self._bus_error:
                    logger.error("A1X: blad lacza CAN: %s", exc)
                self._bus_error = str(exc)
                self._stop.wait(0.1)
                continue
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -0.05:
                nxt = time.monotonic()         # spoznienie - bez serii nadrabiajacej
            elif delay > 0:
                self._stop.wait(delay)

    def _drain(self, now: float) -> bool:
        """Odbiera KAZDA ramke z kolejki (czesciowe czytanie daje odczyty sprzed sekund)."""
        got = False
        P = self.P
        while True:
            f = self._bus.recv(0.0)
            if f is None:
                return got
            if f.can_id != P.FB_ID or len(f.data) != P.FB_LEN:
                continue
            data = bytes(f.data)
            self._fb = P.decode_feedback(data)
            self._identical = self._identical + 1 if data == self._fb_raw else 0
            self._fb_raw, self._fb_t = data, now
            self._n_fb += 1
            got = True

    def _stale_now(self, now: float) -> bool:
        return (now - self._fb_t) * 1000.0 > self.acfg.stale_ms

    def _frozen(self) -> bool:
        return self._identical > self.acfg.frozen_frames

    def io_step(self, now: float) -> None:
        """Jeden takt wymiany: odbior wszystkich ramek, potem (gdy wolno) 0x050 i 0x051."""
        with self._lock:
            if self._bus is None:
                return
            self._drain(now)
            dt = (1.0 / self.acfg.stream_hz if self._last_io is None
                  else min(max(now - self._last_io, 0.0), 2.0 / self.acfg.stream_hz))
            self._last_io = now
            if self._fb is None or self._stale_now(now) or self._frozen():
                # Stara albo zamrozona telemetria: nic nie wysylamy i cele przepadaja. Blizniak robi
                # STOP i rusza dopiero od potwierdzonego pomiaru - backend nie moze w tym czasie sam
                # jechac do celu sprzed przerwy (ramie po wylacz/wlacz albo zablokowane stoi gdzie
                # indziej, a A1X nie ma hamulcow). Po powrocie ramek nic nie idzie do nastepnego
                # `send_joints`: ramie trzyma poze, w ktorej stoi, chwytak ostatni p_des.
                self._p_target = self._p_stream = None
                self._g_target = self._g_stream = None
                self._g_frozen = False
                self._g_rest_eff, self._g_rest, self._g_hi = [], None, None
                self._deaf_since, self._deaf = {}, ""
                return
            if self._pinned:
                self._p_target = self._measured_rad()
                self._p_stream = list(self._p_target)
            if self._p_target is not None:
                if self._p_stream is None:
                    self._p_stream = self._measured_rad()     # start od POMIARU, nie od celu
                step = math.radians(self.acfg.max_step_deg_s) * dt
                self._p_stream = [s + max(-step, min(step, t - s)) for s, t in zip(self._p_stream, self._p_target)]
                self._send(self.P.CMD_ID, self.P.encode_arm(self._p_stream, self.acfg.kp, self.acfg.kd))
                self._watch_deaf(now)
            if self._g_target is not None:
                if self._g_stream is None:
                    self._g_stream = self._pdes(self._grip_app())
                closing = self._g_target > self._g_stream
                if closing:
                    self._g_hi = self._g_stream if self._g_hi is None else max(self._g_hi, self._g_stream)
                if self._g_target < self._g_stream:
                    # Otwieranie: spoczynek mierzy sie od nowa dopiero, gdy szczeki odeszly o GRIP_REOPEN_PDES
                    # od najciasniejszego punktu zamykania. Po uchyleniu o jedna kreske suwaka szczeki dalej
                    # dociskaja kostke, a nowy "spoczynek" bylby dociskiem - prog rosl z kazdym cyklem.
                    # Cel osiagniety (albo zamrozony) spoczynku nie zeruje - inaczej suwak zamykany malymi
                    # krokami tez bralby docisk za spoczynek.
                    hi = self._g_hi
                    if self._g_rest is None or hi is None or self._g_stream < hi - GRIP_REOPEN_PDES:
                        self._g_rest_eff, self._g_rest, self._g_hi = [], None, None
                elif closing and self._g_rest is None:
                    # Poczatek zamykania: spoczynkowy effort grupy 7 (nieznany, bywa staly ~3) z pierwszych
                    # taktow. Staly prog 1,2 zamrazal cel juz w pierwszym takcie i chwytak przestawal sluchac.
                    self._g_rest_eff.append(abs(self._fb.eff[6]))
                    if len(self._g_rest_eff) >= GRIP_REST_STEPS:
                        e = sorted(self._g_rest_eff)
                        self._g_rest = e[len(e) // 2]
                elif closing and self._grip_contact():
                    # Kontakt: cel staje tam, gdzie strumien - chwyta, nie miazdzy.
                    self._g_target, self._g_frozen = self._g_stream, True
                step = self.acfg.gripper_rate_pdes_s * dt
                self._g_stream += max(-step, min(step, self._g_target - self._g_stream))
                self._send(self.P.GRIP_ID, self.P.encode_gripper(self._g_stream, self.acfg.gripper_kp, self.acfg.kd))

    def _grip_contact(self) -> bool:
        """|effort| chwytaka ponad spoczynek zmierzony na poczatku tego zamykania o wiecej niz `gripper_force`."""
        return self._g_rest is not None and abs(self._fb.eff[6]) > self._g_rest + self.acfg.gripper_force

    def _send(self, can_id: int, data: bytes, fd: bool = True) -> None:
        self._bus.send(can_id, data, fd=fd)
        self.tx_frames += 1

    def _watch_deaf(self, now: float) -> None:
        """Rozkaz daleko od pomiaru, a staw stoi przez `deaf_s` = ramie gluche na 0x050 albo zablokowane."""
        a, fb = self.acfg, self._fb
        worst = ""
        for i, name in enumerate(self.arm):
            err = math.degrees(abs(self._p_stream[i] - fb.pos[i]))
            if err > a.deaf_err_deg and abs(math.degrees(fb.vel[i])) < STILL_DEG_S:
                t0 = self._deaf_since.setdefault(i, now)
                if now - t0 >= a.deaf_s and not worst:
                    worst = (f"{name}: {err:.0f} st. od rozkazu przez {now - t0:.1f} s i stoi - ramie gluche na "
                             f"0x050 (Wlacz silniki A1X, FF 1-5-6) albo zablokowane")
            else:
                self._deaf_since.pop(i, None)
        self._deaf = worst

    # ------------------------------------------------------------------ wymiana
    def read_joints(self) -> dict[str, float]:
        """Stopnie ramienia (minus przesuniecie zera) + chwytak 0..100. Przy ciszy - ostatnie wartosci."""
        with self._lock:
            if self._fb is None:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            now = self._clock()
            self._link_silent = self._stale_now(now) or self._frozen()
            self._stale = list(self.arm) if self._link_silent else []
            out = {name: math.degrees(self._fb.pos[i] - self._off[i]) for i, name in enumerate(self.arm)}
            out[GRIPPER] = self._grip_app()
            return out

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Cel ramienia i chwytaka. Zwraca przyciety ROZKAZ w jednostkach aplikacji.

        Chwytak: zwracamy rozkaz, a nie zamrozony przy kontakcie p_des - nadzor blizniaka musi
        dalej widziec rozkaz ciasniejszy od pomiaru, inaczej nie rozpozna chwytu (`_grip_squeezing`,
        "w dloni" w sledzeniu kostki). `{}` przy milczacym laczu i w trakcie sekwencji wlaczenia.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            if self._link_silent or self._pinned or self._fb is None:
                return {}
            sent: dict[str, float] = {}
            arm = [n for n in self.arm if n in targets]
            if arm:
                p = self._p_target if self._p_target is not None else self._measured_rad()
                p = list(p)
                for i, name in enumerate(self.arm):
                    if name not in targets:
                        continue
                    jc = self.cfg.joint(name)
                    lo_b, hi_b = (math.degrees(b - self._off[i]) for b in self._bounds[i])
                    v = min(max(float(targets[name]), jc.min, lo_b), jc.max, hi_b)
                    p[i] = math.radians(v) + self._off[i]
                    sent[name] = v
                self._p_target = p
            if GRIPPER in targets:
                jc = self.cfg.joint(GRIPPER)
                v = min(max(float(targets[GRIPPER]), jc.min, 0.0), jc.max, 100.0)
                reported = self._grip_app()
                if self._g_target is not None or abs(v - reported) > GRIP_START_DELTA:
                    want = self._pdes(v)
                    if self._g_target is None:
                        logger.info("A1X chwytak: pierwszy rozkaz %.0f (zglaszane %.0f%s) -> p_des %.2f", v, reported,
                                    "" if self.gripper_measured() else ", bez pomiaru", want)
                        # Pierwszy rozkaz: strumien od ZGLOSZONEGO rozwarcia, nie od celu.
                        stream = self._pdes(reported)
                        if self._g_assumed or (self._g_cmd is None and not self.gripper_measured()):
                            # Rozwarcie tylko zgadniete: `assume_gripper` ("Ustaw dom = tu": zamkniete)
                            # albo sam `gripper_start` po Polacz (A1X trzyma ostatnia poze miedzy sesjami,
                            # szczeki moga byc otwarte). Start od bardziej otwartego z {zgadniete, cel}:
                            # otwieranie idzie od razu na cel (pusty, zamkniety chwytak nic nie traci,
                            # otwarty nie drgnie), a przy zlym zgadnieciu pierwsza ramka nie zatrzaskuje
                            # szczek (+0,6 z kp 25, bez progu sily - ten dziala tylko przy zamykaniu).
                            # `_g_cmd` z rozkazu, ktory naprawde poszedl, przezywa przerwe lacza i dalej
                            # jest rampa od niego.
                            stream = min(stream, want)
                        self._g_target, self._g_stream, self._g_assumed = want, stream, False
                    elif want <= self._g_target:
                        # Otwieranie (albo ten sam cel) zawsze przechodzi i zdejmuje zamrozenie. Z ostrym `<`
                        # cel zamrozony na krancu otwarcia (-2,0) nie dawal sie juz niczym ruszyc.
                        self._g_target, self._g_frozen = want, False
                    elif not self._g_frozen:
                        if self._grip_contact() and self._g_stream is not None:
                            self._g_target, self._g_frozen = self._g_stream, True
                        else:
                            self._g_target = want
                    self._g_cmd = v
                sent[GRIPPER] = v
            self._goal.update(sent)
            return sent

    def _fresh_fb(self) -> Any:
        """Latest 0x052 frame, or None when missing, stale or frozen."""
        if not self._connected or self._fb is None or self._stale_now(self._clock()) or self._frozen():
            return None
        return self._fb

    def efforts(self) -> dict[str, float] | None:
        """Raw A1X effort scale from frame 0x052; arm joints only, no gripper."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return {name: float(fb.eff[i]) for i, name in enumerate(self.arm)}

    def velocities(self) -> dict[str, float] | None:
        """deg/s from frame 0x052; arm joints only, no gripper."""
        with self._lock:
            fb = self._fresh_fb()
            if fb is None:
                return None
            return {name: math.degrees(fb.vel[i]) for i, name in enumerate(self.arm)}

    def goal_positions(self) -> dict[str, float]:
        """Ostatnie cele [jednostki aplikacji]; `{}` przed pierwszym rozkazem (p_des nie da sie odczytac)."""
        with self._lock:
            return dict(self._goal)

    def faults(self) -> list[str]:
        try:
            with self._lock:
                if not self._connected:
                    return []
                out: list[str] = []
                now = self._clock()
                if self._bus_error:
                    out.append(f"magistrala CAN: {self._bus_error}")
                if self._stale_now(now):
                    out.append(f"brak odpowiedzi ramienia od {(now - self._fb_t) * 1000.0:.0f} ms")
                elif self._frozen():
                    out.append(f"telemetria zamrozona ({self._identical} identycznych ramek) - ramie zwolnione "
                               f"(FF 2)? wylacz i wlacz zasilanie")
                elif self._deaf:
                    out.append(self._deaf)
                state = str(getattr(self._bus, "state", "ok") or "ok")
                if state in ("bus-off", "error-passive"):
                    out.append(f"magistrala CAN: {state}")
                return out
        except Exception as exc:               # pragma: no cover - wolane co cykl petli, nie moze jej wysypac
            logger.exception("Nie udalo sie sprawdzic stanu A1X")
            return [f"nie udalo sie sprawdzic stanu A1X: {exc}"]

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Limity URDF [st.] poszerzone o poze startowa (J3 stoi ~1,5 st. za swoim limitem); bez chwytaka."""
        return {name: (math.degrees(lo - self._off[i]), math.degrees(hi - self._off[i]))
                for i, (name, (lo, hi)) in enumerate(zip(self.arm, self._bounds))}

    def gripper_ticks(self) -> None:
        return None

    def gripper_known(self) -> bool:
        """Czy pozycja chwytaka jest znana: pomiar, rozkaz, ktory NAPRAWDE poszedl w 0x051, albo
        `assume_gripper`. Cel rowny zglaszanemu domyslnemu (`GRIP_START_DELTA`) nic nie wysyla
        i niczego nie ustala."""
        with self._lock:
            return self.gripper_measured() or self._g_cmd is not None

    def assume_gripper(self, v: float) -> bool:
        """Operator potwierdza rozwarcie chwytaka bez pomiaru ("Ustaw dom = tu": szczeki zamkniete = 0).

        Tylko dopoki strumien 0x051 nie ruszyl: potem prawda jest ostatni rozkaz, ktory naprawde poszedl,
        a zalozenie jest odrzucane (False, ostrzezenie w logu). Po udanym `gripper_known()` jest True,
        wiec nastepne rozkazy ida rampa od zalozenia; pierwszy strumien startuje od bardziej otwartego
        z {zalozenie, cel} (`send_joints`, tak samo jak od samego `gripper_start`). Chwytak z pomiarem
        niczego nie zaklada (False).
        """
        with self._lock:
            if self.gripper_measured():
                return False
            if self._g_target is not None:
                logger.warning("A1X chwytak juz sterowany (p_des %.2f) - zalozenie %.0f odrzucone", self._g_target, v)
                return False
            self._g_cmd = min(max(float(v), 0.0), 100.0)
            self._g_assumed = True
            logger.info("A1X chwytak: zalozone rozwarcie %.0f (bez pomiaru)", self._g_cmd)
            return True

    def gripper_drivable(self) -> bool:
        """Czy rozkaz chwytaka dojdzie do ramienia. Nie przez XCAN (macOS): przejsciowka wysyla tylko
        prawidlowe dlugosci CAN FD, wiec 10-bajtowa 0x051 idzie jako 12 bajtow, a chwytak ja ignoruje
        (galaxeo `bus.py`). Nie da sie tego naprawic programowo; chwytak tylko przez SocketCAN."""
        return not str(self.iface or "").lower().startswith("xcan")

    def gripper_travel_s(self) -> float:
        """Czas pelnego skoku chwytaka [s]: |open - closed| / `gripper_rate_pdes_s` + zapas (1,5 s przy
        domyslnych). Tyle czekac po rozkazie, zeby szczeki naprawde doszly (1,2 s bylo za malo na 1,3 s)."""
        a = self.acfg
        return abs(a.gripper_open_pdes - a.gripper_closed_pdes) / a.gripper_rate_pdes_s + GRIP_TRAVEL_MARGIN_S

    def gripper_measured(self) -> bool:
        """Czy chwytak ma pomiar (dwa punkty `gripper_fb_*`). Bez nich `joints()` zglasza ostatni
        rozkaz, a przed pierwszym - `gripper_start` (albo `assume_gripper`), ktore nie musi zgadzac
        sie ze szczekami."""
        a = self.acfg
        return a.gripper_fb_closed is not None and a.gripper_fb_open is not None

    def calibration_warnings(self) -> list[str]:
        a = self.acfg
        if a.gripper_fb_closed is None or a.gripper_fb_open is None:
            return ["chwytak A1X: sprzezenie niezmierzone - panel pokazuje rozkaz, nie pomiar; "
                    "zmierz a1x.gripper_fb_closed/open"]
        return []

    # ------------------------------------------------------------- wlaczenie
    def enable(self) -> str:
        """Sekwencja FF 1 -> 5 -> 6, z p_des = pomiar przed, miedzy i po kazdym kodzie.

        Tylko na zyczenie (panel albo `a1x.enable_on_connect`): zasilone od nowa ramie slucha
        od razu, a FF 5 na chwile puszcza silniki. Cele z `send_joints` sa w tym czasie odrzucane.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("Ramie A1X nie jest polaczone")
            if self._fb is None or self._stale_now(self._clock()):
                raise RuntimeError("A1X: brak swiezych ramek 0x052 - wlaczenie odrzucone")
            if self._frozen():
                raise RuntimeError("A1X: telemetria zamrozona (FF 2) - wylacz i wlacz zasilanie; FF 1-5-6 tego "
                                   "nie naprawia")
            self._pinned = True
            self._p_target = self._measured_rad()
            self._p_stream = list(self._p_target)
        try:
            for code in self.P.ENABLE_SEQUENCE:
                self._wait(ENABLE_STEP_S)
                with self._lock:
                    if self._stale_now(self._clock()):
                        raise RuntimeError(f"A1X: lacze zamilklo w trakcie wlaczania (przed FF {code})")
                    self._send(self.P.FF_ID, self.P.encode_ff(code), fd=False)
            self._wait(ENABLE_STEP_S)
        finally:
            with self._lock:
                # Po zerwaniu lacza w trakcie - bez celu (ostatni pomiar sprzed przerwy to juz nie poza ramienia).
                fresh = self._fb is not None and not self._stale_now(self._clock()) and not self._frozen()
                self._p_target = self._measured_rad() if fresh else None
                if not fresh:
                    self._p_stream = None
                self._pinned = False
        return "wyslano FF 1 -> 5 -> 6; ramie trzyma zmierzona poze"

    def _wait(self, secs: float) -> None:
        """Czas miedzy kodami: watek IO streamuje sam; bez watku - takty tutaj."""
        if self._threaded:
            self._sleep(secs)
            return
        period = 1.0 / self.acfg.stream_hz
        for _ in range(max(1, round(secs / period))):
            self._sleep(period)
            self.io_step(self._clock())
