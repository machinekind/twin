"""Kopia mrowki w bliznaku: nogi z odczytow serw, poza dociagana tagami, montaz taga kalibrowany sam.

    mirror = AntMirror(twin, ant_getter, ZmqAnt(ZmqAntConfig(host)))     # prawdziwa: odczyty z ant_server
    mirror = AntMirror(twin, ant_getter, StandInServos(twin))            # demo: mrowka udajaca prawdziwa
    mirror.start()                                                       # watek; w testach: mirror.tick()

Prawdziwa mrowka jezdzi na wlasnej polityce (realant-rl `robot_teleop.py` po jej stronie); blizniak niczego jej
nie wysyla. Ma za to jej kopie w fizyce sceny (`scene.ANT_PREFIX`), ktora chodzi niezaleznie:

* **nogi** - cele serw kopii to katy odczytane z serw prawdziwej mrowki, wiec kopia stawia te same kroki
  i sama sie przesuwa (takze wtedy, gdy kamery jej nie widza);
* **poza** - kazdy kadr z tagami dociaga pozycje i kurs kopii do pozy z kamer. Kadr ma chwile wykonania,
  a kopia pamieta swoje pozy, wiec blad liczy sie wzgledem kopii z tej samej chwili (opoznienie kamery nie
  ciagnie mrowki do tylu), a poprawka idzie na poze teraz. Wysokosc i pochylenie zostaja z fizyki - nogi na blacie;
* **montaz taga** (`Ant.mount`) kalibruje sie sam: kierunek przodu - z porownania, dokad kopia idzie na odczytach
  serw (odometria fizyki bez poprawek), a dokad prawdziwa wedlug tagow (roznica katow = blad odchylenia montazu,
  z `ant.snap_yaw` przyciagany do 45 + k * 90 st.); wysokosc i pochylenie taga - z taga wzgledem korpusu kopii,
  gdy ta dogonila tagi (korpus stoi na nogach, wiec jego wysokosc i pochylenie sa z fizyki, nie z kamer).

W demo nie ma prawdziwej mrowki - udaje ja druga mrowka w scenie (`scene.STANDIN_PREFIX`, `antdrive`), a jej katy
przychodza tu przez `StandInServos` zaokraglone do jednostek AX-12A, jak z ant_server. Kopia jest wtedy schowana
przed kamerami symulowanymi (`AntPlacement.hidden`) - te widza tylko tagi mrowki udajacej prawdziwa.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import antwalk as aw
from .ant import Ant, locate, snap_yaw, zyx
from .kinematics import inverse
from .scene import ANT_PREFIX, STANDIN_PREFIX

#: Blad pozy kopii wzgledem tagow powyzej tego = przeskok od razu (pierwszy kadr, zgubiona, przeniesiona recznie).
SNAP_M, SNAP_DEG = 0.10, 30.0
#: Odcinek toru do kalibracji odchylenia: tagi przesunely sie co najmniej tyle [m].
YAW_WINDOW_M = 0.06
#: Kopia "dogonila" tagi (kalibracja wysokosci i pochylenia): blad ponizej tego.
LEVEL_OK_M, LEVEL_OK_DEG = 0.012, 3.0


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class StandInServos:
    """Demo: odczyty serw mrowki udajacej prawdziwa - katy zaokraglone do jednostek AX-12A, jak z ant_server."""

    def __init__(self, twin, prefix: str = STANDIN_PREFIX):
        self.twin, self.prefix = twin, prefix
        self._prev: tuple[np.ndarray, float] | None = None

    def joints(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self.twin.lock:
            if self.prefix not in self.twin.scene.ants():
                return None
            q, _ = self.twin.scene.ant_joints(self.prefix)
            t = float(self.twin.scene.data.time)
        q = aw.units_to_joints(aw.joints_to_units(q))
        v = np.zeros(8)
        if self._prev is not None and t > self._prev[1]:
            v = (q - self._prev[0]) / (t - self._prev[1])
        self._prev = (q, t)
        return q, v, 0.0

    def close(self) -> None:
        pass


@dataclass
class MirrorStatus:
    #: wylaczona / czeka na serwa / sledzi / bez tagow
    mode: str = "wylaczona"
    note: str = ""
    servo_age: float | None = None
    tag_age: float | None = None
    #: Ostatni blad kopii wzgledem tagow (przed poprawka) [mm], [st.].
    err_mm: float | None = None
    err_deg: float | None = None
    fixes: int = 0
    T_ant2base: np.ndarray | None = None
    #: Co ostatnio przestawila kalibracja montazu (tekst dla panelu).
    calib: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class AntMirror:
    """Kopia mrowki: nogi z serw, poza z tagow, samokalibracja montazu. `tick` = jeden krok (domyslnie 50 ms)."""

    def __init__(self, twin, get_ant: Callable[[], Ant | None], servos, *, gain: float = 0.5,
                 auto_yaw: bool = True, auto_level: bool = True, snap: bool = True,
                 on_missing: Callable[[np.ndarray], None] | None = None, prefix: str = ANT_PREFIX):
        self.twin = twin
        self.get_ant = get_ant
        self.servos = servos
        self.prefix = prefix
        #: Jaka czesc bledu wzgledem tagow poprawiac przy kazdym kadrze (1 = przeskok do tagow).
        self.gain = gain
        self.auto_yaw, self.auto_level, self.snap = auto_yaw, auto_level, snap
        #: Wolane z poza z tagow, gdy w scenie nie ma jeszcze kopii (panel przebudowuje scene).
        self.on_missing = on_missing
        self.status = MirrorStatus()
        self._t_seen: float | None = None
        self._hist: deque[tuple[float, np.ndarray]] = deque(maxlen=80)
        self._p_prev: np.ndarray | None = None
        self._odo = np.zeros(2)
        self._win: tuple[np.ndarray, np.ndarray] | None = None       # (tag xy, odometria) na poczatku odcinka
        self._last_frame_t = 0.0
        self._thread: threading.Thread | None = None
        self._run = threading.Event()

    # ---------------------------------------------------------------- watek
    def start(self) -> None:
        if self._thread is not None:
            return
        self._run.set()
        self._thread = threading.Thread(target=self._loop, name="kopia-mrowki", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        t0, k = time.monotonic(), 0
        while self._run.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - petla nie moze zginac po cichu
                self.status.note = f"blad petli kopii: {e}"
            k += 1
            dt = t0 + k * aw.DT - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            elif dt < -1.0:
                t0, k = time.monotonic(), 0

    def close(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.servos.close()

    # ---------------------------------------------------------------- krok
    def tick(self) -> None:
        st = self.status
        ant = self.get_ant()
        j = self.servos.joints()
        st.servo_age = None if j is None else float(j[2])
        with self.twin.lock:
            present = self.prefix in self.twin.scene.ants()
            if present and j is not None:
                self.twin.scene.command_ant(j[0], self.prefix)
        if present:
            self._advance()
        now = time.monotonic()
        with self.twin.lock:
            T_phys = self.twin.scene.ant_pose(self.prefix) if present else None
        if T_phys is not None:
            if self._p_prev is not None:
                self._odo += T_phys[:2, 3] - self._p_prev[:2]          # ruch z fizyki, bez poprawek z tagow
            self._hist.append((now, T_phys))
        fix, t_frame = self._tags(ant)
        if fix is not None:
            st.tag_age = now - t_frame
            if not present:
                if self.on_missing is not None:
                    self.on_missing(fix.T_ant2base)
                st.mode, st.note = "czeka na scene", "kopia mrowki trafia do sceny (przebudowa)"
                return
            self._correct(ant, fix, t_frame)
        elif st.tag_age is not None:
            st.tag_age = now - self._last_frame_t if self._last_frame_t else None
        with self.twin.lock:
            T_now = self.twin.scene.ant_pose(self.prefix) if present else None
        self._p_prev = None if T_now is None else T_now[:3, 3].copy()
        st.T_ant2base = T_now
        if not present:
            st.mode, st.note = "czeka na tagi", "kopii nie ma jeszcze w scenie - pierwsza poza z tagow ja postawi"
        elif j is None:
            st.mode, st.note = "czeka na serwa", "brak odczytow serw (antproxy.py i ant_server.py uruchomione?)"
        elif st.tag_age is None or st.tag_age > 1.0:
            st.mode = "bez tagow"
            st.note = "kamery nie widza tagow - kopia idzie sama na odczytach serw"
        else:
            st.mode, st.note = "sledzi", ""

    def _advance(self) -> None:
        """Krok fizyki, gdy nikt inny go nie robi (ramie `sim` niepolaczone, demo bez mrowki udajacej prawdziwa)."""
        with self.twin.lock:
            s = self.twin.scene
            t = float(s.data.time)
            if self._t_seen is None or t - self._t_seen < 0.5 * aw.DT:
                s.step(aw.DT)
            self._t_seen = float(s.data.time)

    def _tags(self, ant: Ant | None):
        """(poza z tagow albo None, chwila kadru). Tylko kadry nowsze niz poprzednio uzyte."""
        if ant is None or not ant.tags.known():
            return None, 0.0
        frames, times = {}, []
        for cam in self.twin.workspace.cameras:
            if not cam.enabled or cam.T_cam2base is None:
                continue
            try:
                img, tc = self.twin.cameras.frame_t(cam.name)
            except Exception:
                continue
            if img is None or (tc and tc <= self._last_frame_t) or (tc and time.monotonic() - tc > 0.5):
                continue
            frames[cam.name] = img
            times.append(tc or time.monotonic())
        if not frames:
            return None, 0.0
        fix = locate(frames, self.twin.workspace.cameras, ant)
        if fix is None:
            return None, 0.0
        t = min(times)
        self._last_frame_t = max(times)
        return fix, t

    def _at(self, t: float) -> np.ndarray | None:
        """Poza kopii z chwili `t` (najblizsza zapamietana)."""
        if not self._hist:
            return None
        return min(self._hist, key=lambda e: abs(e[0] - t))[1]

    def _correct(self, ant: Ant, fix, t_frame: float) -> None:
        st = self.status
        T_then = self._at(t_frame)
        with self.twin.lock:
            T_now = self.twin.scene.ant_pose(self.prefix)
        if T_then is None:
            T_then = T_now
        T_tag = fix.T_ant2base
        e_xy = T_tag[:2, 3] - T_then[:2, 3]
        e_yaw = float(_wrap(aw.heading(T_tag) - aw.heading(T_then)))
        st.err_mm, st.err_deg = float(np.hypot(*e_xy) * 1000), float(np.degrees(e_yaw))
        big = np.hypot(*e_xy) > SNAP_M or abs(np.degrees(e_yaw)) > SNAP_DEG or st.fixes == 0
        g = 1.0 if big else self.gain
        # Obrot o g * e_yaw wokol pionu przez srodek mrowki, przesuniecie o g * e_xy; wysokosc i pochylenie z fizyki.
        c, s_ = np.cos(g * e_yaw), np.sin(g * e_yaw)
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        T_new = T_now.copy()
        T_new[:3, :3] = Rz @ T_now[:3, :3]
        T_new[:2, 3] = T_now[:2, 3] + g * e_xy
        with self.twin.lock:
            self.twin.scene.set_ant_pose(T_new, self.prefix)
        st.fixes += 1
        if big:
            self._win = None                              # przeskok - odcinek odometrii od nowa
            self._hist.clear()
        if self.auto_yaw:
            self._calib_yaw(ant, T_tag)
        if self.auto_level and not big and st.err_mm < LEVEL_OK_M * 1000 and abs(st.err_deg) < LEVEL_OK_DEG:
            self._calib_level(ant, fix, T_then)

    def _calib_yaw(self, ant: Ant, T_tag: np.ndarray) -> None:
        """Odchylenie montazu: kat odometrii kopii minus kat toru z tagow na tym samym odcinku."""
        p = T_tag[:2, 3].copy()
        if self._win is None:
            self._win = (p, self._odo.copy())
            return
        d_tag = p - self._win[0]
        d_odo = self._odo - self._win[1]
        if np.hypot(*d_tag) < YAW_WINDOW_M:
            return
        self._win = (p, self._odo.copy())
        n_tag, n_odo = np.hypot(*d_tag), np.hypot(*d_odo)
        if not 0.5 < n_odo / n_tag < 2.0:                 # kopia nie odtworzyla ruchu (poslizg, nogi nie te)
            self.status.calib = f"kierunek: odcinek pominiety (kopia {n_odo * 100:.0f} cm, tagi {n_tag * 100:.0f} cm)"
            return
        delta = float(np.degrees(_wrap(np.arctan2(d_odo[1], d_odo[0]) - np.arctan2(d_tag[1], d_tag[0]))))
        old = float(ant.mount["yaw"])
        raw = old + delta
        new = snap_yaw(raw)[0] if self.snap else raw
        new = float(np.degrees(_wrap(np.radians(new))))
        if abs(_wrap(np.radians(new - old))) > np.radians(0.5):
            ant.mount = {**ant.mount, "yaw": new}
            self.status.calib = f"kierunek: obrot taga {old:+.0f} -> {new:+.0f} st. (odcinek {n_tag * 100:.0f} cm, blad {delta:+.1f})"
        else:
            self.status.calib = f"kierunek: zgodny (odcinek {n_tag * 100:.0f} cm, blad {delta:+.1f} st.)"

    def _calib_level(self, ant: Ant, fix, T_then: np.ndarray) -> None:
        """Wysokosc i pochylenie taga wzgledem korpusu kopii (stoi na nogach - z fizyki), filtrowane."""
        T_rel = inverse(T_then) @ fix.tags.T_body2base            # tag na grzbiecie w ukladzie kopii
        _, pitch, roll = zyx(T_rel[:3, :3])
        a = 0.1
        m = ant.mount
        ant.mount = {**m, "z": (1 - a) * m["z"] + a * float(T_rel[2, 3]),
                     "pitch": (1 - a) * m["pitch"] + a * pitch, "roll": (1 - a) * m["roll"] + a * roll}


def mirror_pose_error(twin, standin_prefix: str = STANDIN_PREFIX, prefix: str = ANT_PREFIX) -> tuple[float, float]:
    """Demo i testy: (blad polozenia [m], blad kursu [st.]) kopii wzgledem mrowki udajacej prawdziwa."""
    with twin.lock:
        A = twin.scene.ant_pose(prefix)
        B = twin.scene.ant_pose(standin_prefix)
    return (float(np.hypot(*(A[:2, 3] - B[:2, 3]))),
            float(abs(np.degrees(_wrap(aw.heading(A) - aw.heading(B))))))


__all__ = ["AntMirror", "MirrorStatus", "StandInServos", "mirror_pose_error"]
