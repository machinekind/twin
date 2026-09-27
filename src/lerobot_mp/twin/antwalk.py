"""Chod mrowki RealAnt: polityka PPO z realant-rl w numpy, jej obserwacja z tagow i lacze z prawdziwa mrowka.

Przeniesione z realant-rl `robot_policy.py` (galaz new_policies, e4926d1): wczytanie pickla brax bez JAX,
siec (MLP swish, akcja tanh(srodka)), obserwacja jak w `realant_sim/v2_core.py`, chod w czterech kierunkach
przez przemianowanie nog (`DIRECTIONS`), obrot w miejscu z otwartej petli (`turn_cycle`, skalibrowany na
robocie), mapy serw z `rollout_server.py`. Zmienione jest jedno: skad jest poza korpusu.

**Mrowka nie wie, jak jest obrocona** - nie ma IMU. W treningu obserwacja ma przesuniecie korpusu na krok,
jego wysokosc, przechyl, pochylenie i kurs WZGLEDEM kursu odniesienia - z "kamery sledzacej". Tu ta kamera
to tagi na grzbiecie i maszcie widziane przez kamery stanowiska (`ant.locate`). `robot_policy.py` ma na to
tryb `live` oznaczony jako nietestowany; tutaj poza przychodzi w ukladzie podstawy ramienia i obserwator
sam liczy z niej to, co widziala polityka:

* uklad "swiata polityki" = kurs odniesienia `psi` jako +x (w treningu mrowka zawsze startuje przodem
  do +x swiata), wysokosc nad blatem jako z;
* korpus modelu jest obrocony o 180 st. wzgledem przodu mrowki (`scene.ANT_TORSO_IN_ANT`), a przy
  chodzie w bok / w tyl nogi sa przemianowane o 90 / 180 st. - polityka widzi "swoj" przod w kierunku jazdy;
* opoznienie: w treningu pomiar kamery jest o `latency` krokow (100 ms) starszy niz stawy. Pomiary
  maja tu chwile kadru, wiec obserwator bierze ten, ktory byl `latency * DT` temu - kamera wolniejsza
  od treningowej nie dostaje drugiego opoznienia na wierzch (`robot_policy.py` robi to recznie:
  `--camera_delay` 0 na robocie, 2 w symulacji).

Kurs odniesienia to tez kierownica: przy W+A / W+D obraca sie go powoli (`AntDrive`), a polityka,
nagradzana w treningu za trzymanie kursu, skreca do niego sama - z pozy z tagow.
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .robots import resolve_asset

DT = 0.05                                   # krok sterowania [s] - robot i trening (10 x 5 ms)
N_FEATURES = 29
#: Zakres serw modelu v2 (ctrlrange): akcja +-1 -> biodro +-HIP_CTRL rad; kostka -1 (w dol) .. +1 (poziomo) ->
#: ANKLE_CTRL .. 0 rad.
HIP_CTRL, ANKLE_CTRL = 0.80031, 1.14549
#: Poza "osiadania" przed obrotem (robot_route.py): biodra na srodku, nogi lekko w dol.
SETTLE_POSE = np.array([0.0, -0.25] * 4)
#: Kierunki chodu: noga polityki k (0..3) napedza fizyczna noge perm[k]; theta = kierunek jazdy od przodu.
#: Nogi 1 z tylu po prawej, 2 z przodu po prawej, 3 z przodu po lewej, 4 z tylu po lewej.
DIRECTIONS: dict[str, tuple[list[int], float]] = {
    "przod": ([0, 1, 2, 3], 0.0), "lewo": ([1, 2, 3, 0], np.pi / 2),
    "prawo": ([3, 0, 1, 2], -np.pi / 2), "tyl": ([2, 3, 0, 1], np.pi)}
#: Obrot w miejscu (sim_test_trajectory_cyclic_turn.py): cykl 1 s, pary nog po przekatnych na zmiane.
TURN_PERIOD = 1.0
TURN_LEG_PHASE = [0.0, np.pi, 0.0, np.pi]
#: Kroki na pelny obrot, zmierzone na robocie (robot_route.py): w lewo 100, w prawo 90 -> 3,6 / 4 st. na krok.
TURN_STEPS_PER_360 = {"lewo": 100, "prawo": 90}

DEFAULT_POLICY = "assets/policies/ant/exp004_smooth2"


# ------------------------------------------------------------------ polityka (numpy)
def _reconstruct_array(fun, args, arr_state, aval_state):  # zamiast jax._src.array._reconstruct_array
    a = fun(*args)
    a.__setstate__(arr_state)
    return a


class _Opaque:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _Unpickler(pickle.Unpickler):
    """Pickle parametrow brax bez jax / brax / flax."""

    def find_class(self, module, name):
        if module == "jax._src.array" and name == "_reconstruct_array":
            return _reconstruct_array
        if module.split(".")[0] in ("jax", "jaxlib", "brax", "flax"):
            return _Opaque
        return super().find_class(module, name)


@dataclass
class WalkPolicy:
    """Polityka chodu: normalizator obserwacji, warstwy sieci i konfiguracja chodu z treningu (`gait`)."""

    name: str
    cfg: dict[str, Any]
    mean: np.ndarray
    std: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]
    steps: int = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, np.float32) - self.mean) / self.std          # running_statistics.normalize
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x * 0.5 * (1.0 + np.tanh(0.5 * x))                    # swish = x * sigmoid(x)
        n = self.layers[-1][0].shape[1] // 2
        return np.tanh(x[:n]).astype(float)                              # tanh(srodka) - polityka deterministyczna

    @property
    def obs_size(self) -> int:
        return int(self.layers[0][0].shape[0])

    def describe(self) -> str:
        return f"{self.name} ({self.steps / 1e6:.0f} mln krokow PPO, " + " -> ".join(
            str(w.shape[0]) for w, _ in self.layers) + f" -> {self.layers[-1][0].shape[1]})"


def load_policy(path: str | Path = DEFAULT_POLICY) -> WalkPolicy:
    """Polityka z katalogu (`params_*.pkl` + `config.json` z train_mjx.py) albo wprost z pliku pkl."""
    p = resolve_asset(path)
    pkl = p if p.is_file() else sorted(p.glob("params_*.pkl"))[-1]
    config = json.loads((pkl.parent / "config.json").read_text(encoding="utf-8"))
    with open(pkl, "rb") as f:
        normalizer, policy = _Unpickler(f).load()[:2]
    layers = policy["params"]
    ws = [(np.asarray(layers[n]["kernel"]), np.asarray(layers[n]["bias"]))
          for n in sorted(layers, key=lambda s: int(s.split("_")[-1]))]
    cfg = dict(config["gait"])
    if cfg.get("mode", "walk") != "walk":
        raise ValueError(f"{pkl}: polityka w trybie {cfg['mode']!r} - obsluga tylko 'walk' (komenda predkosci "
                         f"joysticka nie jest tu podawana)")
    steps = int(pkl.stem.split("_")[-1]) if pkl.stem.split("_")[-1].isdigit() else 0
    pol = WalkPolicy(config.get("name", pkl.parent.name), cfg, np.asarray(vars(normalizer)["mean"]),
                     np.asarray(vars(normalizer)["std"]), ws, steps)
    expect = N_FEATURES * (cfg["latency"] + cfg["min_obs_stack"]) + 2 + 8
    if pol.obs_size != expect:
        raise ValueError(f"{pkl}: siec bierze {pol.obs_size} wejsc, obserwacja chodu ma {expect}")
    return pol


# ------------------------------------------------------------------ akcje
def action_to_ctrl(a: np.ndarray) -> np.ndarray:
    """Akcja [-1, 1]^8 -> cele serw [rad] (v2_core.action_to_ctrl)."""
    a = np.clip(np.asarray(a, float), -1, 1)
    out = np.empty(8)
    out[0::2] = a[0::2] * HIP_CTRL
    out[1::2] = (1 - a[1::2]) / 2 * ANKLE_CTRL
    return out


def ctrl_to_action(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    out = np.empty(8)
    out[0::2] = q[0::2] / HIP_CTRL
    out[1::2] = 1 - 2 * q[1::2] / ANKLE_CTRL
    return np.clip(out, -1, 1)


def start_action(cfg: dict[str, Any]) -> np.ndarray:
    """Poza startowa polityki: biodra na srodku, kostki na `init_ankle` z treningu (na palcach)."""
    return np.array([0.0, 1 - 2 * cfg["init_ankle"] / ANKLE_CTRL] * 4)


def turn_cycle(direction: float, hip_amp: float = 1.0, ankle_amp: float = 1.0, ankle_offset: float = 0.0,
               ankle_phase_deg: float = 0.0) -> np.ndarray:
    """Jeden cykl (TURN_PERIOD / DT krokow) obrotu w miejscu; direction +1 = w lewo (przeciwnie do zegara)."""
    n = round(TURN_PERIOD / DT)
    out = []
    for t in range(n):
        phi = np.array(TURN_LEG_PHASE) + 2 * np.pi * t / n
        hip = direction * hip_amp * np.sin(phi)
        ankle = ankle_offset + ankle_amp * np.cos(phi + np.radians(ankle_phase_deg))  # noga w gorze na powrocie
        out.append(np.clip(np.stack([hip, ankle], 1).reshape(-1), -1, 1))
    return np.array(out)


def leg_index(direction: str) -> np.ndarray:
    """Fizyczny staw (0..7) kazdego stawu polityki przy chodzie w `direction`."""
    perm, _ = DIRECTIONS[direction]
    return np.array([2 * perm[k] + j for k in range(4) for j in (0, 1)])


# ------------------------------------------------------------------ obserwacja
def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rpy(R: np.ndarray) -> np.ndarray:
    """Przechyl, pochylenie, kurs jak `v2_core.rpy_from_quat` (zewnetrzne xyz)."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def heading(T_ant2base: np.ndarray) -> float:
    """Kurs mrowki (kierunek jej +x) w plaszczyznie blatu [rad]."""
    x = np.asarray(T_ant2base, float)[:3, 0]
    return float(np.arctan2(x[1], x[0]))


class Observer:
    """Obserwacja polityki z pomiarow: poza korpusu z tagow (z chwila kadru), stawy, zegar chodu, poprzednia akcja.

    `psi` - kurs odniesienia (kierunek jazdy) w ukladzie podstawy; `theta` - kierunek jazdy wzgledem przodu
    mrowki (`DIRECTIONS`). Z nich uklad "swiata polityki" i korpus "wirtualnej" mrowki o nogach przemianowanych
    tak, ze jej przod patrzy w kierunku jazdy - dokladnie to, co w treningu widziala mrowka startujaca przodem do +x.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.L = int(cfg["latency"])
        self.n_stack = int(cfg["latency"]) + int(cfg["min_obs_stack"])
        self.poses: deque[tuple[float, np.ndarray]] = deque(maxlen=64)

    def add_pose(self, t: float, T_ant2base: np.ndarray) -> None:
        """Pomiar pozy mrowki (z tagow) z chwila kadru `t` [s, zegar monotoniczny]."""
        if self.poses and t <= self.poses[-1][0]:
            return
        self.poses.append((float(t), np.asarray(T_ant2base, float).copy()))

    def _delayed(self, now: float) -> np.ndarray | None:
        """Poza sprzed `latency * DT` (ostatnia nie nowsza); gdy wszystkie sa nowsze - najstarsza."""
        if not self.poses:
            return None
        want = now - self.L * DT
        best = self.poses[0][1]
        for t, T in self.poses:
            if t <= want + 1e-3:
                best = T
            else:
                break
        return best

    def _measure(self, T_ant2base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(xyz, rpy) korpusu wirtualnej mrowki w swiecie polityki; kurs wzgledny (0 = na kursie odniesienia)."""
        Rw = _rz(self.psi).T
        R_torso = Rw @ T_ant2base[:3, :3] @ _rz(self.theta) @ _rz(np.pi)
        r = rpy(R_torso)
        r[2] = _wrap(r[2] - np.pi)                     # korpus modelu patrzy w tyl: kurs pi = przodem do +x
        return Rw @ T_ant2base[:3, 3], r

    def reset(self, now: float, jpos: np.ndarray, jvel: np.ndarray, prev_action: np.ndarray, psi: float,
              theta: float = 0.0, phase0: float = 0.0) -> np.ndarray:
        self.t, self.phase0, self.psi, self.theta = 0.0, float(phase0), float(psi), float(theta)
        T = self._delayed(now)
        if T is None:
            raise RuntimeError("brak pozy mrowki z kamer")
        self.T_prev = T
        self.stack = np.zeros((self.n_stack, N_FEATURES))
        self.prev_action = np.clip(np.asarray(prev_action, float), -1, 1)
        self._push(T, jpos, jvel)
        return self.observation()

    def step(self, now: float, action: np.ndarray, jpos: np.ndarray, jvel: np.ndarray,
             psi: float | None = None) -> np.ndarray:
        """Po zastosowaniu `action` i jednym kroku: nastepna obserwacja (`psi` - nowy kurs odniesienia)."""
        if psi is not None:
            self.psi = float(psi)
        self.prev_action = np.clip(np.asarray(action, float), -1, 1)
        self.t += DT
        T = self._delayed(now)
        self._push(T if T is not None else self.T_prev, jpos, jvel)
        return self.observation()

    def _push(self, T: np.ndarray, jpos: np.ndarray, jvel: np.ndarray) -> None:
        # Poprzedni i obecny pomiar liczone tym samym kursem odniesienia - obrot kierownicy nie jest ruchem.
        xyz, r = self._measure(T)
        pxyz, pr = self._measure(self.T_prev)
        self.T_prev = T
        self.frame = np.concatenate([xyz - pxyz, xyz[2:], _wrap(r - pr), np.sin(r), np.cos(r),
                                     np.asarray(jpos, float), np.asarray(jvel, float)])
        self.stack = np.concatenate([self.stack[1:], self.frame[None]])

    def observation(self) -> np.ndarray:
        phase = 2 * np.pi * np.mod(self.phase0 + self.t * self.cfg["gait_freq"], 1.0)
        return np.concatenate([self.stack.reshape(-1), [np.sin(phase), np.cos(phase)], self.prev_action])


# ------------------------------------------------------------------ wysokosc korpusu
class HeightHold:
    """Regulator wysokosci z realant-rl (robot_policy.py, c2e6d00): calka na srednim kacie kostek -> wspolny offset
    kostek (jednostki akcji, ujemny = nogi nizej).

    Prawdziwe serwa siadaja pod obciazeniem bardziej niz model, wiec korpus powoli opada, a polityka (ktora na
    wysokosc z kamery nie reaguje) daje dalej te same kostki. Calkuje tylko w trakcie jazdy, po 1 s rozruchu;
    w symulacji z nominalnymi serwami nic nie zmienia. Offset zostaje miedzy odcinkami jazdy.
    """

    BIAS_MIN, BIAS_MAX, TAU, WARMUP = -0.4, 0.05, 1.0, 1.0

    def __init__(self, target_deg: float = 56.1, gain: float = 0.02, bias0: float = 0.0, enabled: bool = True):
        self.target, self.gain, self.enabled = np.radians(target_deg), gain, enabled
        self.bias, self.filt = float(np.clip(bias0, self.BIAS_MIN, self.BIAS_MAX)), None

    @staticmethod
    def clearance(ankle_rad: float) -> float:
        """Przeswit pod korpusem [m] z kata kostek (biodra na srodku, stopy na podlodze; dopasowanie do modelu v2)."""
        return 0.0933 * np.sin(ankle_rad) - 0.0239

    def command(self, a: np.ndarray) -> np.ndarray:
        cmd = np.array(a, float)
        cmd[1::2] = np.clip(cmd[1::2] + self.bias, -1, 1)
        return cmd

    def update(self, ankle_rad: np.ndarray, t: float) -> None:
        """`ankle_rad` - 4 zmierzone kostki, `t` - sekundy od startu polityki w tym odcinku."""
        mean = float(np.mean(ankle_rad))
        self.filt = mean if self.filt is None else self.filt + (DT / self.TAU) * (mean - self.filt)
        if self.enabled and self.gain and t > self.WARMUP:
            self.bias = float(np.clip(self.bias - self.gain * np.degrees(self.target - self.filt) * DT,
                                      self.BIAS_MIN, self.BIAS_MAX))


# ------------------------------------------------------------------ prawdziwa mrowka (ZeroMQ)
#: Serwo (0..7 = s1..s8) kazdego stawu w kolejnosci polityki - rollout_server.apply_controls / get_obs.
SERVO_OF_JOINT = (2, 3, 0, 1, 6, 7, 4, 5)
#: Jednostki AX-12A (0..1023, 512 = srodek) dla akcji: biodro -1 / 0 / +1 (nogi 1-2 maja srodek 510),
#: kostka -1 (w dol) / +1 (poziomo) - rollout_server.map_actions_to_joints.
HIP_UNITS = ((350.0, 510.0, 663.0), (350.0, 510.0, 663.0), (350.0, 512.0, 663.0), (350.0, 512.0, 663.0))
ANKLE_UNITS = (287.0, 511.0)
#: Odczyt serw -> kat (map_joints_measurements_to_network, przeskalowane do zakresu modelu v2).
HIP_UNITS_MEAS = (350.0, 510.0, 661.0)
ANKLE_UNITS_MEAS = (288.0, 511.0)


def action_to_units(a: np.ndarray) -> list[int]:
    """Akcja w kolejnosci polityki -> nastawy s1..s8 w jednostkach serw."""
    a = np.clip(np.asarray(a, float), -1, 1)
    out = [0] * 8
    for k in range(8):
        leg = k // 2
        u = (np.interp(a[k], (-1.0, 0.0, 1.0), HIP_UNITS[leg]) if k % 2 == 0
             else np.interp(a[k], (-1.0, 1.0), ANKLE_UNITS))
        out[SERVO_OF_JOINT[k]] = int(round(u))
    return out


def units_to_joints(units: list[float]) -> np.ndarray:
    """Odczyt s1..s8 [jednostki] -> katy stawow [rad] w kolejnosci polityki (konwencja modelu v2)."""
    u = np.asarray(units, float)
    q = np.empty(8)
    for k in range(8):
        v = u[SERVO_OF_JOINT[k]]
        q[k] = (np.interp(v, HIP_UNITS_MEAS, (-HIP_CTRL, 0.0, HIP_CTRL)) if k % 2 == 0
                else np.interp(v, ANKLE_UNITS_MEAS, (ANKLE_CTRL, 0.0)))
    return q


@dataclass
class ZmqAntConfig:
    host: str = "127.0.0.1"
    #: antproxy.py: XSUB (tu publikujemy rozkazy "cmd") i XPUB (stad pomiary JSON z ant_server.py).
    pub_port: int = 3002
    sub_port: int = 3001


class ZmqAnt:
    """Prawdziwa mrowka przez antproxy / ant_server (realant-hw): rozkazy "s1 X ... s8 Y", pomiary z serw.

    NIETESTOWANE na mrowce z tego stanowiska - mapy serw sa z rollout_server.py (tam oznaczone "TODO validate");
    pierwszy test: przycisk "Test nog" w panelu rusza po kolei jednym stawem.
    """

    def __init__(self, cfg: ZmqAntConfig | None = None, context=None):
        import zmq

        self.cfg = cfg or ZmqAntConfig()
        self._zmq = zmq
        self.ctx = context or zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{self.cfg.host}:{self.cfg.pub_port}")
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{self.cfg.host}:{self.cfg.sub_port}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"{")
        self.sub.setsockopt(zmq.RCVTIMEO, 200)
        self._lock = threading.Lock()
        self._last: tuple[float, np.ndarray, float] | None = None       # (chwila odbioru, katy, ant_time [ms])
        self._prev: tuple[np.ndarray, float] | None = None
        self._vel = np.zeros(8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._rx, name="mrowka-zmq", daemon=True)
        self._thread.start()

    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                parts = self.sub.recv_multipart()
            except self._zmq.Again:
                continue
            except self._zmq.ZMQError:
                return
            try:
                d = json.loads(parts[0])
            except ValueError:
                continue
            if d.get("id") != "serial":
                continue
            try:
                q = units_to_joints([float(d[f"s{i}_angle"]) for i in range(1, 9)])
                t_ms = float(d["ant_time"])
            except (KeyError, ValueError):
                continue
            with self._lock:
                if self._prev is not None and t_ms > self._prev[1]:
                    self._vel = (q - self._prev[0]) / ((t_ms - self._prev[1]) / 1000.0)
                self._prev = (q, t_ms)
                self._last = (time.monotonic(), q, t_ms)

    def send(self, text: str) -> None:
        self.pub.send_multipart([b"cmd", text.encode() + (b"" if text.endswith("\n") else b"\n")])

    def startup(self) -> None:
        """Jak robot_policy.py: reset, wylaczenie i wlaczenie serw, reset (serwa jada do pozy resetu)."""
        self.send("reset")
        time.sleep(0.3)
        self.send("detach_servos")
        time.sleep(0.5)
        self.send("attach_servos")
        self.send("reset")

    def command(self, action: np.ndarray) -> None:
        u = action_to_units(action)
        self.send(" ".join(f"s{i + 1} {v}" for i, v in enumerate(u)))

    def joints(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(katy [rad], predkosci [rad/s], wiek odczytu [s]) albo None przed pierwszym odczytem."""
        with self._lock:
            if self._last is None:
                return None
            t, q, _ = self._last
            return q.copy(), self._vel.copy(), time.monotonic() - t

    def detach(self) -> None:
        self.send("detach_servos")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.pub.close(0)
        self.sub.close(0)


# ------------------------------------------------------------------ czasomierz
@dataclass
class Clock:
    """Zegar petli: w symulacji (testy) podawany z zewnatrz, inaczej monotoniczny."""

    fn: Callable[[], float] = field(default=time.monotonic)

    def __call__(self) -> float:
        return self.fn()
