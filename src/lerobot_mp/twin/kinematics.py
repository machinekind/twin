"""Kinematyka ramienia liczona przez MuJoCo na opisie z `robots`.

Nie ma tu wlasnego lancucha kinematycznego do utrzymywania: kinematyka prosta
to `mj_kinematics` na tym samym MJCF, ktory symuluje i rysuje scene, wiec
blizniak, kalibracja i srodowisko RL nie moga sie rozjechac w geometrii.

Odwrotna kinematyka to tlumione najmniejsze kwadraty na jakobianie site'u TCP,
jak w `a1x_control.Arm.ik` z galaxeo-manipulators, z dwiema zmianami:

* **Pozycja ma pierwszenstwo przed orientacja.** A1X ma szesc stawow ramienia,
  SO-101 piec: dowolnej pelnej orientacji chwytaka nie osiagnie. Wazenie obu
  bledow w jednej sumie kwadratow tu nie dziala - radiany i metry sie nie
  porownuja, i przy celu obroconym o 90 stopni obrot odciagal ramie o 20 cm od
  zadanego punktu. Zamiast tego orientacja jest poprawiana wylacznie w
  przestrzeni zerowej jakobianu pozycji, czyli tylko ruchami, ktore pozycji nie
  zmieniaja (priorytet zadan). Punkt trafia zawsze, gdy jest osiagalny, a obrot
  jest "tak blisko, jak sie da" przy tym punkcie.
* **Wiele startow.** Pojedynczy start DLS utyka w minimum lokalnym - galaxeo
  zmierzylo 30 cm bledu na wiekszosci przestrzeni roboczej A1X. Startujemy
  z podanej pozy, z pozy spoczynkowej i z kilku losowych, i bierzemy najlepszy.

Wszystkie pozy sa w ukladzie PODSTAWY ramienia, nie swiata sceny: kamera
skalibrowana wzgledem podstawy zostaje poprawna, gdziekolwiek ramie postawimy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import mujoco
from collections.abc import Callable

import numpy as np

from .robots import RobotSpec

#: Tiki serwa STS3215 na obrot - te same, co w backendzie `feetech`.
TICKS_PER_REV = 4096


@lru_cache(maxsize=1)
def backend_gripper_ticks() -> tuple[float, float, float]:
    """(zamkniety, otwarty, zero) chwytaka w tikach serwa - z tej samej konfiguracji, co backend."""
    from ..config import load_config

    rc = load_config().robot
    return float(rc.gripper_closed_ticks), float(rc.gripper_open_ticks), float(rc.center_ticks)


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Macierz 4x4 z obrotu i przesuniecia."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    """Odwrotnosc transformacji sztywnej bez ogolnego odwracania macierzy."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass
class IKSolution:
    """Wynik odwrotnej kinematyki w jednostkach aplikacji."""

    joints: dict[str, float]
    #: Blad pozycji TCP [m] i orientacji [rad] w osiagnietej pozie.
    pos_err: float
    rot_err: float
    #: Pozycja trafiona w tolerancji. Orientacja jest miekka - patrz modul.
    ok: bool


@dataclass
class IKStep:
    """One warm step of the training solver (`RobotKinematics.ik_step`), in app units."""

    #: Every spec joint; the gripper is copied from the seed.
    joints: dict[str, float]
    #: TCP position [m] and rotation [rad] error at the RETURNED joints, never one update stale.
    pos_err: float
    rot_err: float
    #: Largest arm joint change from the seed as given (before clipping) [deg].
    jump_deg: float
    #: Updates applied; 0 when the seed already met the tolerances.
    iters: int
    #: Training's `ok`: True only when the loop left through its tolerance test.
    converged: bool


class RobotKinematics:
    """Kinematyka jednego ramienia na skompilowanym modelu MuJoCo.

    `model` moze byc samym ramieniem albo cala scena, w ktora ramie zostalo
    wstawione pod prefiksem nazw (`prefix`) - obliczenia i tak ida na osobnym
    `MjData`, wiec nie ruszaja stanu symulacji.
    """

    def __init__(self, spec: RobotSpec, model: mujoco.MjModel | None = None, prefix: str = "",
                 gripper_ticks: tuple[float, float, float] | None = None):
        self.spec = spec
        if spec.gripper_range_q is None:
            #: (zamkniety, otwarty, zero) chwytaka w tikach serwa - patrz `to_q`.
            self.gripper_ticks = gripper_ticks if gripper_ticks is not None else backend_gripper_ticks()
            self._grip_lin: tuple[float, float] | None = None
        else:
            # Chwytak liniowo na (zamkniety, otwarty) w jednostkach MJCF - tiki serwa
            # nic tu nie znacza, wiec konfiguracji backendu `feetech` nawet nie czytamy.
            self.gripper_ticks = None
            self._grip_lin = tuple(float(v) for v in spec.gripper_range_q)
        self.prefix = prefix
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        ids = [m.joint(prefix + name).id for name in spec.joints]
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([m.jnt_dofadr[i] for i in ids])
        self.lo = m.jnt_range[ids, 0].copy()
        self.hi = m.jnt_range[ids, 1].copy()
        #: Maska stawow ramienia (bez chwytaka) - tylko nimi rusza IK.
        self.arm = np.array([name != spec.gripper for name in spec.joints])
        self.site_id = m.site(prefix + spec.tcp_site).id
        self.base_id = m.body(prefix + spec.base_body).id
        self._approach = np.asarray(spec.tcp_approach, float)
        self._closing = np.asarray(spec.tcp_closing, float)
        #: Stawy sprzezone (`spec.mirror_joints`): (indeks stawu opisu, adres qpos, przelozenie).
        #: Wpisuje je `write_qpos` - przy kazdym zapisie pozy, nie tylko tutaj.
        self.mirror: list[tuple[int, int, float]] = [
            (spec.joints.index(src), int(m.jnt_qposadr[m.joint(prefix + slaved).id]), float(gain))
            for slaved, (src, gain) in spec.mirror_joints.items()
        ]
        #: Adresy qvel stawow sprzezonych - zeruje je `stop_qvel` razem ze stawami opisu.
        self.mirror_dadr = np.array(
            [m.jnt_dofadr[m.joint(prefix + slaved).id] for slaved in spec.mirror_joints], dtype=int)

    # ------------------------------------------------------------ jednostki
    def to_q(self, joints: Mapping[str, float]) -> np.ndarray:
        """Jednostki aplikacji -> wektor kata stawow MJCF [rad], w kolejnosci opisu.

        Stawy ramienia to stopnie tej samej kalibracji, wiec tylko zmiana jednostki.
        Chwytak ma 0..100, a MJCF kat szczeki. Przeliczenie idzie przez TE SAME
        tiki, co w backendzie `feetech` (`gripper_closed_ticks..gripper_open_ticks`,
        zero w `center_ticks`), przyciete do zakresu stawu z MJCF. Wczesniej 0..100
        szlo liniowo na caly zakres MJCF (-10..100 st.), a serwo przejezdza na
        0..100 tylko 60 st. (-5..55): ta sama szczeka byla w blizniaku 1,8 raza
        szerzej otwarta niz na ramieniu, polityka i sledzenie kostki widzialy
        inna liczbe niz w treningu. Brakujace stawy biora wartosc z pozy spoczynkowej.
        """
        q = np.empty(len(self.spec.joints))
        for k, name in enumerate(self.spec.joints):
            value = float(joints.get(name, self.spec.home.get(name, 0.0)))
            if name == self.spec.gripper:
                q[k] = self._grip_to_q(k, value)
            else:
                q[k] = np.radians(value)
        return q

    def _grip_to_q(self, k: int, value: float) -> float:
        if self._grip_lin is not None:
            closed, opened = self._grip_lin
            q = closed + min(max(value / 100.0, 0.0), 1.0) * (opened - closed)
            return float(min(max(q, self.lo[k]), self.hi[k]))
        closed, opened, center = self.gripper_ticks
        if abs(opened - closed) < 1e-9:                    # konfiguracja bez zakresu - liniowo na MJCF
            fraction = min(max(value / 100.0, 0.0), 1.0)
            return float(self.lo[k] + fraction * (self.hi[k] - self.lo[k]))
        ticks = closed + value / 100.0 * (opened - closed)
        q = np.radians((ticks - center) * 360.0 / TICKS_PER_REV)
        return float(min(max(q, self.lo[k]), self.hi[k]))

    def _grip_from_q(self, k: int, q: float) -> float:
        if self._grip_lin is not None:
            closed, opened = self._grip_lin
            return float((q - closed) * 100.0 / (opened - closed)) if opened != closed else 0.0
        closed, opened, center = self.gripper_ticks
        if abs(opened - closed) < 1e-9:
            span = self.hi[k] - self.lo[k]
            return float((q - self.lo[k]) / span * 100.0) if span > 0 else 0.0
        ticks = center + np.degrees(q) * TICKS_PER_REV / 360.0
        return float((ticks - closed) * 100.0 / (opened - closed))

    def from_q(self, q: np.ndarray) -> dict[str, float]:
        """Wektor MJCF [rad] -> jednostki aplikacji."""
        out: dict[str, float] = {}
        for k, name in enumerate(self.spec.joints):
            if name == self.spec.gripper:
                out[name] = self._grip_from_q(k, float(q[k]))
            else:
                out[name] = float(np.degrees(q[k]))
        return out

    def clip(self, joints: Mapping[str, float]) -> dict[str, float]:
        """Te same stawy, przyciete do zakresow z MJCF."""
        return self.from_q(np.clip(self.to_q(joints), self.lo, self.hi))

    # ------------------------------------------------------------ kinematyka
    def write_qpos(self, qpos: np.ndarray, q: np.ndarray) -> None:
        """Wpisuje poze `q` (kolejnosc opisu, jednostki MJCF) do `qpos` modelu razem ze stawami sprzezonymi.

        Kazde miejsce, ktore ustawia poze ramienia wprost (bez dynamiki), idzie tedy:
        inaczej staw sprzezony rownoscia (druga szczeka A1X) zostawalby w starym miejscu,
        a solver szarpalby go do pary dopiero w nastepnym kroku.
        """
        qpos[self.qadr] = q
        for k, adr, g in self.mirror:
            qpos[adr] = g * q[k]

    def stop_qvel(self, qvel: np.ndarray) -> None:
        """Zeruje predkosci stawow ramienia i stawow sprzezonych - para do `write_qpos`.

        Poza wpisana bez dynamiki ma stac: staw sprzezony z predkoscia z poprzedniego
        kroku odjezdzalby od wpisanej pozy i przez rownosc ciagnal za soba chwytak.
        """
        qvel[self.dadr] = 0.0
        qvel[self.mirror_dadr] = 0.0

    def _apply(self, q: np.ndarray) -> None:
        self.write_qpos(self.data.qpos, q)
        mujoco.mj_kinematics(self.model, self.data)

    def _base_inv(self) -> np.ndarray:
        d = self.data
        return inverse(pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id]))

    def tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        """Poza TCP w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[self.site_id].reshape(3, 3), d.site_xpos[self.site_id])

    def body(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """Poza dowolnego ciala ramienia w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.body(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.xmat[i].reshape(3, 3), d.xpos[i])

    def site(self, name: str, joints: Mapping[str, float]) -> np.ndarray:
        """Poza dowolnego site'u ramienia w ukladzie podstawy (4x4)."""
        self._apply(self.to_q(joints))
        i = self.model.site(self.prefix + name).id
        d = self.data
        return self._base_inv() @ pose(d.site_xmat[i].reshape(3, 3), d.site_xpos[i])

    def tool_axes(self, joints: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(kierunek podejscia, os zamykania szczek) w ukladzie podstawy."""
        R = self.tcp(joints)[:3, :3]
        return R @ self._approach, R @ self._closing

    # -------------------------------------------------------------------- IK
    def ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed: Mapping[str, float] | None = None,
        *,
        rot_gain: float = 0.5,
        tol: float = 1e-3,
        iters: int = 300,
        damping: float = 0.02,
        restarts: int = 6,
        rng: np.random.Generator | None = None,
        accept: Callable[[Joints], bool] | None = None,
    ) -> IKSolution:
        """Stawy, przy ktorych TCP trafia w `position`, obrocony jak najblizej `rotation`.

        Obie wielkosci w ukladzie podstawy. Chwytak zostaje taki, jak w `seed`.
        Sposrod startow, ktore trafily w punkt, wygrywa ten z najmniejszym
        bledem obrotu; jesli zaden nie trafil - ten, ktory byl najblizej.
        """
        seed = dict(seed) if seed is not None else dict(self.spec.home)
        rng = rng if rng is not None else np.random.default_rng(0)
        q_seed = self.to_q(seed)

        starts = [q_seed, self.to_q(self.spec.home)]
        for _ in range(restarts):
            q = q_seed.copy()
            q[self.arm] = rng.uniform(self.lo[self.arm], self.hi[self.arm])
            starts.append(q)

        best: tuple[tuple[bool, float], np.ndarray, float, float] | None = None
        target = np.asarray(position, float)
        for q0 in starts:
            q, e_pos, e_rot = self._solve(q0, target, rotation, rot_gain, iters, damping, tol)
            hit = e_pos < tol
            # A hit the caller refuses (a collision) counts as a miss, so another start can win.
            if hit and accept is not None and not accept(self.from_q(q)):
                hit = False
            # Hits first; among hits the rotation error (when one was asked for), then the pose
            # nearest the seed, so a reachable point does not come back on the other elbow branch.
            far = float(np.max(np.abs(q - q_seed)))
            key = (not hit, round(e_rot, 2) if (hit and rotation is not None) else 0.0, far if hit else e_pos)
            if best is None or key < best[0]:
                best = (key, q, e_pos, e_rot)

        assert best is not None
        _, q, e_pos, e_rot = best
        return IKSolution(self.from_q(q), float(e_pos), float(e_rot), bool(e_pos < tol))

    def ik_track(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed: Mapping[str, float] | None = None,
        *,
        iters: int = 30,
        rot_gain: float = 0.5,
        tol: float = 1e-3,
        damping: float = 0.02,
    ) -> IKSolution:
        """Jeden start z `seed` (bez restartow i bez `accept`) - do sledzenia punktu w kazdym takcie petli.

        Cel przesuwa sie miedzy taktami o milimetry, wiec rozwiazanie z poprzedniego taktu
        jest dobrym startem: `_solve` zbiega w kilku iteracjach (~0,8 ms razem z `config_clear`),
        a pelne `ik` z osmioma startami to kilkadziesiat ms. Wynik zostaje blisko `seed` - bez
        przeskoku na inna galaz lokcia, ktory moglby wybrac losowy start.
        """
        seed = dict(seed) if seed is not None else dict(self.spec.home)
        q, e_pos, e_rot = self._solve(self.to_q(seed), np.asarray(position, float), rotation, rot_gain, iters,
                                      damping, tol)
        return IKSolution(self.from_q(q), float(e_pos), float(e_rot), bool(e_pos < tol))

    def ik_step(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        seed: Mapping[str, float],
        *,
        iters: int = 60,
        damping: float = 0.05,
        tol_pos: float = 1e-3,
        tol_rot: float = 5e-3,
        limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> IKStep:
        """One warm start of the grasp training solver (galaxeo `a1x_control.Arm.ik`), transcribed.

        6-D damped least squares, 0.3 rad step cap, no restarts, no randomness: the joints,
        residual included, are the ones the grasp student was trained on. `ik` lands elsewhere
        and takes about 40 ms.

        `seed` must hold every arm joint (`to_q` would silently fill a missing one from home).
        `limits` [app units], e.g. `Twin.joint_limits()`, narrow the MJCF ranges of the arm
        joints, so a step the supervisor would clamp comes back not converged. Base frame.

        The tolerance test runs before each update, as in training. When the loop runs out,
        the errors are recomputed at the final joints, but `converged` stays False even if they
        are inside the tolerances: training refuses that step, and taking it would move the
        command away from what the policy saw.
        """
        missing = [name for name, arm in zip(self.spec.joints, self.arm) if arm and name not in seed]
        if missing:
            raise KeyError(f"brak stawow ramienia w seed: {', '.join(missing)}")
        q_seed = self.to_q(seed)
        lo, hi = self.lo.copy(), self.hi.copy()
        if limits is not None:
            for k, name in enumerate(self.spec.joints):
                if self.arm[k] and name in limits:
                    lo[k] = max(lo[k], np.radians(limits[name][0]))
                    hi[k] = min(hi[k], np.radians(limits[name][1]))
        q = np.clip(q_seed, lo, hi)

        m, d = self.model, self.data
        cols = self.dadr[self.arm]
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        want_quat = np.zeros(4)
        cur_quat = np.zeros(4)
        err_quat = np.zeros(4)
        w = np.zeros(3)

        self._apply(q)
        base_R = d.xmat[self.base_id].reshape(3, 3).copy()
        target = base_R @ np.asarray(position, float) + d.xpos[self.base_id]
        mujoco.mju_mat2Quat(want_quat, (base_R @ np.asarray(rotation, float)).ravel())

        def pose_error() -> np.ndarray:
            self._apply(q)
            mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site_id])
            mujoco.mju_negQuat(cur_quat, cur_quat)
            mujoco.mju_mulQuat(err_quat, want_quat, cur_quat)
            mujoco.mju_quat2Vel(w, err_quat, 1.0)
            return target - d.site_xpos[self.site_id]

        converged = False
        updates = 0
        for _ in range(iters):
            e_p = pose_error()
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(w) < tol_rot:
                converged = True
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), np.concatenate([e_p, w]))
            step = min(1.0, 0.3 / (np.linalg.norm(dq) + 1e-9))
            q[self.arm] = np.clip(q[self.arm] + step * dq, lo[self.arm], hi[self.arm])
            updates += 1
        if not converged:
            e_p = pose_error()

        joints = self.from_q(q)
        if self.spec.gripper in seed:
            joints[self.spec.gripper] = float(seed[self.spec.gripper])
        jump = float(np.degrees(np.max(np.abs(q[self.arm] - q_seed[self.arm]))))
        return IKStep(joints, float(np.linalg.norm(e_p)), float(np.linalg.norm(w)), jump, updates, converged)

    def _solve(
        self,
        q: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray | None,
        rot_gain: float,
        iters: int,
        damping: float,
        tol: float,
    ) -> tuple[np.ndarray, float, float]:
        """Jeden start: pozycja pseudo-odwrotnoscia, obrot gradientem w przestrzeni zerowej."""
        m, d = self.model, self.data
        q = np.clip(q.copy(), self.lo, self.hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.dadr[self.arm]
        n = int(self.arm.sum())
        want_quat = np.zeros(4)
        cur_quat = np.zeros(4)
        err_quat = np.zeros(4)
        w = np.zeros(3)
        e_pos = e_rot = float("inf")

        self._apply(q)
        # Podstawa sie nie rusza, wiec cel przeliczamy do ukladu swiata raz.
        base = pose(d.xmat[self.base_id].reshape(3, 3), d.xpos[self.base_id])
        target = base[:3, :3] @ position + base[:3, 3]
        if rotation is not None:
            mujoco.mju_mat2Quat(want_quat, (base[:3, :3] @ np.asarray(rotation, float)).ravel())

        for _ in range(iters):
            self._apply(q)
            e_p = target - d.site_xpos[self.site_id]
            e_pos = float(np.linalg.norm(e_p))
            if rotation is not None:
                mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site_id])
                mujoco.mju_negQuat(cur_quat, cur_quat)
                mujoco.mju_mulQuat(err_quat, want_quat, cur_quat)
                mujoco.mju_quat2Vel(w, err_quat, 1.0)
                e_rot = float(np.linalg.norm(w))
            else:
                e_rot = 0.0

            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            Jp = jacp[:, cols]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping**2 * np.eye(3))
            dq = Jp_pinv @ e_p
            if rotation is not None:
                # Zadanie drugorzedne rzutowane na przestrzen zerowa pozycji:
                # ruch, ktory poprawia obrot, a punktu TCP nie przesuwa.
                null = np.eye(n) - Jp_pinv @ Jp
                dq_rot = null @ (rot_gain * jacr[:, cols].T @ w)
                dq = dq + dq_rot
                if e_pos < tol * 0.5 and float(np.linalg.norm(dq_rot)) < 1e-4:
                    break  # punkt trafiony, a obrotu da sie juz poprawic tylko kosztem punktu
            elif e_pos < tol * 0.5:
                break

            # Ograniczenie kroku jak w galaxeo: duzy skok przy zlym uwarunkowaniu
            # przerzuca rozwiazanie na inna galaz i iteracja juz z niej nie wraca.
            step = min(1.0, 0.3 / (float(np.linalg.norm(dq)) + 1e-9))
            q[self.arm] = np.clip(q[self.arm] + step * dq, self.lo[self.arm], self.hi[self.arm])

        self._apply(q)
        e_pos = float(np.linalg.norm(target - d.site_xpos[self.site_id]))
        return q, e_pos, e_rot
