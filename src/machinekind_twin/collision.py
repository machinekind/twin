"""Czy ramie w danej pozie - i po drodze do niej - nie wchodzi w stol, obiekty ani w siebie.

Sprawdza to MuJoCo na scenie blizniaka (ramie + stol + to, co o stanowisku
wiadomo), na wlasnym `MjData`, wiec nie rusza stanu symulacji ani prawdziwego
ramienia. Uzywa tego fala kalibracyjna i teleoperacja z UI.

Dwie rzeczy, ktorych nauczyl galaxeo-manipulators:

* **Sprawdzac trzeba droge, nie tylko cel.** Poza docelowa moze byc czysta,
  a przejazd do niej przez stol. Probkujemy odcinek w przestrzeni stawow co
  0,04 rad najwiekszego ruchu - w galaxeo dziesiec probek na dwusekundowy ruch
  przepuszczalo palec przez butelke.
* **Kontakty montazowe sa dozwolone.** Pary cial stykajace sie juz w pozie
  spoczynkowej (podstawa na blacie) nie sa kolizja; kazda nowa para - jest.
  Dzieki temu sprawdzacz dziala dla dowolnego ramienia bez recznej listy wyjatkow.

With a `clearance` the arm must also stay that far from everything in the scene that is not the
arm itself: the table, objects, another robot. The checker works on a copy of the model whose other
geoms are grown by the clearance (boxes, spheres, capsules, cylinders, ellipsoids; planes move up),
so any body added to the scene is kept clear of with no further change. Meshes cannot grow: they
get the clearance as a collision margin, which MuJoCo honours only for some pairs - give a body
that must be kept clear primitive collision geoms. As before, pairs touching at rest (the base
on the table) are allowed.

Other robots of the station (`scene.PassiveRobot`) are obstacles like the table, and their
collision geoms reach `PASSIVE_CLEARANCE` further out, so every check keeps that gap from them,
with or without a `clearance`. A `clearance` grows them only by what it adds beyond that gap.
`closer` is the twin loop's guard against driving into one.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import mujoco
import numpy as np

from .kinematics import RobotKinematics
from .scene import PASSIVE_CLEARANCE, PREFIX, Scene


class CollisionChecker:
    def __init__(self, scene: Scene, clearance: float = 0.0):
        """`clearance` [m]: how close the arm may come to anything in the scene that is not the arm; 0 = touching."""
        self.clearance = float(clearance)
        self.model = scene.model if self.clearance <= 0 else copy.deepcopy(scene.model)
        self.data = mujoco.MjData(self.model)
        self.kin = RobotKinematics(scene.cfg.robot, self.model, PREFIX)
        m = self.model
        base = m.body(PREFIX + scene.cfg.robot.base_body).id
        in_robot = np.zeros(m.nbody, bool)
        for b in range(m.nbody):                        # poddrzewo podstawy ramienia
            p = b
            while p > 0 and p != base:
                p = m.body_parentid[p]
            in_robot[b] = p == base
        self.robot_geom = in_robot[m.geom_bodyid]
        self.passive_geom = scene.passive_geom
        #: How far each geom was grown [m]: a contact with it comes that much early.
        self.grown = np.zeros(m.ngeom)
        if self.clearance > 0:
            self._grow(np.flatnonzero(~self.robot_geom & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))))
        self.T_base2world = scene.T_base2world
        self.allowed = set(self._contacts(scene.cfg.robot.home))

    def _grow(self, geoms: np.ndarray) -> None:
        m, G = self.model, mujoco.mjtGeom
        for g in geoms:
            t = int(m.geom_type[g])
            if t not in (int(G.mjGEOM_BOX), int(G.mjGEOM_ELLIPSOID), int(G.mjGEOM_SPHERE),
                         int(G.mjGEOM_CAPSULE), int(G.mjGEOM_CYLINDER), int(G.mjGEOM_PLANE)):
                m.geom_margin[g] = max(float(m.geom_margin[g]), self.clearance)
                continue
            # A passive robot's margin keeps PASSIVE_CLEARANCE already: grow it only by the rest.
            c = self.clearance - (PASSIVE_CLEARANCE if self.passive_geom[g] else 0.0)
            if c <= 0:
                continue
            if t in (int(G.mjGEOM_BOX), int(G.mjGEOM_ELLIPSOID)):
                m.geom_size[g, :3] += c
            elif t == int(G.mjGEOM_SPHERE):
                m.geom_size[g, 0] += c
            elif t in (int(G.mjGEOM_CAPSULE), int(G.mjGEOM_CYLINDER)):
                m.geom_size[g, :2] += c
            else:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                m.geom_pos[g] += R.reshape(3, 3)[:, 2] * c
                self.grown[g] = c
                continue
            m.geom_rbound[g] += c
            m.geom_aabb[g, 3:] += c
            self.grown[g] = c
        # The midphase culls by the bodies' bounding volume hierarchies: they grow with their geoms.
        for b in set(m.geom_bodyid[geoms].tolist()):
            adr, num = int(m.body_bvhadr[b]), int(m.body_bvhnum[b])
            if adr >= 0:
                m.bvh_aabb[adr:adr + num, 3:] += self.clearance

    def _own(self, pair: tuple[int, int]) -> bool:
        return bool(self.robot_geom[pair[0]] and self.robot_geom[pair[1]])

    def _contacts(self, joints: Mapping[str, float]) -> dict[tuple[int, int], float]:
        """{geom pair: distance [m] of the real geoms, negative = penetration} for every pair with a geom of the
        arm. Near a grown box's edge or corner the distance comes out shorter than it is."""
        d = self.data
        self.kin.write_qpos(d.qpos, self.kin.to_q(joints))
        mujoco.mj_fwdPosition(self.model, d)
        out: dict[tuple[int, int], float] = {}
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.robot_geom[g1] or self.robot_geom[g2]:
                pair = (min(g1, g2), max(g1, g2))
                dist = float(c.dist) + float(self.grown[g1] + self.grown[g2])
                out[pair] = min(out.get(pair, np.inf), dist)
        return out

    def set_world(self, data: mujoco.MjData) -> None:
        """Stan swiata (obiekty: qpos i mocap) z `data` tego samego modelu; ramie i tak ustawia kazde
        sprawdzenie. Dozwolone kontakty zostaja z budowy."""
        self.data.qpos[:] = data.qpos
        self.data.mocap_pos[:] = data.mocap_pos
        self.data.mocap_quat[:] = data.mocap_quat

    def _describe(self, pair: tuple[int, int]) -> str:
        m = self.model
        names = []
        for g in pair:
            n = m.geom(g).name or m.body(m.geom_bodyid[g]).name
            names.append(n.removeprefix(PREFIX))
        return " z ".join(names)

    def config_clear(self, joints: Mapping[str, float]) -> str | None:
        """None, gdy poza jest czysta; inaczej opis pierwszej kolizji (albo pary blizej niz `clearance`)."""
        bad = sorted((pair, dist) for pair, dist in self._contacts(joints).items() if pair not in self.allowed)
        if not bad:
            return None
        pair, dist = bad[0]
        if self._own(pair) or dist < 0:
            return f"kolizja: {self._describe(pair)}"
        return f"za blisko: {self._describe(pair)} ({dist * 100:.1f} cm)"

    def sight_blocked(self, joints: Mapping[str, float], start: np.ndarray, end: np.ndarray, skip_body: str) -> bool:
        """Whether something in the scene lies on the segment `start` -> `end` (base frame) with the arm at
        `joints`. `skip_body` (of the arm: the hand holding the card) is left out."""
        d = self.data
        self.kin.write_qpos(d.qpos, self.kin.to_q(joints))
        mujoco.mj_kinematics(self.model, d)
        R, t = self.T_base2world[:3, :3], self.T_base2world[:3, 3]
        a, b = R @ np.asarray(start, float) + t, R @ np.asarray(end, float) + t
        geom = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(self.model, d, a, b - a, None, 1, self.model.body(PREFIX + skip_body).id, geom)
        return 0.0 <= hit < 1.0

    def path_clear(self, start: Mapping[str, float], end: Mapping[str, float],
                   step_rad: float = 0.04) -> str | None:
        """None, gdy caly liniowy przejazd w przestrzeni stawow jest czysty."""
        q0, q1 = self.kin.to_q(start), self.kin.to_q(end)
        travel = float(np.max(np.abs(q1 - q0)))
        n = max(1, int(np.ceil(travel / step_rad)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            why = self.config_clear(self.kin.from_q(q0 + s * (q1 - q0)))
            if why:
                return f"{why} w {s:.0%} drogi"
        return None

    def passive_gap(self, joints: Mapping[str, float]) -> tuple[float, str] | None:
        """The arm's closest approach to a passive robot in this pose, when nearer than
        `PASSIVE_CLEARANCE` (or a larger `clearance`): (distance [m], negative = overlap; what meets
        what). None = farther."""
        if not self.passive_geom.any():
            return None
        near = [(dist, pair) for pair, dist in self._contacts(joints).items()
                if self.passive_geom[list(pair)].any()]
        if not near:
            return None
        dist, pair = min(near)
        return dist, self._describe(pair)

    def closer(self, start: Mapping[str, float], end: Mapping[str, float]) -> str | None:
        """Why the step from `start` to `end` is refused: `end` is within `PASSIVE_CLEARANCE` of a passive
        robot and nearer to it than `start`. None = allowed, so the arm can always back away."""
        near = self.passive_gap(end)
        if near is None:
            return None
        was = self.passive_gap(start)
        if was is not None and near[0] >= was[0] - 1e-4:
            return None
        return (f"za blisko innego robota: {near[1]} "
                f"({max(near[0], 0.0) * 100:.1f} cm, minimum {PASSIVE_CLEARANCE * 100:.0f} cm)")
