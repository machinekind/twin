"""Czy ramie w danej pozie - i po drodze do niej - nie wchodzi w stol, obiekty ani w siebie.

Sprawdza to MuJoCo na scenie blizniaka (ramie + stol + to, co o stanowisku
wiadomo), na wlasnym `MjData`, wiec nie rusza stanu symulacji ani prawdziwego
ramienia. Uzywa tego fala kalibracyjna i teleoperacja z UI.

Dwie rzeczy, ktorych nauczyl galaxeo-manipulators:

* **Sprawdzac trzeba droge, nie tylko cel.** Poza docelowa moze byc czysta,
  a przejazd do niej przez stol. Probkujemy odcinek w przestrzeni stawow co
  0,04 rad for hinges and 1 mm for sliders - w galaxeo dziesiec probek na dwusekundowy ruch
  przepuszczalo palec przez butelke.
* Only fixed mounting contacts are permanently allowed. A collision involving
  a movable link at home does not create an exemption. Named grasp objects may
  touch fingertip geoms; palm, table and arm contacts remain prohibited.
* Object poses are copied from the current scene before each path check.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy

import mujoco
import numpy as np

from .kinematics import RobotKinematics
from .scene import PREFIX, Scene


class CollisionChecker:
    def __init__(self, scene: Scene, *, grasp_objects: tuple[str, ...] = (), clearance_m: float = 0.0):
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError("Collision clearance must be finite and nonnegative")
        self.scene = scene
        # The environmental clearance must not change the dynamics/contact model.
        self.model = copy.copy(scene.model) if clearance_m else scene.model
        if clearance_m:
            self.model.geom_margin[:] = np.maximum(self.model.geom_margin, clearance_m)
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
        self._sync_scene()
        # Only fixed mounting contacts are permanent exceptions. A collision in
        # the home pose must not whitelist that link/object pair forever.
        fixed = m.body_weldid[m.geom_bodyid] == 0
        self.allowed = {p for p in self._contacts(scene.cfg.robot.home)
                        if all(fixed[g] for g in p) and self.robot_geom[p[0]] != self.robot_geom[p[1]]}
        tips = {m.geom(PREFIX + name).id for name in scene.cfg.robot.fingertips if name}
        objects = {m.body(name).id for name in grasp_objects if mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0}
        self._tips = tips
        self._grasp_bodies = objects
        self.allowed.update((min(tip, g), max(tip, g)) for tip in tips
                            for g in range(m.ngeom) if m.geom_bodyid[g] in objects)

    def forbidden_pairs(self) -> np.ndarray:
        """Symmetric geom-pair mask shared by CPU and GPU training contacts."""
        mask = self.robot_geom[:, None] | self.robot_geom[None, :]
        for a, b in self.allowed:
            mask[a, b] = mask[b, a] = False
        return mask

    def _sync_scene(self) -> None:
        """Use current object poses, not their reset positions. Caller holds the scene lock."""
        self.data.qpos[:] = self.scene.data.qpos
        self.data.mocap_pos[:] = self.scene.data.mocap_pos
        self.data.mocap_quat[:] = self.scene.data.mocap_quat

    def blocked_gripper_opening(self) -> float | None:
        """Current opening when both fingertips contact the same allowed object.

        A closing motor setpoint lies inside the grasped object; it is not a
        reachable geometry pose. Only for an existing bilateral fingertip contact
        may the caller check that closing command at the measured opening instead.
        No other contact exemption is introduced, and no motor command is changed.
        """
        if len(self._tips) != 2 or not self._grasp_bodies:
            return None
        self._sync_scene()
        mujoco.mj_fwdPosition(self.model, self.data)
        touched = {body: set() for body in self._grasp_bodies}
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for tip, other in ((g1, g2), (g2, g1)):
                body = int(self.model.geom_bodyid[other])
                if tip in self._tips and body in touched:
                    touched[body].add(tip)
        if not any(tips == self._tips for tips in touched.values()):
            return None
        return self.kin.from_q(self.data.qpos[self.kin.qadr])[self.kin.spec.gripper]

    def _contacts(self, joints: Mapping[str, float]) -> set[tuple[int, int]]:
        d = self.data
        self.kin.write_qpos(d.qpos, self.kin.to_q(joints))
        mujoco.mj_fwdPosition(self.model, d)
        out = set()
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            # Mechanical joint housings can be separated by less than the
            # environmental buffer. Still reject actual self penetration.
            if c.dist > 0 and self.robot_geom[g1] and self.robot_geom[g2]:
                continue
            if self.robot_geom[g1] or self.robot_geom[g2]:
                out.add((min(g1, g2), max(g1, g2)))
        return out

    def _describe(self, pair: tuple[int, int]) -> str:
        m = self.model
        names = []
        for g in pair:
            n = m.geom(g).name or m.body(m.geom_bodyid[g]).name
            names.append(n.removeprefix(PREFIX))
        return " z ".join(names)

    def config_clear(self, joints: Mapping[str, float]) -> str | None:
        """None, gdy poza jest czysta; inaczej opis pierwszej kolizji."""
        self._sync_scene()
        return self._config_clear(joints)

    def _config_clear(self, joints: Mapping[str, float]) -> str | None:
        if not all(np.isfinite(float(v)) for v in joints.values()):
            return "non-finite joint target"
        bad = self._contacts(joints) - self.allowed
        return f"kolizja: {self._describe(min(bad))}" if bad else None

    def path_clear(self, start: Mapping[str, float], end: Mapping[str, float],
                   step_rad: float = 0.04) -> str | None:
        """None, gdy caly liniowy przejazd w przestrzeni stawow jest czysty."""
        if not np.isfinite(step_rad) or step_rad <= 0:
            raise ValueError("step_rad must be finite and positive")
        self._sync_scene()
        q0, q1 = self.kin.to_q(start), self.kin.to_q(end)
        if not np.isfinite(q0).all() or not np.isfinite(q1).all():
            return "non-finite joint target"
        kinds = [self.model.joint(PREFIX + name).type[0] for name in self.kin.spec.joints]
        step = np.where(np.asarray(kinds) == mujoco.mjtJoint.mjJNT_SLIDE, 0.001, step_rad)
        n = max(1, int(np.ceil(np.max(np.abs(q1 - q0) / step))))
        # Include the measured start. With a short move, sampling only the end
        # can miss a near-table tracking pose while the commanded end is clear.
        for s in np.linspace(0.0, 1.0, n + 1):
            why = self._config_clear(self.kin.from_q(q0 + s * (q1 - q0)))
            if why:
                return f"{why} w {s:.0%} drogi"
        return None
