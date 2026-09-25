"""Caly lancuch sim-2-real dla pickplace: kostka WYLACZNIE z kamer, cel odlozenia jak z panelu.

Ten sam lancuch co `test_twin_lift_from_cameras.py` (kadry symulowanych kamer ->
dopasowanie sylwetki z maska ramienia -> sledzenie kostki w dloni -> runner -> nadzor
-> fizyka), plus podkladka celu (mocap, bez kolizji) i cel odlozenia podany runnerowi
tak, jak podaje go uchwyt celu w panelu. Wynik sprawdzamy na PRAWDZIE z fizyki: kostka
lezy na celu, szczeki jej nie dotykaja, stoi, a tracker znow widzi ja z kamer.

Czas symulowany, bez watkow (jak w tescie lift). Test czeka na bazowa polityke
`pickplace-v1` w assets/policies - do tego czasu jest pomijany.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("torch")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse, pose  # noqa: E402
from lerobot_mp.twin.perception import CubeDetector, CubeTracker, TableMapper  # noqa: E402
from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.env import TARGET_RGBA, TARGET_SIZE  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, bundled_dir  # noqa: E402
from lerobot_mp.twin.rl.runner import PolicyRunner  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.ui.watch import arm_mask  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace  # noqa: E402

POLICY = bundled_dir() / "pickplace-v1" / "policy.pt"


def look(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


@pytest.mark.render
@pytest.mark.skipif(not POLICY.is_file(), reason="brak bazowej polityki pickplace-v1 w assets/policies")
def test_pickplace_policy_puts_a_cube_seen_only_by_cameras_on_the_target():
    ws = Workspace()
    K = [[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]]
    for name, eye in (("a", [0.55, -0.45, 0.45]), ("b", [0.5, 0.5, 0.5])):
        T = look(eye, [0.2, 0, 0]).tolist()
        ws.add_camera(CameraRecord(name, "sim", 640, 480, K=K, sim_pose=T, T_cam2base=T,
                                   calibration={"trusted": True}))
    pol = Policy.load(POLICY)
    task = pol.task
    assert task.name == "pickplace"
    h = task.cube_half
    twin = Twin(ws)
    try:
        twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=task.cube_mass),
                                sc.Box("target", TARGET_SIZE, (0.3, 0.1), rgba=TARGET_RGBA, free=False, mocap=True)],
                       grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        mapper = TableMapper.from_workspace(ws)
        det = CubeDetector()
        kin = RobotKinematics(ws.spec())
        rng = np.random.default_rng(5)
        pos, quat = tk.sample_cubes(task, rng, 1)
        target = tk.sample_targets(task, rng, pos)[0]
        with twin.lock:
            s = twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            dv = m.jnt_dofadr[m.body_jntadr[b]]
            jaw_adr = [m.sensor(f"cube_jaw{k}").adr[0] for k in range(2)]
            T = s.T_base2world
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            d.qpos[a:a + 3] = T[:3, :3] @ pos[0] + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            d.mocap_pos[m.body_mocapid[m.body("target").id]] = T[:3, :3] @ np.r_[target[:2], TARGET_SIZE[2]] + T[:3, 3]
            mujoco.mj_forward(m, d)

        tracker = CubeTracker.for_spec(ws.spec())
        last = {"det": None}
        clock = {"t": 0.0}

        def vision():
            frames = {c: twin.render(c) for c in ("a", "b")}
            occ = twin.render_with(lambda sc_: {c: arm_mask(sc_, c, dilate=5) for c in ("a", "b")})
            last["det"] = det.detect_frames(frames, mapper, occ)

        runner = None

        def provider():
            joints = dict(twin.status.measured) or twin.joints()
            q = kin.to_q(joints)
            g = runner.g
            return tracker.update(last["det"], kin.tcp(joints), q[g], runner.q_cmd[g], runner.limits.lo[g],
                                  clock["t"])

        runner = PolicyRunner(twin, pol, cube_provider=provider, cube_source=lambda: tracker.source)
        runner.goal = target + np.array([0.0, 0.0, 0.1])     # uchwyt celu nad blatem - runner rzutuje na blat
        assert np.allclose(runner.target, target)
        runner.start(threaded=False)
        # Zegar co 10 ms: fizyka co 20 ms, wizja co 100 ms, polityka co 50 ms.
        for k in range(1, int(30.0 / 0.01) + 1):
            clock["t"] = k * 0.01
            if k % 2 == 0:
                twin.step(0.02)
            if k % 10 == 0:
                vision()
            if k % 5 == 0 and not runner.step_once(clock["t"]):
                break
        with twin.lock:
            s = twin.scene
            Ti = inverse(s.T_base2world)
            cube = Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3]
            jaws = [float(s.data.sensordata[j]) for j in jaw_adr]
            speed = float(np.linalg.norm(s.data.qvel[dv:dv + 3]))
        d_t = float(np.hypot(*(cube[:2] - target[:2])))
        where = (f"takt {runner.status.step}, powod {runner.status.stopped_because!r}, kostka ze zrodla "
                 f"{tracker.source!r}, {d_t * 1000:.1f} mm od celu, {(cube[2] - h) * 1000:.1f} mm nad blatem, "
                 f"szczeki {jaws}, {speed * 1000:.1f} mm/s")
        assert runner.status.stopped_because == "zadanie wykonane", where
        assert d_t < task.place_radius and cube[2] - h < task.place_height, where
        assert max(jaws) < 0.5 and speed < task.settle_speed, where
        assert tracker.source == "kamery", where
    finally:
        twin.close()
