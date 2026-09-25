"""Percepcja z kalibrowanych kamer: jeden swiadek nigdy nie podaje polityce podniesionego ducha."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.kinematics import pose  # noqa: E402
from lerobot_mp.twin.perception import CubeDetector, CubeTracker, TableMapper  # noqa: E402

K = np.array([[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]])


def look_at(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


@pytest.mark.render
def test_lift_from_one_camera_never_feeds_a_lifted_ghost_and_recovers_a_dropped_cube():
    """Caly lancuch lift-v3 z TYLKO kamera `a` (jak `test_twin_lift_from_cameras`, ale jeden swiadek).

    Seed 5: pierwszy chwyt wypuszcza kostke 3-5 cm dalej. Przed poprawka: duch z
    n_cameras=0 szedl do polityki jako "kamery" przy kostce 1,6 cm nad blatem, a potem
    lezaca obok kostka (widziana dobrze, ale przez jedna kamere przy dloni) byla
    pomijana - 130 taktow chwytania pustego miejsca i stop "kamery jej nie widza".
    """
    pytest.importorskip("torch")
    from lerobot_mp.twin.kinematics import RobotKinematics, inverse
    from lerobot_mp.twin.rl import task as tk
    from lerobot_mp.twin.rl.policy import Policy, bundled_dir
    from lerobot_mp.twin.rl.runner import PolicyRunner
    from lerobot_mp.twin.runtime import Twin
    from lerobot_mp.twin.ui.watch import arm_mask
    from lerobot_mp.twin.workspace import CameraRecord, Workspace

    path = bundled_dir() / "lift-v3" / "policy.pt"
    if not path.is_file():
        pytest.skip("brak bazowej polityki lift-v3 w assets/policies")
    ws = Workspace()
    T = look_at([0.55, -0.45, 0.45], [0.2, 0, 0]).tolist()
    ws.add_camera(CameraRecord("a", "sim", 640, 480, K=K.tolist(), sim_pose=T, T_cam2base=T,
                               calibration={"trusted": True}))
    pol = Policy.load(path)
    h = pol.task.cube_half
    twin = Twin(ws)
    try:
        twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=pol.task.cube_mass)], grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        mapper = TableMapper.from_workspace(ws)
        kin = RobotKinematics(ws.spec())
        pos, quat = tk.sample_cubes(pol.task, np.random.default_rng(5), 1)
        with twin.lock:
            s = twin.scene
            b = s.model.body("cube").id
            a = s.model.jnt_qposadr[s.model.body_jntadr[b]]
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, s.T_base2world[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            s.data.qpos[a:a + 3] = s.T_base2world[:3, :3] @ pos[0] + s.T_base2world[:3, 3]
            s.data.qpos[a + 3:a + 7] = qw
            mujoco.mj_forward(s.model, s.data)

        def truth():
            with twin.lock:
                s = twin.scene
                Ti = inverse(s.T_base2world)
                return Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3]

        tracker, det = CubeTracker(), CubeDetector()
        last, clock, log = {"det": None}, {"t": 0.0}, []

        def provider():
            joints = dict(twin.status.measured) or twin.joints()
            q = kin.to_q(joints)
            d, last["det"] = last["det"], None                # tylko swieze detekcje, jak w panelu
            out = tracker.update(d, kin.tcp(joints), q[5], runner.q_cmd[5], runner.limits.lo[5], clock["t"])
            log.append((truth()[2] - h, tracker.source))
            return out

        runner = PolicyRunner(twin, pol, cube_provider=provider)
        runner.start(threaded=False)
        for k in range(1, int(20.0 / 0.01) + 1):
            clock["t"] = k * 0.01
            if k % 2 == 0:
                twin.step(0.02)
            if k % 10 == 0:
                occ = twin.render_with(lambda sc_: {"a": arm_mask(sc_, "a", dilate=5)})
                last["det"] = det.detect_frames({"a": twin.render("a")}, mapper, occ, t=clock["t"])
            if k % 5 == 0 and not runner.step_once(clock["t"]):
                break
        lifted_from_cameras = [lift for lift, src in log if lift > 0.01 and src == "kamery"]
        assert not lifted_from_cameras, f"podniesiona kostka jako 'kamery' {len(lifted_from_cameras)} razy"
        assert runner.status.stopped_because == "zadanie wykonane", runner.status.stopped_because
        assert truth()[2] - h > pol.task.lift_height
    finally:
        twin.close()
