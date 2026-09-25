"""Panel blizniaka: startuje, odswieza sie i obsluguje kamery symulowane bez przegladarki.

Dalej - to, czego panel pilnuje przed prawdziwym ramieniem: jeden wlasciciel ramienia,
uchwyt TCP na tej samej galezi IK, swiezosc detekcji kostki, kalibracja i zadania w tle.
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("mujoco")

from lerobot_mp.twin.ui import jobs  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Client:
    def __init__(self):
        self.notes: list[tuple[str, str]] = []

    def add_notification(self, title, body, **kw):
        self.notes.append((title, body))


class _Event:
    """Zdarzenie od uzytkownika (z `client`), jak z przegladarki."""

    def __init__(self):
        self.client = _Client()


def _look_at_pose(eye, target):
    from lerobot_mp.twin.kinematics import pose
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


@pytest.mark.render
def test_panel_starts_adds_a_simulated_camera_and_maps_the_table(tmp_path):
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path / "twin.json", host="127.0.0.1", port=free_port())
    try:
        app._tick_slow()
        app._add_sim_camera(app.T_b2w @ _look_at_pose([0.6, -0.4, 0.45], [0.2, 0.0, 0.0]))
        rec = app.ws.cameras[0]
        assert rec.simulated and rec.sim_pose is not None
        frames = app._tick_slow()
        assert rec.name in frames and frames[rec.name].shape == (480, 640, 3)
        assert app.frame_times[rec.name] > 0
        # "uznaj prawde za kalibracje" -> kamera zaufana -> mapa stolu z niej
        rec.T_cam2base = [list(r) for r in rec.sim_pose]
        rec.calibration = {"trusted": True, "reason": "", "rms_px": 0.0}
        app._refresh_cameras()
        app._tick_map(frames)
        assert app.mapper is not None and rec.name in app.mapper.cameras
        assert (tmp_path / "twin.json").is_file()
        app.twin.connect("sim")
        app._tick_arm()
        assert app.twin.connected
    finally:
        app.close()


# ------------------------------------------------------------------ jeden panel na reszte testow
@pytest.fixture(scope="module")
def shared_app(tmp_path_factory):
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path_factory.mktemp("panel") / "twin.json", host="127.0.0.1", port=free_port())
    yield app
    app.close()


@pytest.fixture
def app(shared_app):
    a = shared_app
    yield a
    if a.runner is not None:
        a.runner.stop("koniec testu")
    for job in (a.calib_job, a.sysid_job, a.intr_job):
        job.stop()
        job.wait(10.0)
        job.state, job.result, job.data = jobs.IDLE, None, {}
    a.twin.disconnect()
    a.arm_engage.value = False
    a.tcp_gizmo_on.value = False
    for owner in (a.twin.owner,):
        if owner is not None:
            a.twin.release(owner)


def test_tcp_gizmo_rejects_a_jump_to_the_other_ik_branch(app):
    """Z shoulder_pan 109,5 st. przeciagniecie o 1 cm dawalo "poprawne" IK 180 st. dalej."""
    spec = app.ws.spec()
    start = dict(spec.home, shoulder_pan=109.5)
    kin = app._kin_gizmo
    p = kin.tcp(start)[:3, 3]
    tang = np.array([-p[1], p[0], 0.0]) / np.hypot(p[0], p[1])
    old = kin.ik(p - 0.01 * tang, seed=start, restarts=2)      # tak liczyl uchwyt przed poprawka
    assert old.ok and max(abs(old.joints[k] - start[k]) for k in start if k != spec.gripper) > 90

    with app.twin.lock:
        app.twin.scene.set_joints(start)
    app.twin.connect("sim", threaded=False)
    app.tcp_gizmo_on.value = True
    app.arm_engage.value = True
    app._on_engage(_Event())
    assert app.twin.owner == "panel"

    def drag(p_base):
        app.tcp_gizmo.position = (app.T_b2w @ np.r_[p_base, 1.0])[:3]
        app._on_tcp_gizmo()
        return app.twin._target

    assert drag(p - 0.01 * tang) is None                        # odrzucone, cel bez zmian
    app._gizmo_seed = None
    target = drag(p + 0.01 * tang)                              # w druga strone - ta sama galaz
    assert target is not None
    assert max(abs(target[k] - start[k]) for k in start if k != spec.gripper) < 5.0


def test_reconnect_ends_a_running_wave_and_it_never_drives_the_new_arm(app):
    """Fala w sim + "Polacz" = reszta fali na prawdziwym ramieniu, bez potwierdzenia karty."""
    home = dict(app.ws.spec().home)
    app.twin.connect("sim")
    app._take_arm(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    moves = []

    def wave(job):
        arm = jobs._OwnedArm(app.twin, job)
        for k in range(1000):
            arm.move(dict(home, shoulder_pan=20.0 if k % 2 else -20.0), 0.3)
            moves.append(k)

    app.calib_job.start(wave)
    time.sleep(0.8)
    app.calib_confirm.value = True
    app._connect("sim", None)
    assert app.calib_job.wait(5.0)
    assert app.calib_job.state == jobs.FAILED
    assert not app.calib_confirm.value                          # nowe ramie - nowe potwierdzenie
    n = len(moves)
    time.sleep(0.5)
    assert len(moves) == n and app.twin.owner is None and not app.twin.status.engaged


def test_card_wave_goes_home_only_while_it_still_owns_the_arm(app, monkeypatch):
    import lerobot_mp.twin.calib.session as session_mod

    home = dict(app.ws.spec().home)
    app.twin.connect("sim", threaded=False)

    class InterruptedSession:
        def __init__(self, robot, *a, **kw):
            self.robot, self.done = robot, False

        def step(self):
            app.twin.preempt("STOP w trakcie fali")             # miedzy przejazdami
            self.robot.move(home, 0.2)

    homes = []
    monkeypatch.setattr(session_mod, "Session", InterruptedSession)
    monkeypatch.setattr(app.twin, "home", lambda: homes.append(1))
    app.twin.claim(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    app.calib_job.start(lambda job: jobs.run_card_calibration(job, app.twin, [], quick=True))
    assert app.calib_job.wait(60.0)
    assert app.calib_job.state == jobs.FAILED and "odebrane" in app.calib_job.error
    assert homes == [] and app.twin.owner is None               # ramie nie nasze - bez jazdy do domu
    monkeypatch.undo()

    class DoneSession:
        def __init__(self, robot, *a, **kw):
            self.done = True

        def solve(self):
            return "wynik"

    monkeypatch.setattr(session_mod, "Session", DoneSession)
    app.twin.claim(jobs.CALIB_OWNER, preempt=app.calib_job.stop)
    app.calib_job.start(lambda job: jobs.run_card_calibration(job, app.twin, [], quick=True))
    assert app.calib_job.wait(60.0)
    # Dom po udanej fali odbiera ramie - ale nie moze oznaczyc fali jako przerwanej.
    assert app.calib_job.state == jobs.DONE and app.calib_job.result == "wynik"
    assert app.twin.owner is None and app.calib_job.data["tag_size"] == pytest.approx(app.ws.card_obj().tag_size)


def test_intrinsics_abort_keeps_K_and_the_result_goes_to_the_measured_camera(app, monkeypatch):
    import threading

    from lerobot_mp.twin.calib import intrinsics
    from lerobot_mp.twin.workspace import CameraRecord

    class Hub:
        def frame(self, name):
            return None

    board = intrinsics.Board(square=0.028, marker=0.021)
    solved = []
    monkeypatch.setattr(intrinsics.Collector, "solve", lambda self: solved.append(1) or "K")
    job = jobs.Job("intrynsyki")
    job.start(lambda j: jobs.run_intrinsics(j, Hub(), "kam1", board, (640, 480), period=0.01))
    time.sleep(0.05)
    job.stop()                                                  # "Przerwij"
    assert job.wait(2.0) and job.state == jobs.CANCELLED and job.result is None and solved == []
    solve = threading.Event()
    job.start(lambda j: jobs.run_intrinsics(j, Hub(), "kam1", board, (640, 480), period=0.01, solve=solve))
    solve.set()                                                 # "Oblicz i zapisz K"
    assert job.wait(2.0) and job.state == jobs.DONE and job.result == "K" and job.data["camera"] == "kam1"

    class Res:
        K = np.array([[610.0, 0, 320], [0, 611.0, 240], [0, 0, 1]])
        dist = np.zeros(5)
        rms_px, n_views, coverage, trusted, reason = 0.3, 14, 0.7, True, ""

    for n in ("kam1", "kam2"):
        app.ws.add_camera(CameraRecord(n, "sim"))
    try:
        app._tick_slow()
        app.intr_cam.value = "kam2"                             # lista w panelu zmieniona w trakcie
        ij = app.intr_job
        ij.state, ij.result, ij.data = jobs.DONE, Res(), {"camera": "kam1"}
        app._tick_calibration()
        assert app.ws.camera("kam1").K[0][0] == pytest.approx(610.0)
        assert app.ws.camera("kam2").K is None
    finally:
        for n in ("kam1", "kam2"):
            app.ws.remove_camera(n)
        app.intr_job.state = jobs.IDLE


def test_stale_or_repeated_cube_detection_does_not_keep_the_policy_going(app):
    from lerobot_mp.twin.perception import CubeDetection, CubeTracker

    def det(age: float) -> CubeDetection:
        return CubeDetection(np.array([0.25, 0.0, 0.015]), np.eye(3), 100.0, 0.9, n_cameras=2,
                             t=time.monotonic() - age)

    app.cube_tracker, app._cube_used_t = CubeTracker(hold_s=0.2), 0.0
    app.last_cube = det(0.0)
    assert app._vision_cube() is not None
    time.sleep(0.3)
    # Ta sama detekcja podawana co takt (mapa wylaczona, kamera zamrozona) odnawiala hold_s bez konca.
    assert app._vision_cube() is None and app.cube_tracker.source == "brak"
    app.last_cube = det(2.0)                                    # detekcja z kadrow sprzed 2 s
    assert app._vision_cube() is None
    # Mapa wylaczona: zadnej starej detekcji w `last_cube`.
    app.last_cube = det(0.0)
    app.map_on.value = False
    try:
        app._tick_map({})
        assert app.last_cube is None
    finally:
        app.map_on.value = True


@pytest.mark.render
def test_saving_a_card_fit_refuses_a_tag_size_changed_since_the_wave(app):
    from lerobot_mp.twin.calib.handeye import CameraFit, Fit
    from lerobot_mp.twin.workspace import CameraRecord

    app.ws.add_camera(CameraRecord("front", "sim"))
    try:
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.3]
        app.calib_job.result = Fit({"front": CameraFit(T, 0.2, 0.5, 30, 20.0, trusted=True)})
        app.calib_job.data = {"tag_size": 0.050}
        app.tag_mm.value = 49.2
        with pytest.raises(RuntimeError, match="bok taga"):
            app._apply_calibration()
        app.tag_mm.value = 50.0
        assert app._apply_calibration() == ["front"]
        assert app.ws.camera("front").calibration["tag_size"] == pytest.approx(0.050)
    finally:
        app.ws.remove_camera("front")
        app.calib_job.result, app.calib_job.data = None, {}


def _click(app, label: str) -> _Event:
    """Przycisk panelu bez przegladarki - te same callbacki, co klikniecie."""
    hs = [h for h in app.server.gui._gui_input_handle_from_uuid.values() if getattr(h, "label", None) == label]
    assert len(hs) == 1, label
    ev = _Event()
    for cb in hs[0]._impl.update_cb:
        cb(ev)
    return ev


@pytest.mark.parametrize("action", ["stop", "home"])
def test_stop_home_or_halt_while_the_policy_loads_never_lets_it_drive(app, monkeypatch, action):
    """STOP/Dom 0,1 s po Uruchom trafialy w okno ladowania polityki (108-142 ms): STOP szedl do
    starego runnera, nowa polityka brala ramie pod STOP-em, a po Skasuj STOP jechala - 83,7 st.
    ruchu stawu (verify2, s15d); Dom byl przerywany przez `claim` polityki (s15c)."""
    from lerobot_mp.twin.rl import policy as P

    app.twin.connect("sim", threaded=False)
    app._refresh_policies()
    app.pol_pick.value = next(o for o in app.pol_pick.options if "reach" in o)
    real = P.Policy.load
    press = {"stop": lambda: app._emergency_stop(None), "home": lambda: app._go_home()}[action]

    def load(path, device="cpu"):
        pol = real(path, device)
        press()                                             # operator klika w trakcie ladowania
        return pol
    monkeypatch.setattr(P.Policy, "load", staticmethod(load))
    with pytest.raises(RuntimeError, match="aktywny STOP|w trakcie startu"):
        app._start_policy(_Event())
    assert app.twin.owner is None and not app.runner.status.running and not app.twin._engaged
    assert "nie ruszyla" in app.runner.status.stopped_because
    if action == "stop":
        app.twin.clear_estop()                              # Skasuj STOP
    start = dict(app.twin.joints())
    for _ in range(100):
        app.twin.step(0.02)
    assert app.twin.owner is None and not app.runner.status.running
    if action == "home":                                    # Dom jedzie swoja rampa - polityka jej nie przerwala
        assert app.twin.safety_state.value in ("HOMING", "IDLE")
    else:
        assert max(abs(app.twin.joints()[k] - start[k]) for k in start) < 1.0
    # To samo dla fali i identyfikacji (`_take_arm` z chwila startu).
    mark = app._motion_mark()
    press()
    if action == "stop":
        app.twin.clear_estop()
    with pytest.raises(RuntimeError, match="w trakcie startu"):
        app._take_arm(jobs.SYSID_OWNER, mark=mark)
    assert app.twin.owner is None


def test_a1x_enable_button_needs_a_connected_a1x_a_confirmation_and_a_disarmed_panel(app, monkeypatch):
    """"Wlacz silniki A1X" tylko z polaczonym a1x, z potwierdzeniem i przy rozbrojonym panelu."""
    from lerobot_mp.twin.runtime import RobotStatus

    app.twin.connect("sim", threaded=False)
    ev = _click(app, "Wlacz silniki A1X (FF 1-5-6)")
    assert ev.client.notes and "tylko dla polaczonego ramienia a1x" in ev.client.notes[-1][1]
    app.twin.disconnect()

    class FakeA1X:
        calls = 0

        def enable(self):
            FakeA1X.calls += 1
            return "wyslano FF 1 -> 5 -> 6"

        def disconnect(self):
            pass

    monkeypatch.setattr(app.twin, "_backend", FakeA1X())
    monkeypatch.setattr(app.twin, "status", RobotStatus(connected=True, backend="a1x", simulated=False))
    ev = _click(app, "Wlacz silniki A1X (FF 1-5-6)")
    assert "zaznacz potwierdzenie" in ev.client.notes[-1][1] and FakeA1X.calls == 0
    app.a1x_confirm.value = True
    app.arm_engage.value = True
    ev = _click(app, "Wlacz silniki A1X (FF 1-5-6)")
    assert "rozbroj" in ev.client.notes[-1][1] and FakeA1X.calls == 0
    app.arm_engage.value = False
    _click(app, "Wlacz silniki A1X (FF 1-5-6)")
    deadline = time.monotonic() + 2.0
    while "wyslano" not in app.a1x_md.content and time.monotonic() < deadline:
        time.sleep(0.01)
    assert FakeA1X.calls == 1 and "wyslano FF 1 -> 5 -> 6" in app.a1x_md.content
    assert app.a1x_confirm.value is False                 # nastepne wlaczenie - nowe potwierdzenie
    app.twin._backend = None
