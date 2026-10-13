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
    a.strategies.stop("koniec testu")
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
    assert not app.calib_ask.ask(("fala", ("kam",)))           # operator kliknal raz: pytanie czeka
    app._connect("sim", None)
    assert app.calib_job.wait(5.0)
    assert app.calib_job.state == jobs.FAILED
    assert not app.calib_ask.pending()                          # nowe ramie - nowe potwierdzenie
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


def test_status_line_says_the_gripper_is_disabled_when_not_drivable(app, monkeypatch):
    """A1X przez XCAN: chwytak nie dostaje 0x051, a panel bez sprzezenia pokazuje rozkaz - linia stanu to mowi."""
    from lerobot_mp.twin.runtime import RobotStatus

    class FakeA1X:
        drivable = False

        def gripper_drivable(self):
            return FakeA1X.drivable

        def disconnect(self):
            pass

    monkeypatch.setattr(app.twin, "_backend", FakeA1X())
    monkeypatch.setattr(app.twin, "status", RobotStatus(connected=True, backend="a1x", simulated=False))
    assert "chwytak wylaczony" in app._status_text()
    FakeA1X.drivable = True
    assert "XCAN" not in app._status_text()
    monkeypatch.setattr(app.twin, "_backend", object())   # backend bez tej metody (sim, SO-101): steruje
    assert "XCAN" not in app._status_text()
    app.twin._backend = None



def test_status_line_mentions_the_gripper_over_xcan(app, monkeypatch):
    """XCAN z proba chwytaka (a1x.gripper_over_xcan / A1X_GRIPPER_XCAN=1): zamiast odmowy "proba"."""
    from lerobot_mp.twin.runtime import RobotStatus

    class FakeA1X:
        trial = True

        def gripper_drivable(self):
            return True

        def gripper_xcan_trial(self):
            return FakeA1X.trial

        def disconnect(self):
            pass

    monkeypatch.setattr(app.twin, "_backend", FakeA1X())
    monkeypatch.setattr(app.twin, "status", RobotStatus(connected=True, backend="a1x", simulated=False))
    text = app._status_text()
    assert "chwytak przez XCAN" in text and "wylaczony" not in text
    FakeA1X.trial = False                                  # can0: ani odmowy, ani proby
    assert "XCAN" not in app._status_text()
    app.twin._backend = None


def test_cameras_tab_with_no_cameras_shows_only_the_add_section(app):
    """Bez kamer: podpowiedz i dodawanie; wybrana kamera i jej przyciski schowane."""
    assert not app.ws.cameras
    app._refresh_cameras()
    assert not app.cam_box.visible and "Brak kamer" in app.cams_html.content
    from lerobot_mp.twin.workspace import CameraRecord
    app.ws.add_camera(CameraRecord("usb1", "5"))
    try:
        app._refresh_cameras(select="usb1")
        assert app.cam_box.visible and "usb1" in app.cams_html.content
        assert not app.cam_sim_box.visible                      # prawdziwej nie przestawia sie z panelu
        _click(app, "Usun kamere")                              # pierwszy klik tylko pyta
        assert app.ws.camera("usb1") and "Na pewno" in app.cam_remove.label
        _click(app, "Na pewno usunac usb1? Kliknij ponownie")
        assert not app.ws.cameras and not app.cam_box.visible
    finally:
        if any(c.name == "usb1" for c in app.ws.cameras):
            app.ws.remove_camera("usb1")
        app._refresh_cameras()


def test_wave_starts_on_one_click_in_sim_and_needs_a_second_click_on_a_real_arm(app, monkeypatch):
    from lerobot_mp.twin.runtime import RobotStatus

    started = []
    monkeypatch.setattr(app, "_wave_cameras", lambda: ["kam"])
    monkeypatch.setattr(app, "_start_card_calibration", lambda ev, cams, quick: started.append((cams, quick)))
    app._render_calibration()
    assert app.wave_btn.disabled and "polacz ramie" in app.wave_md.content.lower()
    app.twin.connect("sim", threaded=False)
    app._render_calibration()
    assert not app.wave_btn.disabled and app.wave_btn.label == "Start: ramie pokaze karte kamerom (kam)"
    _click(app, "Start: ramie pokaze karte kamerom (kam)")
    assert started == [(["kam"], False)]
    started.clear()
    monkeypatch.setattr(app.twin, "status", RobotStatus(connected=True, backend="a1x", simulated=False))
    _click(app, "Start: ramie pokaze karte kamerom (kam)")
    assert started == [] and app.wave_btn.color == "orange" and "Kliknij jeszcze raz" in app.wave_btn.label
    _click(app, app.wave_btn.label)
    assert started == [(["kam"], False)]
    app.calib_ask.reset()


def test_a_moved_camera_brings_up_its_relocalize_button(app, monkeypatch):
    from lerobot_mp.twin.workspace import CameraRecord

    rec = CameraRecord("kam1", "sim")
    rec.T_cam2base = np.eye(4).tolist()
    app.ws.add_camera(rec)
    started = []
    monkeypatch.setattr(app, "_start_card_calibration", lambda ev, cams, quick: started.append((cams, quick)))
    try:
        app.twin.connect("sim", threaded=False)
        app._render_calibration()
        assert not app.reloc_box.visible
        monkeypatch.setattr(app.watch, "moved", lambda name: name == "kam1")
        monkeypatch.setattr(app, "_wave_cameras", lambda: ["kam1"])
        app._render_calibration()
        label = "Kamera kam1 zostala poruszona - pokaz jej karte jeszcze raz (ok. 1 min)"
        assert app.reloc_box.visible and app.reloc_btn.label == label
        assert "PORUSZONA" in app.calib_html.content
        _click(app, label)
        assert started == [(["kam1"], True)]
    finally:
        app.ws.remove_camera("kam1")


def test_after_saving_one_lens_the_calibration_tab_moves_on_to_the_next_camera(app):
    """Po zapisie obiektywu kamery 1 przycisk mowil dalej "kamera 1", choc zostala kamera 2."""
    from lerobot_mp.twin.workspace import CameraRecord

    class Res:
        K = np.array([[610.0, 0, 320], [0, 611.0, 240], [0, 0, 1]])
        dist = np.zeros(5)
        rms_px, n_views, coverage, trusted, reason = 0.3, 14, 0.7, True, ""

    for n, src in (("u1", "5"), ("u2", "6")):
        app.ws.add_camera(CameraRecord(n, src))
    try:
        app._render_calibration()
        assert app.intr_box.visible and app.intr_cam.value == "u1"
        assert app.intr_start.label == "Zacznij: pokazuj szachownice kamerze u1"
        assert "zostalo: u1, u2" in app.intr_title.content
        ij = app.intr_job
        ij.state, ij.result, ij.data = jobs.DONE, Res(), {"camera": "u1"}
        app._tick_calibration()
        assert app.intr_cam.value == "u2" and app.intr_start.label.endswith("kamerze u2")
        assert "u1** zmierzony" in app.intr_md.content and "OBIEKTYW OK" in app.calib_html.content
        assert app.wave_btn.disabled and "krok 1 dla: u2" in app.wave_md.content
    finally:
        for n in ("u1", "u2"):
            app.ws.remove_camera(n)
        app.intr_job.state = jobs.IDLE


def test_fixed_texts_written_at_creation_reach_the_browser(app):
    """Staly tekst z `add_markdown` (opis zakladki, notki Jak uzywac) nie docieral do przegladarki."""
    from viser._gui_handles import GuiMarkdownHandle

    h = app.server.gui.add_markdown("staly tekst")
    try:
        assert h._tw_sent == "staly tekst" and h._impl.props._markdown
        h.content = "staly tekst"                               # ten sam tekst - bez wysylania
        assert GuiMarkdownHandle.content.fset._dedup
    finally:
        h.remove()


@pytest.mark.parametrize("filename", ("logitech_brio_4k_calib.txt",))
def test_shipped_lens_files_parse(filename):
    """Pliki z obiektywami w assets/calib czyta ten sam parser, co przycisk w panelu."""
    from lerobot_mp.twin.calib import intrinsics as intr
    from lerobot_mp.twin.ui.app import LENS_DIR

    res = intr.read_file(LENS_DIR / filename)
    assert res.K.shape == (3, 3) and res.K[0, 0] > 500 and res.dist.shape == (5,)
    assert res.trusted and res.reason == f"z pliku {filename}"


def test_lens_from_file_lands_on_the_chosen_camera_and_refuses_another_resolution(app, monkeypatch, tmp_path):
    """Krok 1 bez szachownicy: obiektyw z pliku .txt (K, dystorsja) wpisuje sie do wybranej kamery,
    plik z innej rozdzielczosci jest odrzucany, a poza ustalona ze starym K przestaje byc zaufana."""
    from lerobot_mp.twin.ui import app as app_mod
    from lerobot_mp.twin.workspace import CameraRecord

    (tmp_path / "brio.txt").write_text(
        "# camera_matrix (3x3, row-major)\n937.5 0 628.3\n0 937.7 348.0\n0 0 1\n"
        "# dist_coeffs (5: k1 k2 p1 p2 k3)\n0.195 -0.519 0.0002 -0.0001 0.361\nnum_views: 81\n")
    (tmp_path / "ueye.txt").write_text(
        "image_size (w h): 1280 1024\nfx: 1512.4\nfy: 1509.8\ncx: 665.1\ncy: 545.7\n"
        "# dist_coeffs (5: k1 k2 p1 p2 k3)\n-0.237 0.177 0.0006 0.0015 0.0\n")
    monkeypatch.setattr(app_mod, "LENS_DIR", tmp_path)
    rec = app.ws.add_camera(CameraRecord("u1", "5", 1280, 720))
    rec.T_cam2base = np.eye(4).tolist()
    rec.calibration = {"trusted": True, "reason": "", "rms_px": 0.2}
    try:
        app._render_calibration()
        assert app.lens_file_box.visible and app.lens_file.options == ("brio.txt", "ueye.txt")
        assert "u1" in app._lens_todo()
        with pytest.raises(RuntimeError, match="1280x1024"):
            app._import_lens("u1", "ueye.txt")
        assert rec.K is None
        assert app._import_lens("u1", "brio.txt") == "u1"
        assert rec.K[0][0] == 937.5 and rec.dist[4] == 0.361 and rec.intrinsics_from == "szachownica"
        assert rec.intrinsics_info["trusted"] and rec.intrinsics_info["n_views"] == 81
        assert not rec.intrinsics_problem() and "u1" not in app._lens_todo()
        assert not rec.calibration["trusted"] and "krok 2" in rec.calibration["reason"]
        assert "u1** wczytany z pliku brio.txt" in app.intr_md.content
        app._render_calibration()
        assert "OBIEKTYW OK" in app.calib_html.content
    finally:
        app.ws.remove_camera("u1")


def test_a_camera_that_failed_to_open_or_whose_stream_died_is_reopened_by_itself(app, monkeypatch):
    """Brio i uEye przegrywaly wyscig z resetem USB przy starcie panelu i zostawaly BEZ OBRAZU,
    az ktos wylaczyl i wlaczyl kamere w panelu. Panel probuje sam, co HEAL_S, poki jest blad."""
    from lerobot_mp.twin.ui import app as app_mod
    from lerobot_mp.twin.workspace import CameraRecord

    calls = []
    hub = app.twin.cameras
    monkeypatch.setattr(hub, "sync", lambda: calls.append(1))
    errors = {"u9": "nie otworzyla sie"}
    monkeypatch.setattr(hub, "error", lambda name: errors.get(name))
    app.ws.add_camera(CameraRecord("u9", "77"))
    try:
        app._heal_cameras(now=1000.0)
        app._heal_cameras(now=1000.0 + app_mod.HEAL_S / 2)
        assert calls == [1]                                    # nie czesciej niz HEAL_S
        app._heal_cameras(now=1000.0 + app_mod.HEAL_S)
        assert calls == [1, 1]
        errors.clear()
        app._heal_cameras(now=1000.0 + 2 * app_mod.HEAL_S)
        assert calls == [1, 1]                                 # bez bledu nic nie robi
    finally:
        app.ws.remove_camera("u9")


def test_home_works_straight_after_a_stop(app):
    """Po STOP-ie ramie bez hamulcow moglo stac w gorze, a Dom byl wylaczony do Skasuj STOP.
    Dom ma byc jednym kliknieciem: sam kasuje STOP i jedzie."""
    app.twin.connect("sim", threaded=False)
    try:
        app.twin.estop("test")
        assert app._estop_active()
        app._render_arm(force=True)
        assert not app.home_btn.disabled
        app._go_home()
        assert not app._estop_active()
    finally:
        app.twin.disconnect()


def test_startup_flag_connects_the_arm(app):
    """`lerobot-twin ui --arm`: ramie polaczone przy starcie - bez klikania."""
    app._startup(arm=True)
    assert app.twin.connected and app.twin.status.simulated


# ------------------------------------------------------------------ strategies from installed packages
def choose_strategy(app, name: str, **values):
    """Strategie: Odswiez liste, choose `name`, type `values` into its form."""
    tab = app.strategies
    tab.refresh()
    tab.pick.value = name
    for key, value in values.items():
        tab._inputs[key].value = value
    return tab


def a_config_file(tmp_path, monkeypatch, text: str) -> None:
    """`lerobot-twin --config` with this file."""
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("LEROBOT_MP_CONFIG", str(path))


def test_the_strategies_tab_draws_an_installed_strategys_settings(app, swing):
    tab = choose_strategy(app, swing)

    assert swing in tab.pick.options
    assert tab.plugin.name == "swing"
    assert "lerobot-mp-swing 0.1.0" in tab.about_md.content
    assert list(tab._inputs) == ["joint", "amplitude", "period_s", "cycles", "wait_for_return"]
    assert type(tab._inputs["joint"]).__name__ == "GuiTextHandle"
    assert type(tab._inputs["amplitude"]).__name__ == "GuiNumberHandle"
    assert type(tab._inputs["cycles"]).__name__ == "GuiNumberHandle"
    assert type(tab._inputs["wait_for_return"]).__name__ == "GuiCheckboxHandle"
    assert tab._inputs["joint"].value == ""
    assert tab._inputs["amplitude"].value == 10.0
    assert tab._inputs["wait_for_return"].value is True


def test_the_strategys_config_section_prefills_its_form(app, swing, tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "swing:\n  joint: elbow_flex\n  cycles: 3\n")

    tab = choose_strategy(app, swing)

    assert tab.section_problem == ""
    assert tab._inputs["joint"].value == "elbow_flex"
    assert tab._inputs["cycles"].value == 3


def test_a_strategy_from_a_package_runs_on_the_sim_arm_through_the_strategy_runner(app, swing):
    from lerobot_mp.twin.strategy import StrategyRunner

    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="shoulder_pan", period_s=1.0)

    tab.start()

    assert type(tab.runner) is StrategyRunner
    assert type(tab.runner.strategy).__module__ == "swing"
    assert tab.runner.strategy.config.period_s == 1.0
    assert app.twin.owner == "strategia"
    assert app._busy("polityka") == "strategia"                  # the Policies tab finds the arm taken

    deadline = time.monotonic() + 5.0
    while tab.runner.status.step < 3 and time.monotonic() < deadline:
        app.twin.step(0.02)
        time.sleep(0.01)

    assert tab.runner.status.step >= 3
    assert "Strategia: jedzie" in app._status_text()

    tab.halt()

    assert not tab.running
    assert tab.runner.status.stopped_because == "zatrzymana z panelu"
    assert app.twin.owner is None
    tab.tick()
    assert "zatrzymana z panelu" in tab.run_md.content


def test_a_wrong_key_in_the_strategys_config_section_is_refused_before_start(app, swing, tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "swing:\n  joint: shoulder_pan\n  amplitud: 5\n")
    app.twin.connect("sim", threaded=False)

    tab = choose_strategy(app, swing, joint="shoulder_pan")

    assert "amplitud" in tab.section_problem
    assert "Uruchom odmowi" in tab.about_md.content

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji swing: \['amplitud'\]"):
        tab.start()

    assert app.twin.owner is None
    assert not tab.running


def test_a_required_setting_left_empty_is_refused_before_start(app, swing):
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing)

    with pytest.raises(ValueError, match=r"Brak kluczy w sekcji swing: \['joint'\]"):
        tab.start()

    assert app.twin.owner is None


def test_a_strategy_needs_a_fresh_confirmation_on_a_real_arm(app, swing, monkeypatch):
    from test_twin_runtime import FakeArm, connect_fake

    connect_fake(app.twin, monkeypatch, FakeArm())
    tab = choose_strategy(app, swing, joint="shoulder_pan")

    with pytest.raises(RuntimeError, match="zaznacz potwierdzenie"):
        tab.start()
    assert app.twin.owner is None

    tab.confirm.value = True
    tab.start()

    assert app.twin.owner == "strategia"
    assert tab.confirm.value is False


def test_stop_ends_a_running_strategy(app, swing):
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="shoulder_pan")
    tab.start()

    app._emergency_stop(None)

    assert not tab.running
    assert "STOP" in tab.runner.status.stopped_because
    assert app.twin.owner is None
    app.twin.clear_estop()

