"""Panel blizniaka: startuje, odswieza sie i obsluguje kamery symulowane bez przegladarki.

Dalej - to, czego panel pilnuje przed prawdziwym ramieniem: jeden wlasciciel ramienia,
uchwyt TCP na tej samej galezi IK, kalibracja i zadania w tle.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("mujoco")

from conftest import BOT  # noqa: E402

from machinekind_twin.ui import jobs  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def a_panel(path):
    """Panel na nowym stanowisku z robotem testowym (`path` - plik stanowiska)."""
    from machinekind_twin.ui.app import TwinApp

    path.write_text(json.dumps({"robot": BOT}), encoding="utf-8")
    return TwinApp(path, host="127.0.0.1", port=free_port())


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
    from machinekind_twin.kinematics import pose
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


class _SceneRecorder:
    """A viser scene API that records what the mirror adds: (method, name)."""

    def __init__(self):
        self.added: list[tuple[str, str]] = []

    def __getattr__(self, method):
        def add(name, *args, **kw):
            self.added.append((method, name))
            return SimpleNamespace(remove=lambda: None)
        return add


def test_the_3d_view_draws_every_geom_of_an_arm_built_from_primitives():
    """Cylinders and capsules never reached the 3D view: the geom type is a numpy.int32, and
    `numpy.int32 in (mjtGeom, ...)` is False. The test robot showed only its boxes."""
    import mujoco

    from machinekind_twin.robots import get_spec
    from machinekind_twin.ui.bridge import SceneMirror

    model = mujoco.MjModel.from_xml_path(str(get_spec(BOT).mjcf_path))
    scene = _SceneRecorder()
    SceneMirror(SimpleNamespace(scene=scene), model)

    shapes = sorted(method for method, _ in scene.added if method != "add_frame")
    assert shapes == ["add_box"] * 3 + ["add_cylinder"] * 3


@pytest.mark.render
def test_panel_starts_adds_a_simulated_camera_and_maps_the_table(tmp_path):
    app = a_panel(tmp_path / "twin.json")
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
    app = a_panel(tmp_path_factory.mktemp("panel") / "twin.json")
    yield app
    app.close()


@pytest.fixture
def app(shared_app):
    a = shared_app
    yield a
    a.perception.stop_all("koniec testu")
    a.strategies.stop("koniec testu")
    for job in (a.calib_job, a.sysid_job, a.intr_job):
        job.stop()
        job.wait(10.0)
        job.state, job.result, job.data = jobs.IDLE, None, {}
    a.twin.disconnect()
    a.arm_engage.value = False
    a.tcp_gizmo_on.value = False
    a.strategies.goal_on.value = False
    for owner in (a.twin.owner,):
        if owner is not None:
            a.twin.release(owner)


def test_tcp_gizmo_rejects_a_jump_to_the_other_ik_branch(app, monkeypatch):
    """Przy podstawie obroconej daleko przeciagniecie o 1 cm dawalo "poprawne" IK 180 st. dalej.

    Robot testowy ma jedna galaz IK - druga galaz (podstawa 180 st. dalej) podstawia tu `ik`.
    """
    spec = app.ws.spec()
    start = dict(spec.home, j1=30.0)
    kin = app._kin_gizmo
    p = kin.tcp(start)[:3, 3]
    tang = np.array([-p[1], p[0], 0.0]) / np.hypot(p[0], p[1])
    ik = kin.ik

    def other_branch(*a, **kw):
        sol = ik(*a, **kw)
        sol.joints = dict(sol.joints, j1=sol.joints["j1"] - 180.0)
        return sol

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

    monkeypatch.setattr(kin, "ik", other_branch)
    assert drag(p - 0.01 * tang) is None                        # odrzucone, cel bez zmian
    assert "inna galaz" in app.tcp_md.content
    monkeypatch.setattr(kin, "ik", ik)
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
            arm.move(dict(home, j1=20.0 if k % 2 else -20.0), 0.3)
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
    import machinekind_twin.calib.session as session_mod

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

    from machinekind_twin.calib import intrinsics
    from machinekind_twin.workspace import CameraRecord

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


@pytest.mark.render
def test_saving_a_card_fit_refuses_a_tag_size_changed_since_the_wave(app):
    from machinekind_twin.calib.handeye import CameraFit, Fit
    from machinekind_twin.workspace import CameraRecord

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
def test_stop_or_home_during_a_start_never_lets_the_task_take_the_arm(app, action):
    """STOP/Dom w oknie startu zadania (fala, identyfikacja): `_take_arm` z chwila startu odmawia,
    takze po Skasuj STOP (verify2, s15c/s15d)."""
    app.twin.connect("sim", threaded=False)
    press = {"stop": lambda: app._emergency_stop(None), "home": lambda: app._go_home()}[action]
    mark = app._motion_mark()
    press()
    if action == "stop":
        app.twin.clear_estop()
    with pytest.raises(RuntimeError, match="w trakcie startu"):
        app._take_arm(jobs.SYSID_OWNER, mark=mark)
    assert app.twin.owner is None


def test_a1x_enable_button_needs_a_connected_a1x_a_confirmation_and_a_disarmed_panel(app, monkeypatch):
    """"Wlacz silniki A1X" tylko z polaczonym a1x, z potwierdzeniem i przy rozbrojonym panelu."""
    from machinekind_twin.runtime import RobotStatus

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
    from machinekind_twin.runtime import RobotStatus

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
    monkeypatch.setattr(app.twin, "_backend", object())   # backend bez tej metody (sim, testbot): steruje
    assert "XCAN" not in app._status_text()
    app.twin._backend = None



def test_status_line_mentions_the_gripper_over_xcan(app, monkeypatch):
    """XCAN z proba chwytaka (a1x.gripper_over_xcan / A1X_GRIPPER_XCAN=1): zamiast odmowy "proba"."""
    from machinekind_twin.runtime import RobotStatus

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
    from machinekind_twin.workspace import CameraRecord
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
    from machinekind_twin.runtime import RobotStatus

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
    monkeypatch.setattr(app.twin, "status", RobotStatus(connected=True, backend="testbot", simulated=False))
    _click(app, "Start: ramie pokaze karte kamerom (kam)")
    assert started == [] and app.wave_btn.color == "orange" and "Kliknij jeszcze raz" in app.wave_btn.label
    _click(app, app.wave_btn.label)
    assert started == [(["kam"], False)]
    app.calib_ask.reset()


def test_a_moved_camera_brings_up_its_relocalize_button(app, monkeypatch):
    from machinekind_twin.workspace import CameraRecord

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
    from machinekind_twin.workspace import CameraRecord

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
    from machinekind_twin.calib import intrinsics as intr
    from machinekind_twin.ui.app import LENS_DIR

    res = intr.read_file(LENS_DIR / filename)
    assert res.K.shape == (3, 3) and res.K[0, 0] > 500 and res.dist.shape == (5,)
    assert res.trusted and res.reason == f"z pliku {filename}"


def test_lens_from_file_lands_on_the_chosen_camera_and_refuses_another_resolution(app, monkeypatch, tmp_path):
    """Krok 1 bez szachownicy: obiektyw z pliku .txt (K, dystorsja) wpisuje sie do wybranej kamery,
    plik z innej rozdzielczosci jest odrzucany, a poza ustalona ze starym K przestaje byc zaufana."""
    from machinekind_twin.ui import app as app_mod
    from machinekind_twin.workspace import CameraRecord

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
    from machinekind_twin.ui import app as app_mod
    from machinekind_twin.workspace import CameraRecord

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


class _Port:
    def __init__(self, device: str, vid: int | None):
        self.device, self.vid = device, vid


def plugged(monkeypatch, *devices: str) -> None:
    """These USB-serial devices are plugged in (a serial port without a USB id is not one)."""
    lp = pytest.importorskip("serial.tools.list_ports")
    ports = [_Port("/dev/ttyS0", None)] + [_Port(d, 0x1A86) for d in devices]
    monkeypatch.setattr(lp, "comports", lambda: ports)


def test_the_port_is_the_saved_one_while_it_is_plugged_in(monkeypatch):
    from machinekind_twin.ui.app import find_port

    plugged(monkeypatch, "/dev/ttyACM0", "/dev/ttyACM1")
    assert find_port("/dev/ttyACM1") == "/dev/ttyACM1"


def test_the_port_is_the_one_device_plugged_in_when_the_saved_one_is_gone(monkeypatch):
    """A swapped arm has a new port name: the saved one must not win over the device that is there."""
    from machinekind_twin.ui.app import find_port

    plugged(monkeypatch, "/dev/ttyACM0")
    assert find_port("/dev/ttyUSB3") == "/dev/ttyACM0"
    assert find_port(None) == "/dev/ttyACM0"


def test_the_port_is_blank_when_several_devices_are_plugged_in_and_none_is_the_saved_one(monkeypatch):
    from machinekind_twin.ui.app import find_port

    plugged(monkeypatch, "/dev/ttyACM0", "/dev/ttyACM1")
    assert find_port("/dev/ttyUSB3") == ""
    plugged(monkeypatch)
    assert find_port("/dev/ttyUSB3") == ""


def test_a_bridge_address_is_kept_as_the_port(monkeypatch):
    from machinekind_twin.ui.app import find_port

    plugged(monkeypatch, "/dev/ttyACM0")
    assert find_port("socket://10.0.0.5:5555") == "socket://10.0.0.5:5555"


def test_a_preset_without_a_port_fills_the_field_with_the_port_plugged_in(app, monkeypatch):
    sim, typed = "symulacja (blizniak jest ramieniem)", "Robot testowy (w pamieci)"
    before = app.arm_backend.value
    plugged(monkeypatch, "/dev/ttyACM0")
    try:
        monkeypatch.setattr(app.ws, "port", "/dev/ttyUSB3")
        app.arm_backend.value = sim
        app.arm_backend.value = typed
        assert app.arm_port.value == "/dev/ttyACM0"
    finally:
        app.arm_backend.value = before


def test_connect_takes_the_port_plugged_in_over_a_stale_one_in_the_field(app, monkeypatch):
    """The arm was swapped while the panel ran: Polacz goes to the device that is there."""
    typed = "Robot testowy (w pamieci)"
    before = app.arm_backend.value
    seen = {}

    def connect(backend, port=None, **kw):
        seen["port"] = port

    monkeypatch.setattr(app.twin, "connect", connect)
    plugged(monkeypatch, "/dev/ttyACM7")
    try:
        app.arm_backend.value = typed
        app.arm_port.value = "/dev/ttyUSB3"
        app._connect(app._preset_backend(), app._port_from_field(), False)
        assert seen["port"] == "/dev/ttyACM7"
        assert app.arm_port.value == "/dev/ttyACM7"
    finally:
        app.arm_backend.value = before


def test_startup_flag_connects_the_arm(app):
    """`twin ui --arm`: ramie polaczone przy starcie - bez klikania."""
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
    """`twin --config` with this file."""
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(path))


def test_the_strategies_tab_draws_an_installed_strategys_settings(app, swing):
    tab = choose_strategy(app, swing)

    assert swing in tab.pick.options
    assert tab.plugin.name == "swing"
    assert "machinekind-twin-swing 0.1.0" in tab.about_md.content
    assert list(tab._inputs) == ["joint", "amplitude", "period_s", "cycles", "wait_for_return"]
    assert type(tab._inputs["joint"]).__name__ == "GuiTextHandle"
    assert type(tab._inputs["amplitude"]).__name__ == "GuiNumberHandle"
    assert type(tab._inputs["cycles"]).__name__ == "GuiNumberHandle"
    assert type(tab._inputs["wait_for_return"]).__name__ == "GuiCheckboxHandle"
    assert tab._inputs["joint"].value == ""
    assert tab._inputs["amplitude"].value == 10.0
    assert tab._inputs["wait_for_return"].value is True


def test_the_strategys_config_section_prefills_its_form(app, swing, tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "swing:\n  joint: j2\n  cycles: 3\n")

    tab = choose_strategy(app, swing)

    assert tab.section_problem == ""
    assert tab._inputs["joint"].value == "j2"
    assert tab._inputs["cycles"].value == 3


def test_a_strategy_from_a_package_runs_on_the_sim_arm_through_the_strategy_runner(app, swing):
    from machinekind_twin.strategy import StrategyRunner

    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="j1", period_s=1.0)

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
    a_config_file(tmp_path, monkeypatch, "swing:\n  joint: j1\n  amplitud: 5\n")
    app.twin.connect("sim", threaded=False)

    tab = choose_strategy(app, swing, joint="j1")

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
    tab = choose_strategy(app, swing, joint="j1")

    with pytest.raises(RuntimeError, match="zaznacz potwierdzenie"):
        tab.start()
    assert app.twin.owner is None

    tab.confirm.value = True
    tab.start()

    assert app.twin.owner == "strategia"
    assert tab.confirm.value is False


def test_stop_ends_a_running_strategy(app, swing):
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="j1")
    tab.start()

    app._emergency_stop(None)

    assert not tab.running
    assert "STOP" in tab.runner.status.stopped_because
    assert app.twin.owner is None
    app.twin.clear_estop()


COUNTER = '''
from dataclasses import dataclass

@dataclass
class Settings:
    count: int
    speed: float = 1

class Strategy:
    hz = 10.0
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def act(self, obs):
        return {}
'''


def install_counter(install_dist) -> str:
    install_dist("counter-strategy",
                 {"machinekind_twin.strategies": {"counter": "counter_strategy:Strategy"}},
                 {"counter_strategy": COUNTER})
    return "counter"


def test_a_required_number_nobody_set_is_refused_before_start(app, install_dist):
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, install_counter(install_dist))

    with pytest.raises(ValueError, match=r"Brak kluczy w sekcji counter: \['count'\]"):
        tab.start()
    assert app.twin.owner is None

    tab._inputs["count"].value = 3
    tab.start()

    assert tab.runner.strategy.config.count == 3
    assert app.twin.owner == "strategia"


def test_a_float_setting_with_an_int_default_keeps_its_decimals(app, install_dist):
    tab = choose_strategy(app, install_counter(install_dist), count=1, speed=1.5)

    assert tab._inputs["speed"].value == 1.5


GOALS = '''
class Strategy:
    hz = 50.0
    goals = []

    def act(self, obs):
        Strategy.goals.append(obs.goal)
        return {}
'''


def next_goal(tab):
    """The goal the running strategy sees two ticks from now: the tick under way may predate a change."""
    goals = tab.runner.strategy.goals
    seen = len(goals)
    deadline = time.monotonic() + 5.0
    while len(goals) < seen + 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    return goals[-1]


def test_the_3d_handle_starts_on_the_tcp(app):
    tab = app.strategies

    tab.goal_on.value = True

    assert tab.goal_handle.visible
    assert np.allclose(tab.goal_handle.position, app.twin.scene.data.site_xpos[app.twin.scene.kin.site_id])


def test_the_3d_handle_gives_a_running_strategy_its_goal_in_the_base_frame(app, install_dist):
    install_dist("goal-strategy", {"machinekind_twin.strategies": {"goals": "goal_strategy:Strategy"}},
                 {"goal_strategy": GOALS})
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, "goals")
    tab.start()

    assert next_goal(tab) is None

    tab.goal_on.value = True
    tab.goal_handle.position = (app.T_b2w @ [0.20, 0.05, 0.10, 1.0])[:3]

    assert np.allclose(next_goal(tab), [0.20, 0.05, 0.10])

    tab.goal_on.value = False

    assert next_goal(tab) is None


def test_a_second_start_while_the_first_is_starting_is_refused(app, swing, monkeypatch):
    """viser runs each click on its own thread: two runners would drive the arm under one owner."""
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="j1")

    inside = threading.Event()
    let_go = threading.Event()
    create = tab._create

    def slow_create():
        inside.set()
        let_go.wait(5.0)
        return create()

    monkeypatch.setattr(tab, "_create", slow_create)
    first = threading.Thread(target=tab.start)
    first.start()
    assert inside.wait(5.0)

    with pytest.raises(RuntimeError, match="start strategii juz trwa"):
        tab.start()

    let_go.set()
    first.join(5.0)
    assert tab.running


def test_no_strategy_package_is_imported_until_one_is_chosen(app, swing):
    tab = app.strategies
    tab.pick.value = "-"
    tab.refresh()

    assert tab.pick.options[0] == "-"
    assert swing in tab.pick.options
    assert tab.plugin is None
    assert tab._inputs == {}
    assert "Wybierz strategie" in tab.about_md.content


def test_the_strategies_tab_gives_its_runner_the_objects_the_detectors_see(app, swing):
    app.twin.connect("sim", threaded=False)
    tab = choose_strategy(app, swing, joint="j1")

    tab.start()

    assert tab.runner._objects == app.perception.objects


# ------------------------------------------------------------------ detectors from installed packages
def choose_detector(app, name: str, **values):
    """Detektory: Odswiez liste, choose `name`, type `values` into its form."""
    tab = app.detectors
    tab.refresh()
    tab.pick.value = name
    for key, value in values.items():
        tab.form.inputs[key].value = value
    return tab


def start_detector(tab) -> None:
    """Uruchom detektor, and wait for its start thread."""
    tab.start()
    tab.start_thread.join(10.0)
    assert not tab.start_thread.is_alive()


def test_the_detectors_tab_draws_an_installed_detectors_settings(app, cube_example):
    tab = choose_detector(app, cube_example)

    assert cube_example in tab.pick.options
    assert tab.plugin.name == "cube"
    assert "machinekind-twin-cube 0.1.0" in tab.about_md.content
    inputs = tab.form.inputs
    assert list(inputs) == ["color", "size_m", "label", "min_area_px", "interval_s"]
    assert type(inputs["color"]).__name__ == "GuiTextHandle"
    assert type(inputs["label"]).__name__ == "GuiTextHandle"
    assert type(inputs["size_m"]).__name__ == "GuiNumberHandle"
    assert type(inputs["interval_s"]).__name__ == "GuiNumberHandle"
    assert inputs["color"].value == "red"
    assert inputs["size_m"].value == 0.03


def test_the_detectors_config_section_prefills_its_form(app, cube_example, tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "cube:\n  color: green\n")

    tab = choose_detector(app, cube_example)

    assert tab.section_problem == ""
    assert tab.form.inputs["color"].value == "green"


def test_a_wrong_key_in_the_detectors_config_section_blocks_its_start(app, cube_example, tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "cube:\n  colour: green\n")

    tab = choose_detector(app, cube_example)

    assert "colour" in tab.section_problem
    assert "Uruchom odmowi" in tab.about_md.content
    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji cube: \['colour'\]"):
        tab.start()
    assert app.perception.running() == {}


def test_start_runs_the_detector_and_stop_removes_it_and_its_objects(app, cube_example):
    tab = choose_detector(app, cube_example, color="green")

    start_detector(tab)

    runner = app.perception.running()["cube"]
    assert runner.status.running
    assert runner.detector.config.color == "green"
    assert "detektory: cube" in app._status_text()
    tab.tick()
    assert "**dziala**" in tab._rows["cube"].md.content
    assert tab._rows["cube"].stop.label == "Zatrzymaj cube"
    assert not tab.none_md.visible

    tab.stop("cube")

    assert app.perception.running() == {}
    assert app.perception.objects() == {}
    assert not runner.status.running
    tab.tick()
    assert tab._rows["cube"].md.content == "zatrzymany: zatrzymany z panelu"
    tab.refresh()
    assert tab._rows == {}
    assert tab.none_md.visible


def test_a_second_start_of_a_running_detector_replaces_it(app, cube_example):
    tab = choose_detector(app, cube_example)
    start_detector(tab)
    first = app.perception.running()["cube"]

    tab.form.inputs["color"].value = "blue"
    start_detector(tab)

    second = app.perception.running()["cube"]
    assert second is not first and second.status.running
    assert second.detector.config.color == "blue"
    assert not first.status.running
    assert first.status.stopped_because == "ponownie z nowymi ustawieniami"


def test_a_second_start_while_the_first_is_starting_is_refused_and_the_panel_does_not_wait(
        app, cube_example, monkeypatch):
    tab = choose_detector(app, cube_example)
    inside = threading.Event()
    let_go = threading.Event()
    start = app.perception.start_detector

    def slow_start(plugin, raw):
        inside.set()
        let_go.wait(5.0)
        return start(plugin, raw)

    monkeypatch.setattr(app.perception, "start_detector", slow_start)
    tab.start()                                                   # returns while the detector starts
    assert tab.activity_md.visible
    assert tab.activity_md.content == "⏳ **Startuje...**"
    assert tab._rows["cube"].md.content == "**startuje...**"  # no panel tick needed
    assert inside.wait(5.0)
    tab.tick()
    assert tab._rows["cube"].md.content == "**startuje...**"

    with pytest.raises(RuntimeError, match="start detektora juz trwa"):
        tab.start()

    let_go.set()
    tab.start_thread.join(5.0)
    assert "cube" in app.perception.running()


@pytest.mark.parametrize("fails", [False, True])
def test_a_pending_detector_call_shows_elapsed_time_until_it_finishes(app, cube_example, monkeypatch, fails):
    tab = choose_detector(app, cube_example, interval_s=60.0)
    entered, release = threading.Event(), threading.Event()
    now = [0.0]
    monkeypatch.setattr(app.perception, "clock", lambda: now[0])

    def slow_detect(self, request):
        entered.set()
        assert release.wait(5.0)
        if fails:
            raise RuntimeError("test request failed")
        return []

    monkeypatch.setattr(tab.plugin.cls, "detect", slow_detect)
    try:
        start_detector(tab)
        assert entered.wait(5.0)
        runner = app.perception.running()["cube"]
        runner._wait = lambda _: True  # end the thread after this call, without stopping its status
        now[0] = 7.0
        tab.tick()

        assert tab.activity_md.visible
        assert "Wywolanie w toku" in tab.activity_md.content and "7.0 s" in tab.activity_md.content
        line = tab._rows["cube"].md.content
        assert "wywolanie 1 w toku" in line and "7.0 s" in line
        assert "oczekiwanie na pierwszy wynik" in line and "wynik: nic" not in line
        assert runner.status.calls == 0

        release.set()
        runner.join(5.0)
        assert not runner._thread.is_alive()
        tab.tick()

        assert not tab.activity_md.visible
        line = tab._rows["cube"].md.content
        assert "w toku" not in line and "ukonczonych wywolan 1" in line
        if fails:
            assert "test request failed" in line and "kolejna proba za" in line
        else:
            assert "ostatni wynik: nic" in line
    finally:
        release.set()


def test_a_retry_in_progress_shows_the_previous_error_without_a_retry_countdown():
    from machinekind_twin.detection import DetectorStatus
    from machinekind_twin.ui.detector_tab import _running_line

    status = DetectorStatus(running=True, call_started=0.0, calls=1, failures=1,
                            last_error="RuntimeError: model unavailable")
    line = _running_line(SimpleNamespace(status=status), 7.0)

    assert "wywolanie 2 w toku" in line and "7.0 s" in line
    assert "poprzedni blad" in line and "model unavailable" in line
    assert "kolejna proba za" not in line


def test_a_detector_refusing_its_settings_shows_why_in_its_row(app, cube_example):
    tab = choose_detector(app, cube_example, color="purple")

    start_detector(tab)

    assert app.perception.running() == {}
    tab.tick()
    assert tab._rows["cube"].md.content.startswith("**nie ruszyl:** ValueError: cube: unknown color 'purple'")


def test_the_detectors_tab_lists_the_tracked_objects_and_draws_them_in_3d(app, monkeypatch):
    from machinekind_twin.objects import Located

    loc = Located("cube", np.array([0.231, -0.04, 0.015]), np.eye(3), 0.93, 2, time.monotonic(),
                  (0.03, 0.03, 0.03), "sylwetka")
    app.perception.tracks.observe("cube", [loc], latency_s=0.05, interval_s=0.33)
    tab = app.detectors

    # A step within 10 ms of an earlier one returns the cached result; an earlier test may have
    # just stepped, so lift the rate limit and step once.
    monkeypatch.setattr(app.perception.tracks, "MIN_STEP_S", 0.0)
    app.perception.objects()
    tab.tick()

    line = tab.objects_md.content
    assert line.startswith("- **cube**: x 23.1 cm, y -4.0 cm, z 1.5 cm, obrot 0 st., kamery, 2 kam. (sylwetka), "
                           "pewnosc 0.93, wiek ")
    assert line.endswith(" s [cube]")
    assert tab._markers[0].frame.visible and tab._markers[0].key == ("pudelko", (0.03, 0.03, 0.03))

    tab.show_3d.value = False
    tab.tick()
    assert not tab._markers[0].frame.visible

    tab.show_3d.value = True
    app.perception.stop_all("koniec testu")
    tab.tick()
    assert tab.objects_md.content == "brak obiektow"
    assert not tab._markers[0].frame.visible


def test_the_map_tab_lists_what_the_detectors_track_and_runs_no_detection_of_its_own(app, monkeypatch):
    from machinekind_twin.objects import Located

    for gone in ("cube_det", "cube_on", "cube_color", "cube_node", "_mapper_now"):
        assert not hasattr(app, gone)
    assert app.mapper is app.perception.mapper()
    assert app._objects_text() == "  \nobiekty: zaden detektor nie dziala (zakladka Detektory)"

    monkeypatch.setattr(app.perception, "running", lambda: {"cube": None})
    assert app._objects_text() == "  \nobiekty: nie widac"

    loc = Located("cube", np.array([0.231, -0.04, 0.015]), np.eye(3), 0.93, 2, time.monotonic(),
                  (0.03, 0.03, 0.03), "sylwetka")
    app.perception.tracks.observe("cube", [loc], latency_s=0.05, interval_s=0.33)
    # A step within 10 ms of an earlier one returns the cached result: lift the rate limit.
    monkeypatch.setattr(app.perception.tracks, "MIN_STEP_S", 0.0)
    assert "cube" in app.perception.objects()

    text = app._objects_text()
    assert text.startswith("  \ncube: x 23.1 cm, y -4.0 cm, obrot 0 st., pewnosc 0.93, 2 kam., kamery, kadr sprzed ")
    assert text.endswith(" ms")
    app.perception.stop_all("koniec testu")


def test_closing_the_panel_stops_its_detectors(tmp_path, cube_example):
    app = a_panel(tmp_path / "twin.json")
    try:
        start_detector(choose_detector(app, cube_example))
        runner = app.perception.running()["cube"]
    finally:
        app.close()

    assert app.perception.running() == {}
    assert not runner.status.running
    assert runner.status.stopped_because == "zamkniecie panelu"


def test_the_detector_flag_starts_its_detector_at_panel_start(tmp_path, monkeypatch, cube_example):
    """`twin ui --detector cube`: the detector runs without a click, with its config section's settings."""
    from machinekind_twin.ui.app import TwinApp

    a_config_file(tmp_path, monkeypatch, "cube:\n  color: green\n")
    path = tmp_path / "twin.json"
    path.write_text(json.dumps({"robot": BOT}), encoding="utf-8")
    app = TwinApp(path, host="127.0.0.1", port=free_port(), startup={"detectors": [cube_example]})
    try:
        thread = app.detectors.start_thread
        assert thread is not None and thread.name == "detektor-start:cube"
        thread.join(10.0)
        assert not thread.is_alive()

        runner = app.perception.running()["cube"]
        assert runner.status.running
        assert runner.detector.config.color == "green"
    finally:
        app.close()


def test_a_detector_that_cannot_start_at_panel_start_shows_why_and_the_panel_runs_on(app, cube_example,
                                                                                     tmp_path, monkeypatch):
    a_config_file(tmp_path, monkeypatch, "cube:\n  colour: green\n")
    tab = app.detectors

    app._startup(detectors=["no_such_detector", cube_example])   # never raises

    assert app.perception.running() == {}
    tab.tick()
    assert tab._rows["no_such_detector"].md.content.startswith("**nie ruszyl:** KeyError: \"Nieznany detektor")
    assert tab._rows["cube"].md.content.startswith("**nie ruszyl:** ValueError: Nieznane klucze w sekcji cube")
    tab.refresh()


def test_the_detector_flag_reaches_the_panel_from_twin_ui(tmp_path, monkeypatch, capsys):
    """`twin ui --detector NAME` forwards each name to the panel, which hands them to its startup."""
    from machinekind_twin.cli import main as twin
    from machinekind_twin.ui import app as ui_app

    panel_main = ui_app.main
    started: list[list[str]] = []
    monkeypatch.setattr(ui_app, "main", lambda argv: started.append(argv) or 0)
    station = tmp_path / "twin.json"
    station.write_text(json.dumps({"robot": BOT}), encoding="utf-8")

    assert twin(["ui", "--workspace", str(station), "--detector", "cube", "--detector", "llm"]) == 0
    assert started == [["--host", "127.0.0.1", "--port", "8080", "--workspace", str(station),
                        "--detector", "cube", "--detector", "llm"]]
    assert "Start panelu: ramie nie, detektory cube, llm" in capsys.readouterr().out

    made: list[dict] = []

    class Panel:
        def __init__(self, *args, startup=None, **kw):
            made.append(startup)

        def run(self) -> None:
            pass

    monkeypatch.setattr(ui_app, "TwinApp", Panel)
    assert panel_main(started[0]) == 0
    assert panel_main(["--workspace", str(station)]) == 0
    assert made == [{"arm": False, "detectors": ["cube", "llm"]}, {"arm": False, "detectors": []}]
