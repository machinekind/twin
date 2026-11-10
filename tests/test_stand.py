"""Stand folders: a physical stand and its measurement history, saved from and loaded into a workspace.

Also the camera controls a stand carries, set through a fake V4L2 ioctl.
"""

from __future__ import annotations

import errno
import json
import struct

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import BOT

from machinekind_twin import camera_controls as cc
from machinekind_twin import cli
from machinekind_twin import stand as st
from machinekind_twin.calib.handeye import CameraFit, Fit
from machinekind_twin.calib.session import KeepOut, WaveLimits
from machinekind_twin.camera_controls import CameraControls
from machinekind_twin.dynamics import Dynamics
from machinekind_twin.robots import resolve_asset
from machinekind_twin.workspace import CameraRecord, RobotRecord, Workspace, nominal_K


def a_workspace(tmp_path, name: str = "ws") -> Workspace:
    """The example sim workspace plus what a real stand has: a USB camera with a measured lens, a
    pose and fixed focus, servo zero offsets, home, two other robots, dynamics and a real backend."""
    data = json.loads(resolve_asset("examples/twin.sim.json").read_text(encoding="utf-8"))
    data["robot"] = BOT
    path = tmp_path / name / "twin.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    ws = Workspace.load(path)
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.0, 0.4]
    ws.add_camera(CameraRecord(
        "kamera1", "/dev/v4l/by-id/usb-cam-video-index0", 1280, 720, K=nominal_K(1280, 720).tolist(),
        dist=[0.1, -0.05, 0.0, 0.0, 0.0], intrinsics_from="szachownica",
        intrinsics_info={"rms_px": 0.3, "trusted": True, "reason": "", "time": "2026-10-28T13:50:00"},
        T_cam2base=T.tolist(), calibration={"trusted": True, "reason": "", "rms_px": 0.2,
                                            "time": "2026-10-28T14:02:11", "tag_size": 0.05},
        controls=CameraControls(autofocus=False, focus_absolute=30)))
    ws.joint_offsets = {"j2": 1.5}
    ws.home = {"j1": 5.0, "j2": 25.0, "gripper": 0.0}
    ws.other_robots = [RobotRecord("lewe", BOT, xyz=[0.0, 0.45, 0.0], joints={"j1": 10.0}),
                       RobotRecord("prawe", BOT, xyz=[0.0, -0.45, 0.0], yaw_deg=180.0, backend="testbot",
                                   port="/dev/ttyACM1", joints={"j1": 5.0})]
    ws.dynamics = Dynamics(damping=1.2, source="pomiar", fitted=("damping",), band={"damping": 0.1}).to_dict()
    ws.backend, ws.port = "testbot", "/dev/ttyACM0"
    ws.save()
    return ws


def files(folder) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(folder.rglob("*.json")) if "station" not in p.parts}


# ------------------------------------------------------------------ round trip
def test_save_then_load_gives_the_same_workspace(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand", name="biurko", description="testbot przy oknie")
    loaded, stand, backup = st.load_into(tmp_path / "stand", tmp_path / "other" / "twin.json")
    assert backup is None and stand.name == "biurko" and stand.description == "testbot przy oknie"
    want = ws.to_dict()
    want["other_robots"][1]["joints"] = {}       # a live robot's last reading is not part of the stand
    assert loaded.to_dict() == want
    assert Workspace.load(tmp_path / "other" / "twin.json").to_dict() == want
    # Bound: loading and saving again writes nothing new to the stand.
    before = files(tmp_path / "stand")
    loaded.save()
    assert files(tmp_path / "stand") == before
    assert loaded.presets_dir() == (tmp_path / "stand" / "calib" / "poses").resolve()


def test_the_stand_file_holds_the_setup_and_the_current_entries(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    data = json.loads((tmp_path / "stand" / "stand.json").read_text(encoding="utf-8"))
    assert list(data)[:5] == ["format", "name", "description", "obsolete", "robots"]
    assert data["format"] == 1 and data["name"] == "stand" and data["obsolete"] is False
    assert data["robots"][0] == {"robot": BOT, "read_only": False, "backend": "testbot", "port": "/dev/ttyACM0",
                                 "home": {"j1": 5.0, "j2": 25.0, "gripper": 0.0}}
    assert data["robots"][1]["joints"] == {"j1": 10.0} and "joints" not in data["robots"][2]
    cam = data["cameras"][-1]
    assert cam["controls"] == {"autofocus": False, "focus_absolute": 30} and "K" not in cam
    calib = data["calib"]
    assert calib["poses"] == "20261028-140211" and calib["lens"]["kamera1"] == "20261028-135000"
    assert set(calib["lens"]) == {"sym-lewa", "sym-prawa", "kamera1"} and calib["dynamics"]
    lens = json.loads((tmp_path / "stand/calib/lens/kamera1/20261028-135000.json").read_text(encoding="utf-8"))
    assert (lens["width"], lens["height"]) == (1280, 720) and lens["dist"] == [0.1, -0.05, 0.0, 0.0, 0.0]
    pose = json.loads((tmp_path / "stand/calib/poses/20261028-140211.json").read_text(encoding="utf-8"))
    assert pose["joint_offsets"] == {"j2": 1.5} and "kamera1" in pose["cameras"]


def test_the_stand_carries_the_waves_keepout_boxes_and_base_turn(tmp_path):
    ws = a_workspace(tmp_path)
    ws.wave = WaveLimits(keepout=(KeepOut(x=(0.25, 0.35), y=(-0.3, -0.2), z=(0.0, 0.5), name="statyw"),),
                         base_turn=(-90.0, 60.0))
    ws.save()
    folder = tmp_path / "stand"
    st.save(ws, folder)
    data = json.loads((folder / "stand.json").read_text(encoding="utf-8"))
    assert data["wave"] == {"keepout": [{"x": [0.25, 0.35], "y": [-0.3, -0.2], "z": [0.0, 0.5], "name": "statyw"}],
                            "base_turn": [-90.0, 60.0]}
    assert list(data).index("wave") == list(data).index("look") + 1
    station = json.loads((folder / "station" / "station.json").read_text(encoding="utf-8"))
    assert station["wave"] == data["wave"]

    loaded, _, _ = st.load_into(folder, tmp_path / "other" / "twin.json")
    assert loaded.wave == ws.wave
    assert Workspace.load(tmp_path / "other" / "twin.json").wave == ws.wave

    # A stand.json without the section: no limits beyond the arm's own.
    del data["wave"]
    (folder / "stand.json").write_text(json.dumps(data), encoding="utf-8")
    assert st.load(folder).workspace().wave == WaveLimits()


# ------------------------------------------------------------------ history
def test_calibration_history_is_appended_never_overwritten(tmp_path):
    ws = a_workspace(tmp_path)
    folder = tmp_path / "stand"
    st.save(ws, folder)
    first = files(folder / "calib")

    # A new step-2 result and a new lens for kamera1, saved the way the panel saves.
    rec = ws.camera("kamera1")
    T = np.array(rec.T_cam2base)
    T[0, 3] += 0.01
    ws.apply_fit(Fit({"kamera1": CameraFit(T, 0.18, 0.6, 30, 12.5, trusted=True, reason="")}),
                 intrinsics={"kamera1": rec.intrinsics()})
    rec.calibration["time"] = "2026-11-01T10:00:00"
    ws.save()
    rec.K = (np.array(rec.K) * [[1.01], [1.01], [1.0]]).tolist()
    rec.intrinsics_info = {**rec.intrinsics_info, "time": "2026-11-02T09:00:00"}
    ws.save()

    now = files(folder / "calib")
    assert {k: now[k] for k in first} == first               # every earlier entry, byte for byte
    assert "poses/20261101-100000.json" in now and "lens/kamera1/20261102-090000.json" in now
    data = json.loads((folder / "stand.json").read_text(encoding="utf-8"))
    assert data["calib"]["poses"] == "20261101-100000" and data["calib"]["lens"]["kamera1"] == "20261102-090000"
    assert ws.presets() == ["20261101-100000", "20261028-140211"]   # the panel's dropdown: the stand's history

    # Nothing new: nothing written. Going back to an older preset reuses its entry.
    assert st.sync(ws) == []
    ws.load_preset("20261028-140211")
    ws.save()
    assert files(folder / "calib").keys() == now.keys()
    assert json.loads((folder / "stand.json").read_text(encoding="utf-8"))["calib"]["poses"] == "20261028-140211"


def test_a_different_entry_from_the_same_second_gets_a_suffix(tmp_path):
    d = tmp_path / "calib" / "poses"
    assert st.write_entry(d, "20261101-100000", {"time": "a", "cameras": {}}) == ("20261101-100000", True)
    assert st.write_entry(d, "20261101-100000", {"time": "b", "cameras": {}}) == ("20261101-100000", False)
    assert st.write_entry(d, "20261101-100000", {"cameras": {"x": 1}}) == ("20261101-100000-2", True)
    assert json.loads((d / "20261101-100000.json").read_text(encoding="utf-8")) == {"time": "a", "cameras": {}}


def test_save_brings_the_workspace_preset_history_along(tmp_path):
    ws = a_workspace(tmp_path)
    pose = {"T_cam2base": np.eye(4).tolist(), "calibration": {}}
    for stem, cams in (("20260901-120000", {"kamera1": pose}), ("20260902-120000", {"brio": pose})):
        (ws.presets_dir() / f"{stem}.json").write_text(json.dumps({"cameras": cams}), encoding="utf-8")
    _, skipped = st.save(ws, tmp_path / "with")
    assert sorted(p.stem for p in (tmp_path / "with/calib/poses").glob("*.json")) == \
        ["20260901-120000", "20261028-140211"]
    assert skipped == ["20260902-120000"]                # for cameras this stand does not have
    ws2 = a_workspace(tmp_path, "ws2")
    st.save(ws2, tmp_path / "without", history=False)
    assert [p.stem for p in (tmp_path / "without/calib/poses").glob("*.json")] == ["20261028-140211"]


def test_save_refuses_a_stands_repository_and_names_that_are_not_file_names(tmp_path):
    ws = a_workspace(tmp_path)
    repo = tmp_path / "stands"
    st.save(ws, repo / "a")
    with pytest.raises(st.StandError, match="to repozytorium stanowisk"):
        st.save(ws, repo)
    ws.camera("kamera1").name = "lewa/gora"
    with pytest.raises(RuntimeError, match="nazwa 'lewa/gora'"):
        ws.save()
    assert not (repo / "a" / "calib" / "lens" / "lewa").exists()


def test_a_rehearsal_in_sim_keeps_the_stands_backend(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    ws.backend, ws.port = "sim", None                   # what "Polacz" in sim does
    ws.save()
    driven = json.loads((tmp_path / "stand/stand.json").read_text(encoding="utf-8"))["robots"][0]
    assert (driven["backend"], driven["port"]) == ("testbot", "/dev/ttyACM0")


def test_loading_over_another_setup_keeps_a_copy_of_it(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    target = tmp_path / "other" / "twin.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"robot": BOT}), encoding="utf-8")
    _, _, backup = st.load_into(tmp_path / "stand", target)
    assert backup is not None and json.loads(backup.read_text(encoding="utf-8")) == {"robot": BOT}
    _, _, again = st.load_into(tmp_path / "stand", target)
    assert again is None                                # the same setup: nothing to keep


def test_a_workspace_without_a_stand_saves_as_before(tmp_path):
    ws = Workspace.load(resolve_asset("examples/twin.sim.json"))
    ws.save(tmp_path / "twin.json")
    data = json.loads((tmp_path / "twin.json").read_text(encoding="utf-8"))
    assert "stand" not in data and all("controls" not in c for c in data["cameras"])
    assert ws.presets_dir() == tmp_path / "calib"


# ------------------------------------------------------------------ obsolete
def test_an_obsolete_stand_loads_with_a_warning_and_lists_last(tmp_path, capsys):
    ws = a_workspace(tmp_path)
    repo = tmp_path / "stands"
    st.save(ws, repo / "a-stare", name="a-stare")
    st.save(ws, repo / "b-nowe", name="b-nowe")
    (repo / "c-zepsute").mkdir()
    (repo / "c-zepsute" / "stand.json").write_text("{", encoding="utf-8")
    assert cli.main(["stand", "obsolete", str(repo / "a-stare"), "--reason", "kamera2 przestawiona",
                     "--date", "2026-11-03"]) == 0
    capsys.readouterr()

    assert cli.main(["stand", "load", str(repo / "a-stare"), "--path", str(tmp_path / "w" / "twin.json")]) == 0
    err = capsys.readouterr().err
    assert "UWAGA: stanowisko a-stare jest NIEAKTUALNE od 2026-11-03: kamera2 przestawiona" in err

    assert cli.main(["stand", "list", str(repo)]) == 0
    heads = [line for line in capsys.readouterr().out.splitlines() if not line.startswith(" ")]
    assert heads[0].startswith("aktualne") and "b-nowe" in heads[0]
    assert heads[1].startswith("NIEAKTUALNE") and "a-stare" in heads[1]
    assert heads[2].startswith("BLAD") and "c-zepsute" in heads[2]

    st.mark_obsolete(repo / "a-stare", clear=True)
    assert st.load(repo / "a-stare").obsolete is None


# ------------------------------------------------------------------ validation
def _stand_with(tmp_path, change) -> str:
    ws = a_workspace(tmp_path)
    folder = tmp_path / "stand"
    st.save(ws, folder)
    path = folder / "stand.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data, folder)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    with pytest.raises(st.StandError) as exc:
        st.load(folder)
    return str(exc.value)


def _set(key, value):
    def change(data, folder):
        data[key] = value
    return change


@pytest.mark.parametrize("change, message", [
    (lambda d, f: d.pop("format"), "stand.json: brak pola format"),
    (_set("format", 2), "format 2 jest nowszy niz ten blizniak (zna 1)"),
    (_set("colour", "red"), "nieznane pola ['colour']"),
    (_set("name", ""), "stand.json: name: oczekiwano niepustego tekstu"),
    (lambda d, f: d["cameras"][2].update(width=0), "cameras[2] (kamera1).width: oczekiwano liczby calkowitej >= 1"),
    (lambda d, f: d["cameras"][2].update(controls={"autofocus": "off"}), "autofocus: true albo false"),
    (lambda d, f: d["cameras"][2].update(controls={"autofocus": True, "focus_absolute": 30}),
     "focus_absolute dziala tylko z wylaczonym autofokusem"),
    (lambda d, f: d["cameras"][2].update(controls={"zoom": 2}), "controls: nieznane pola ['zoom']"),
    (lambda d, f: d["cameras"].append(dict(d["cameras"][0])), "kamera 'sym-lewa' juz jest"),
    (lambda d, f: d["robots"].append({"robot": BOT}), "dokladnie jedno ramie sterowane (read_only: false), jest 2"),
    (lambda d, f: d["robots"][0].update(xyz=[0, 0, 0]), "jest poczatkiem ukladu stanowiska"),
    (lambda d, f: d["robots"][1].update(name="robot"), "robots[1].name: tylko litery"),
    (lambda d, f: d["robots"][1].update(xyz=[0, 1]), "robots[1].xyz: oczekiwano listy 3 liczb"),
    (lambda d, f: d["robots"][1].update(base_drive="tak"), "robots[1].base_drive: oczekiwano true albo false"),
    (_set("obsolete", {"date": "3.11.2026"}), "obsolete.date: oczekiwano daty RRRR-MM-DD"),
    (lambda d, f: d["table"].update(legs=4), "table: Nieznane klucze"),
    (lambda d, f: d["wave"].update(base_turn=[60.0, -90.0]), "stand.json: wave: wave.base_turn: od < do"),
    (lambda d, f: d["wave"].update(keepout=[{"x": [0.2, 0.1], "y": [0, 1], "z": [0, 1]}]),
     "stand.json: wave: strefa 'strefa': x to [od, do] w metrach, od < do"),
    (lambda d, f: d["calib"]["lens"].update(kamera9="x"), "calib.lens.kamera9: nie ma takiej kamery"),
    (lambda d, f: d["calib"].update(poses="20200101-000000"), "20200101-000000.json: nie ma takiego pliku"),
    (lambda d, f: d["cameras"][2].update(width=1920, height=1080),
     "rozdzielczosc 1920x1080, a obiektyw calib/lens/kamera1/20261028-135000.json jest z 1280x720"),
])
def test_validation_rejects_a_broken_stand_with_a_clear_message(tmp_path, change, message):
    assert message in _stand_with(tmp_path, change)


def test_validation_names_bad_json_and_a_folder_without_a_stand(tmp_path):
    (tmp_path / "s").mkdir()
    with pytest.raises(st.StandError, match="brak stand.json - to nie jest folder stanowiska"):
        st.load(tmp_path / "s")
    (tmp_path / "s" / "stand.json").write_text('{"format": 1,\n "name": }', encoding="utf-8")
    with pytest.raises(st.StandError, match=r"zly JSON \(linia 2"):
        st.load(tmp_path / "s")


def test_a_broken_stand_is_not_overwritten_by_a_save(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    path = tmp_path / "stand" / "stand.json"
    broken = path.read_text(encoding="utf-8").replace('"format": 1', '"format": "1"')
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(RuntimeError, match="format: oczekiwano liczby calkowitej"):
        ws.save()
    assert path.read_text(encoding="utf-8") == broken
    assert json.loads(ws.path.read_text(encoding="utf-8"))["stand"] == str((tmp_path / "stand").resolve())


# ------------------------------------------------------------------ station
def test_export_writes_the_station(tmp_path, capsys):
    import mujoco

    ws = a_workspace(tmp_path)
    folder = tmp_path / "stand"
    st.save(ws, folder)
    (folder / "station" / "stale.txt").write_text("old", encoding="utf-8")
    assert cli.main(["stand", "export", str(folder)]) == 0
    assert "scene.xml, assets/, station.json" in capsys.readouterr().out
    station = folder / "station"
    assert not (station / "stale.txt").exists()
    model = mujoco.MjModel.from_xml_path(str(station / "scene.xml"))
    names = {model.camera(i).name for i in range(model.ncam)}
    assert {"sym-lewa", "sym-prawa", "kamera1"} <= names
    data = json.loads((station / "station.json").read_text(encoding="utf-8"))
    assert "stand" not in data and data == {k: v for k, v in st.load(folder).workspace().to_dict().items()
                                            if k != "stand"}


def test_a_saved_calibration_refreshes_the_station(tmp_path):
    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    ws.card["tag_size"] = 0.0492
    ws.save()
    data = json.loads((tmp_path / "stand/station/station.json").read_text(encoding="utf-8"))
    assert data["card"]["tag_size"] == 0.0492


def test_ui_with_a_stand_loads_it_and_starts_the_panel(tmp_path, monkeypatch, capsys):
    pytest.importorskip("viser")
    from machinekind_twin.ui import app

    ws = a_workspace(tmp_path)
    st.save(ws, tmp_path / "stand")
    seen = []
    monkeypatch.setattr(app, "main", lambda argv: seen.append(argv) or 0)
    target = tmp_path / "panel" / "twin.json"
    assert cli.main(["ui", "--stand", str(tmp_path / "stand"), "--workspace", str(target)]) == 0
    assert seen and seen[0][seen[0].index("--workspace") + 1] == str(target)
    assert Workspace.load(target).stand == str((tmp_path / "stand").resolve())


# ------------------------------------------------------------------ camera controls
class FakeIoctl:
    """A V4L2 camera behind `ioctl`: S_CTRL stores a value (and is recorded), G_CTRL reads it, QUERYCTRL
    describes it; a control in `refuse` does not exist."""

    def __init__(self, refuse: set[int] = frozenset(), reads: dict[int, int] | None = None):
        self.calls: list[tuple[int, int, int, int]] = []
        self.refuse = refuse
        self.values: dict[int, int] = {}
        #: What G_CTRL reports instead of the stored value.
        self.reads = reads or {}

    def __call__(self, fd, request, arg):
        cid = struct.unpack_from("=I", arg)[0]
        if cid in self.refuse:
            raise OSError(errno.EINVAL, "Invalid argument")
        if request == cc.VIDIOC_S_CTRL:
            self.calls.append((fd, request, cid, struct.unpack("=Ii", arg)[1]))
            self.values[cid] = struct.unpack("=Ii", arg)[1]
            return arg
        if request == cc.VIDIOC_G_CTRL:
            return struct.pack("=Ii", cid, self.reads.get(cid, self.values.get(cid, 0)))
        if request == cc.VIDIOC_QUERYCTRL:
            return struct.pack("=II32siiiiI2I", cid, 1, b"focus", 0, 250, 5, 0, 0, 0, 0)
        raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")


def test_camera_controls_are_set_through_v4l2_in_order():
    ioctl, closed = FakeIoctl(), []
    done = cc.apply("/dev/v4l/by-id/usb-cam-video-index0", CameraControls(focus_absolute=30, exposure_absolute=150),
                    platform="linux", ioctl=ioctl, opener=lambda path: 7, closer=closed.append)
    assert ioctl.calls == [(7, cc.VIDIOC_S_CTRL, cc.CID_FOCUS_AUTO, 0), (7, cc.VIDIOC_S_CTRL, cc.CID_FOCUS_ABSOLUTE, 30),
                           (7, cc.VIDIOC_S_CTRL, cc.CID_EXPOSURE_AUTO, cc.EXPOSURE_MANUAL),
                           (7, cc.VIDIOC_S_CTRL, cc.CID_EXPOSURE_ABSOLUTE, 150)]
    assert done.ok == ["autofocus", "focus_absolute", "auto_exposure", "exposure_absolute"] and closed == [7]
    assert cc.VIDIOC_S_CTRL == (3 << 30) | (8 << 16) | (ord("V") << 8) | 28
    opened = []
    cc.apply("2", CameraControls(autofocus=True, auto_exposure=True), platform="linux", ioctl=FakeIoctl(),
             opener=lambda path: opened.append(path) or 3, closer=lambda fd: None)
    assert opened == ["/dev/video2"]


def test_camera_controls_are_skipped_off_linux_and_for_sources_that_are_not_v4l2():
    ioctl = FakeIoctl()
    controls = CameraControls(autofocus=False)
    assert "nie kamera" in cc.apply("/dev/video0", controls, platform="darwin", ioctl=ioctl).skipped
    assert cc.apply("nagranie.mp4", controls, platform="linux", ioctl=ioctl).skipped
    assert cc.apply("/dev/video0", CameraControls(), platform="linux", ioctl=ioctl) == cc.Applied()
    assert ioctl.calls == []


def test_a_control_the_camera_refuses_is_reported_and_the_rest_are_set():
    ioctl = FakeIoctl(refuse={cc.CID_FOCUS_ABSOLUTE})
    done = cc.apply("/dev/video0", CameraControls(focus_absolute=30, auto_exposure=True), platform="linux",
                    ioctl=ioctl, opener=lambda path: 4, closer=lambda fd: None)
    assert done.ok == ["autofocus", "auto_exposure"] and done.failed == {"focus_absolute": "Invalid argument"}
    assert "focus_absolute: Invalid argument" in done.text()
    missing = cc.apply("/dev/video9", CameraControls(autofocus=False), platform="linux", ioctl=ioctl)
    assert missing.failed and not missing.ok             # no such device on this machine: reported, not raised


def test_a_camera_gets_its_controls_when_the_twin_opens_it(monkeypatch):
    from machinekind_twin import camera_stream
    from machinekind_twin.config import CameraConfig

    class FakeCapture:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def get(self, *args):
            return 0.0

        def read(self):
            return False, None

        def release(self):
            pass

    applied = []
    monkeypatch.setattr(camera_stream.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cc, "apply", lambda source, controls, **kw: applied.append((source, controls)) or cc.Applied())
    controls = CameraControls(autofocus=False, focus_absolute=30)
    stream = camera_stream.CameraStream(CameraConfig(source="/dev/video3", controls=controls)).open()
    stream.close()
    assert applied == [("/dev/video3", controls)]


def test_camera_controls_round_trip_through_the_workspace_file(tmp_path):
    ws = Workspace(robot=BOT)
    ws.add_camera(CameraRecord("kamera1", "/dev/video0", controls=CameraControls(autofocus=False, focus_absolute=30)))
    ws.add_camera(CameraRecord("kamera2", "/dev/video2"))
    ws.save(tmp_path / "twin.json")
    cams = json.loads((tmp_path / "twin.json").read_text(encoding="utf-8"))["cameras"]
    assert cams[0]["controls"] == {"autofocus": False, "focus_absolute": 30} and "controls" not in cams[1]
    assert Workspace.load(tmp_path / "twin.json").camera("kamera1").controls == ws.camera("kamera1").controls
    cams[0]["controls"] = {"focus": 30}
    (tmp_path / "twin.json").write_text(json.dumps({"robot": BOT, "cameras": cams}), encoding="utf-8")
    with pytest.raises(ValueError, match="Nieznane klucze"):
        Workspace.load(tmp_path / "twin.json")


class LiveStandIn:
    """An open camera stream: what its open did with the camera's controls."""

    def __init__(self, done: cc.Applied):
        self.controls_applied = done
        self.error, self.is_running = None, True

    def close(self) -> None:
        pass


def a_lens_at_focus(focus: int | None, **kw) -> CameraRecord:
    info = {"rms_px": 0.3, "trusted": True, "reason": ""} | ({} if focus is None else {"focus": focus})
    return CameraRecord("kam", "/dev/video0", K=nominal_K(640, 480).tolist(), intrinsics_from="szachownica",
                        intrinsics_info=info, T_cam2base=np.eye(4).tolist(),
                        calibration={"trusted": True, "reason": ""}, **kw)


def test_the_twin_sets_the_lenss_focus_on_open_and_trusts_the_lens_only_when_it_reads_back(monkeypatch):
    from machinekind_twin import cameras

    rec = a_lens_at_focus(30)
    assert rec.applied_controls() == CameraControls(autofocus=False, focus_absolute=30)
    assert a_lens_at_focus(30, controls=CameraControls(focus_absolute=45)).applied_controls().focus_absolute == 45
    assert a_lens_at_focus(30, controls=CameraControls(autofocus=True)).applied_controls().autofocus
    ws = Workspace(robot=BOT)
    ws.add_camera(rec)
    opened = []
    held = cc.Applied(ok=["autofocus", "focus_absolute"], read={"autofocus": 0, "focus_absolute": 30})
    monkeypatch.setattr(cameras, "open_live", lambda record: opened.append(record) or LiveStandIn(held))
    hub = cameras.CameraHub(ws)
    hub.sync()
    assert opened == [rec] and rec.focus_problem == "" and rec.trusted and rec.intrinsics_problem() == ""
    moved = cc.Applied(ok=["autofocus"], failed={"focus_absolute": "odczyt 35 zamiast 30"},
                       read={"autofocus": 0, "focus_absolute": 35})
    hub.close()
    monkeypatch.setattr(cameras, "open_live", lambda record: LiveStandIn(moved))
    hub = cameras.CameraHub(ws)
    hub.sync()
    assert "odczyt 35 zamiast 30" in rec.focus_problem and not rec.trusted
    assert rec.intrinsics_problem().startswith("obiektyw niewazny w tej sesji")
    assert "focus_problem" not in ws.to_dict()["cameras"][0]
    # Applying the camera's own controls again decides anew; controls given for a moment do not.
    sent = []
    monkeypatch.setattr(cc, "apply", lambda source, controls, **kw: sent.append(controls) or held)
    hub.close()
    assert hub.apply_controls("kam", CameraControls(autofocus=False, focus_absolute=90)) is held
    assert rec.focus_problem and hub.controls_applied("kam") is None
    hub.apply_controls("kam")
    assert sent[-1] == CameraControls(autofocus=False, focus_absolute=30) and rec.focus_problem == ""
    assert hub.controls_applied("kam") is held


def test_a_lens_without_a_focus_is_trusted_as_before_whatever_the_controls_do(monkeypatch):
    from machinekind_twin import cameras

    rec = a_lens_at_focus(None, controls=CameraControls(autofocus=False, focus_absolute=0))
    assert rec.lens_focus is None and rec.applied_controls() == rec.controls
    ws = Workspace(robot=BOT)
    ws.add_camera(rec)
    monkeypatch.setattr(cameras, "open_live", lambda record: LiveStandIn(cc.Applied(skipped="brak sterowania")))
    hub = cameras.CameraHub(ws)
    hub.sync()
    assert rec.focus_problem == "" and rec.trusted and rec.intrinsics_problem() == ""


def test_a_live_camera_opens_with_the_controls_its_lens_needs(monkeypatch):
    from machinekind_twin import cameras

    seen = []

    class Stream:
        def __init__(self, cfg):
            seen.append(cfg)
            self.controls_applied = cc.Applied()

        def open(self):
            return self

    monkeypatch.setattr(cameras, "CameraStream", Stream)
    cameras.open_live(a_lens_at_focus(40))
    assert seen[0].controls == CameraControls(autofocus=False, focus_absolute=40) and seen[0].mirror is False


def test_the_lens_focus_is_kept_in_the_workspace_file_and_the_stand(tmp_path):
    ws = a_workspace(tmp_path)
    ws.camera("kamera1").intrinsics_info["focus"] = 30
    ws.camera("kamera1").focus_problem = "ostrosc 35, a obiektyw zmierzony przy 30"
    ws.save()
    data = json.loads(ws.path.read_text(encoding="utf-8"))
    cam = next(c for c in data["cameras"] if c["name"] == "kamera1")
    assert cam["intrinsics_info"]["focus"] == 30 and "focus_problem" not in cam
    again = Workspace.load(ws.path).camera("kamera1")
    assert again.lens_focus == 30 and again.focus_problem == ""
    st.save(ws, tmp_path / "stand")
    entry = st.load(tmp_path / "stand").lens["kamera1"]
    assert entry["intrinsics_info"]["focus"] == 30
    assert st.load(tmp_path / "stand").workspace().camera("kamera1").lens_focus == 30


def test_a_trust_by_hand_reaches_the_stand_and_survives_a_panel_restart(tmp_path):
    """Trusted in the panel: the workspace file and the bound stand get it (new entries, history kept), and a
    restarted panel, or the stand loaded elsewhere, still trusts the lens and the pose."""
    pytest.importorskip("viser")
    from test_twin_ui import WAVE_TIME, _click, a_camera_after_a_wave, free_port

    from machinekind_twin.ui.app import TwinApp

    ws = a_workspace(tmp_path)
    ws.remove_camera("kamera1")
    ws.add_camera(a_camera_after_a_wave("kamera1", enabled=False))   # no device to open in a test
    ws.add_camera(CameraRecord("kamera2", "/dev/video-k2", 640, 480, enabled=False))
    folder = tmp_path / "stand"
    st.save(ws, folder, name="biurko")
    before = files(folder)
    app = TwinApp(ws.path, host="127.0.0.1", port=free_port())
    try:
        app._refresh_cameras(select="kamera1")
        _click(app, "Zaufaj obiektywowi")
        _click(app, "Zaufaj polozeniu")
        app._refresh_cameras(select="kamera2")
        _click(app, "Zaufaj obiektywowi")
    finally:
        app.close()

    def trusted_by_hand(w: Workspace) -> None:
        k1, k2 = w.camera("kamera1"), w.camera("kamera2")
        assert k1.lens_by_hand and k1.pose_by_hand and k1.trusted and k1.intrinsics_problem() == ""
        assert k1.calibration["rms_px"] == 1.79 and k1.calibration["time"] == WAVE_TIME
        assert k2.lens_by_hand and k2.K == nominal_K(640, 480).tolist() and k2.intrinsics_problem() == ""

    trusted_by_hand(Workspace.load(ws.path))
    trusted_by_hand(st.load(folder).workspace())
    now = files(folder)
    assert set(before) < set(now) and all(now[f] == before[f] for f in before if f != st.STAND_FILE)
    calib = json.loads(now[st.STAND_FILE])["calib"]
    assert f"calib/poses/{calib['poses']}.json" not in before and "kamera2" in calib["lens"]
    again = TwinApp(ws.path, host="127.0.0.1", port=free_port())
    try:
        again._refresh_cameras(select="kamera1")
        assert "**zaufany recznie**" in again.trust_md.content and "**zaufane recznie**" in again.trust_md.content
        assert again.lens_untrust.visible and again.pose_untrust.visible
    finally:
        again.close()


def test_the_panel_shows_the_stand_and_saves_camera_settings_to_it(tmp_path):
    pytest.importorskip("viser")
    from test_twin_ui import _click, free_port

    from machinekind_twin.ui.app import TwinApp

    ws = a_workspace(tmp_path)
    ws.camera("kamera1").enabled = False                # no device to open in a test
    folder = tmp_path / "stand"
    st.save(ws, folder, name="biurko")
    st.mark_obsolete(folder, reason="stol przesuniety", date="2026-11-03")
    app = TwinApp(ws.path, host="127.0.0.1", port=free_port())
    try:
        assert "UWAGA: stanowisko biurko jest NIEAKTUALNE od 2026-11-03: stol przesuniety" in app.stand_html.content
        app._refresh_cameras(select="kamera1")
        assert app.cam_ctl_box.visible and app.cam_af.value == "wyl." and app.cam_focus.value == "30"
        app.cam_focus.value, app.cam_ae.value, app.cam_exposure.value = "45", "wyl.", "200"
        _click(app, "Zastosuj i zapisz")
        want = {"autofocus": False, "focus_absolute": 45, "auto_exposure": False, "exposure_absolute": 200}
        cam = json.loads((folder / "stand.json").read_text(encoding="utf-8"))["cameras"][-1]
        assert cam["controls"] == want
        app._refresh_cameras(select="sym-lewa")
        assert not app.cam_ctl_box.visible
    finally:
        app.close()


def test_a_wobbly_mount_ticked_in_the_panel_reaches_the_stand_and_survives_a_restart(tmp_path):
    """Kamery > Chwiejny statyw and its tolerance: in the workspace file and the bound stand, and back in a
    restarted panel. Cameras without it keep their records as they were."""
    pytest.importorskip("viser")
    from test_twin_ui import _change, free_port

    from machinekind_twin.ui.app import TwinApp

    ws = a_workspace(tmp_path)
    ws.camera("kamera1").enabled = False                # no device to open in a test
    folder = tmp_path / "stand"
    st.save(ws, folder, name="biurko")
    app = TwinApp(ws.path, host="127.0.0.1", port=free_port())
    try:
        app._refresh_cameras(select="kamera1")
        assert not app.cam_wobbly.value and not app.cam_wobble_px.visible
        _change(app, "Chwiejny statyw", True)
        assert app.cam_wobble_px.visible and app.cam_wobble_px.value == 4.0
        _change(app, "Tolerancja statywu [px]", 6.0)
        assert "chwiejny statyw, tolerancja 6 px" in app.cam_detail.content
    finally:
        app.close()

    for cams in (json.loads(ws.path.read_text(encoding="utf-8"))["cameras"],
                 json.loads((folder / st.STAND_FILE).read_text(encoding="utf-8"))["cameras"]):
        by_name = {c["name"]: c for c in cams}
        assert by_name["kamera1"]["wobbly"] is True and by_name["kamera1"]["wobble_px"] == 6.0
        assert not {"wobbly", "wobble_px"} & set(by_name["sym-lewa"])
    for w in (Workspace.load(ws.path), st.load(folder).workspace()):
        assert w.camera("kamera1").wobbly and w.camera("kamera1").wobble_px == 6.0
        assert not w.camera("sym-lewa").wobbly
    again = TwinApp(ws.path, host="127.0.0.1", port=free_port())
    try:
        again._refresh_cameras(select="kamera1")
        assert again.cam_wobbly.value and again.cam_wobble_px.value == 6.0 and again.cam_wobble_px.visible
    finally:
        again.close()

    data = json.loads((folder / st.STAND_FILE).read_text(encoding="utf-8"))
    data["cameras"][2]["wobble_px"] = 0
    (folder / st.STAND_FILE).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(st.StandError, match=r"cameras\[2\] \(kamera1\)\.wobble_px: oczekiwano liczby > 0"):
        st.load(folder)
