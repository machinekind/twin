"""Stanowisko blizniaka: zaufanie do pozy kamery po kalibracji (bramka percepcji).

Bez renderu i bez sprzetu - to jest warstwa danych, na ktorej stoi reszta.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.calib.handeye import CameraFit, Fit  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace, nominal_K  # noqa: E402


def charuco_camera(name: str = "front", trusted: bool = True) -> CameraRecord:
    return CameraRecord(name, K=nominal_K(640, 480).tolist(), dist=[0.1, -0.05, 0.0, 0.0, 0.0],
                        intrinsics_from="szachownica",
                        intrinsics_info={"rms_px": 0.3, "trusted": trusted, "reason": "" if trusted else "malo kadrow"})


def trusted_fit(name: str = "front") -> Fit:
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.2]
    return Fit({name: CameraFit(T, 0.18, 0.6, 30, 12.5, trusted=True, reason="")})


def test_real_camera_pose_is_not_trusted_without_a_trusted_charuco_K():
    """Dopasowanie z nominalnym K przechodzi prog residuum, a poza jest przesunieta o centymetry."""
    ws = Workspace()
    ws.add_camera(CameraRecord("nominalna"))                       # krok 1 pominiety
    ws.add_camera(charuco_camera("slaba", trusted=False))           # sesja ChArUco niezaufana
    ws.add_camera(CameraRecord("sym", "sim", K=nominal_K(640, 480).tolist(), intrinsics_from="symulacja"))
    fit = Fit({n: trusted_fit(n).cameras[n] for n in ("nominalna", "slaba", "sym")})
    ws.apply_fit(fit)
    assert not ws.camera("nominalna").trusted and "nominalne" in ws.camera("nominalna").calibration["reason"]
    assert not ws.camera("slaba").trusted and "malo kadrow" in ws.camera("slaba").calibration["reason"]
    assert ws.camera("sym").trusted                                 # symulowana zna swoje K


def test_room_round_trips_through_the_workspace_file_and_none_is_not_written(tmp_path):
    from lerobot_mp.twin.workspace import ROOM_OFFSET, room_config, room_path

    ws = Workspace()
    ws.save(tmp_path / "twin.json")
    assert "room" not in (tmp_path / "twin.json").read_text()        # no room = no key, older files still load
    ws.room = room_config("assets/rooms/pokoj.glb")
    ws.room["offset"] = [0.1, -0.2, -0.7]
    ws.save()
    back = Workspace.load(tmp_path / "twin.json")
    assert back.room == ws.room and back.room["hide_table"] and back.room["hide_floor"]
    assert room_path(back.room).is_file()                            # relative to the repo root
    # A new file keeps the placement it had; a fresh room starts at the default.
    assert room_config("other.glb", back.room)["offset"] == [0.1, -0.2, -0.7]
    assert room_config("other.glb")["offset"] == list(ROOM_OFFSET)


def test_default_room_placement_puts_the_rooms_desk_on_the_twins_table():
    """The dark desk of pokoj.glb (top at glTF y 0.743, x 0.72..1.57, z -1.79..-0.15) -> table top, centre."""
    from lerobot_mp.twin.ui.bridge import room_pose
    from lerobot_mp.twin.workspace import ROOM_OFFSET, ROOM_YAW_DEG, Table

    T = room_pose(ROOM_OFFSET, ROOM_YAW_DEG)
    centre = T @ np.array([1.145, 0.743, -0.97, 1.0])
    t = Table(size=(1.2, 0.8), base_xy=(-0.45, 0.0))
    assert centre[:3] == pytest.approx([-t.base_xy[0], -t.base_xy[1], 0.0], abs=1e-3)
    along = T[:3, :3] @ np.array([0.0, 0.0, 1.0])                    # the desk's long side (glTF z)
    assert abs(along[1]) == pytest.approx(1.0)
    window = T[:3, :3] @ np.array([1.0, 0.0, 0.0])                   # the window wall is at glTF +x
    assert window == pytest.approx([1.0, 0.0, 0.0])
    assert (T[:3, :3] @ np.array([0.0, 1.0, 0.0])) == pytest.approx([0.0, 0.0, 1.0])   # glTF up = z up


def test_workspace_cli_writes_and_removes_the_room(tmp_path, capsys):
    from lerobot_mp.twin import cli

    path = tmp_path / "twin.json"
    assert cli.main(["workspace", "--path", str(path), "--room", "assets/rooms/pokoj.glb"]) == 0
    assert Workspace.load(path).room["glb"] == "assets/rooms/pokoj.glb"
    assert "pokoj assets/rooms/pokoj.glb" in capsys.readouterr().out
    assert cli.main(["workspace", "--path", str(path), "--room", str(tmp_path / "missing.glb")]) == 1
    assert cli.main(["workspace", "--path", str(path), "--no-room"]) == 0
    assert Workspace.load(path).room is None
