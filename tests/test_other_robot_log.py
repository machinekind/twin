"""A drive of another robot written down: its odometry, where the twin put its base and the tag sightings, so
the odometry's drift can be read against the cameras afterwards (`drivelog`, `Twin.log_drive`)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from test_other_robot_tag import ANCHOR, ODOMETRY, STOOD, flat, nudged, seen_at, station

from machinekind_twin import drivelog, passive
from machinekind_twin.backends.factory import BaseDriveArm
from machinekind_twin.cli import main as twin_cli
from machinekind_twin.passive import PassiveReader
from machinekind_twin.runtime import Twin

#: 30 cm ahead, then 40 cm to the left: 70 cm walked, 50 cm from the start. The trunk bobs 3 cm on the way.
LEG = [flat(0.0, 0.0, 0.0, z=0.27), flat(0.3, 0.0, 0.0, z=0.24), flat(0.3, 0.4, 90.0, z=0.27)]


@pytest.fixture
def twin(walker):
    twin = Twin(station(walker))
    yield twin
    twin.close()


def lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FullDisk:
    """A log file on a full disk: no line goes in, and closing it closes the real file."""

    def __init__(self, real):
        self.real = real
        self.closed = False

    def write(self, text: str) -> int:
        raise OSError(28, "No space left on device")

    def close(self) -> None:
        self.real.close()
        self.closed = True


def on_a_full_disk(twin: Twin) -> FullDisk:
    """The walker's running log, from now on on a full disk."""
    log = twin.drive_logs["walker"]
    log["file"] = FullDisk(log["file"])
    return log["file"]


def walk(twin: Twin, start: float) -> None:
    """The walker reports the poses of `LEG`, a tenth of a second apart from `start`."""
    for step, T in enumerate(LEG):
        twin.set_passive_joints("walker", {}, (T, start + 0.1 * step))


def test_the_header_names_the_robot_where_its_record_puts_it_and_its_tag(twin, walker, tmp_path):
    path = twin.log_drive("walker", True)
    header = lines(path)[0]

    assert path.parent == tmp_path / "workspace" / "episodes"
    assert path.name.startswith("other-walker-")
    assert header["format"] == "machinekind-twin-other-robot"
    assert header["version"] == 1
    assert header["robot"] == "walker"
    assert header["description"] == walker
    assert header["xyz"] == [0.35, 0.15, 0.0]
    assert header["yaw_deg"] == 90.0
    assert header["base_tag"] == {"id": 7, "size": 0.08}


def test_a_reported_base_pose_is_one_line_in_the_robots_frame_and_in_the_arms(twin):
    path = twin.log_drive("walker", True)
    twin.set_passive_joints("walker", {}, (ODOMETRY, time.monotonic()))
    written = lines(path)
    turned = np.radians(110.0) / 2                               # the record's 90 degrees and odometry's 20

    assert len(written) == 2
    assert written[1]["odometry"]["position"] == pytest.approx(ODOMETRY[:3, 3])
    assert written[1]["base"]["position"] == pytest.approx(STOOD[:3, 3])
    assert written[1]["base"]["quaternion"] == pytest.approx([np.cos(turned), 0.0, 0.0, np.sin(turned)])


def test_a_commanded_velocity_is_on_the_lines_while_it_runs_and_null_after(twin, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(passive, "time", SimpleNamespace(monotonic=lambda: clock.now))
    plugin = SimpleNamespace(connect=lambda: None, disconnect=lambda: None, read_joints=dict,
                             base_pose=lambda: (ODOMETRY, time.monotonic()), send_velocity=lambda *v: None)
    reader = PassiveReader("walker", lambda: BaseDriveArm(plugin, "drivebus"),
                           lambda joints, base: twin.set_passive_joints("walker", joints, base))
    twin._readers["walker"] = reader
    path = twin.log_drive("walker", True)
    reader.drive(0.1, 0.0, 0.2, 1.0)
    reader.read_once()
    clock.now += 2.0
    reader.read_once()

    assert [line["velocity"] for line in lines(path)[1:]] == [[0.1, 0.0, 0.2], None]


def test_a_tag_sighting_is_a_line_with_the_measured_base_and_the_drift_after_it(twin, monkeypatch):
    now = time.monotonic()
    seen = nudged(STOOD, dx=0.04, turn_deg=10.0)
    path = twin.log_drive("walker", True)
    twin.set_passive_joints("walker", {}, (ODOMETRY, now))
    seen_at(twin, monkeypatch, seen, now)
    twin.locate_other("walker")
    tag = lines(path)[2]["tag"]

    assert tag["base"]["position"] == pytest.approx(seen[:3, 3])
    assert tag["cameras"] == 1
    assert tag["drift_m"] == pytest.approx(0.02)
    assert tag["drift_deg"] == pytest.approx(5.0)


def test_turning_the_log_off_ends_it_with_the_path_the_odometry_walked_in_the_plane(twin):
    path = twin.log_drive("walker", True)
    walk(twin, time.monotonic())
    twin.log_drive("walker", False)

    assert lines(path)[-1]["end"]["path_m"] == pytest.approx(0.7)
    assert twin.drive_logs == {}


def test_stopping_the_readers_ends_an_open_log(twin):
    path = twin.log_drive("walker", True)
    twin.stop_passive()

    assert "end" in lines(path)[-1]


def test_the_summary_reads_back_the_path_the_sightings_and_the_final_and_largest_drift(twin, monkeypatch):
    start = time.monotonic()
    there = ANCHOR @ LEG[-1]
    path = twin.log_drive("walker", True)
    walk(twin, start)
    seen_at(twin, monkeypatch, nudged(there, dx=0.04), start + 0.2)
    twin.locate_other("walker")                                  # half of 4 cm: the copy 2 cm off odometry
    twin.set_passive_joints("walker", {}, (LEG[-1], start + 0.25))
    seen_at(twin, monkeypatch, there, start + 0.25)
    twin.locate_other("walker")                                  # half of those 2 cm back: 1 cm off
    twin.set_passive_joints("walker", {}, (LEG[-1], start + 0.3))
    twin.log_drive("walker", False)
    summary = drivelog.summary(path)

    assert summary is not None
    assert summary["path_m"] == pytest.approx(0.7)
    assert summary["sightings"] == 2
    assert summary["drift_m"] == pytest.approx(0.01)
    assert summary["max_drift_m"] == pytest.approx(0.02)


def test_the_end_line_and_the_summary_give_the_same_numbers(twin, monkeypatch):
    start = time.monotonic()
    path = twin.log_drive("walker", True)
    walk(twin, start)
    seen_at(twin, monkeypatch, nudged(ANCHOR @ LEG[-1], dx=0.03, turn_deg=7.0), start + 0.2)
    twin.locate_other("walker")
    twin.set_passive_joints("walker", {}, (LEG[-1] @ flat(0.2, 0.0, 0.0), start + 0.3))
    twin.log_drive("walker", False)
    end = lines(path)[-1]["end"]
    summary = drivelog.summary(path)

    assert summary is not None
    assert end == {"path_m": summary["path_m"], "sightings": summary["sightings"],
                   "drift_m": summary["drift_m"], "drift_deg": summary["drift_deg"]}


def test_the_end_line_comes_from_running_totals_not_from_the_file_read_back(twin, tmp_path):
    path = twin.log_drive("walker", True)
    walk(twin, time.monotonic())
    moved = path.rename(tmp_path / "moved.jsonl")              # the open file writes on under its new name
    twin.log_drive("walker", False)

    assert lines(moved)[-1]["end"]["path_m"] == pytest.approx(0.7)


def test_the_final_drift_is_the_drift_at_the_last_pose_after_walking_on_from_the_sighting(twin, monkeypatch):
    now = time.monotonic()
    path = twin.log_drive("walker", True)
    twin.set_passive_joints("walker", {}, (ODOMETRY, now))
    seen_at(twin, monkeypatch, nudged(STOOD, turn_deg=10.0), now)
    twin.locate_other("walker")
    twin.set_passive_joints("walker", {}, (ODOMETRY @ flat(0.5, 0.0, 0.0), now + 0.1))
    twin.log_drive("walker", False)

    # Turned 5 degrees about where it stood, the copy is 2 x 0.5 m x sin(2.5 deg) off once 0.5 m on.
    assert lines(path)[-1]["end"]["drift_m"] == pytest.approx(2 * 0.5 * np.sin(np.radians(2.5)), abs=1e-6)


def test_a_line_the_disk_refuses_drops_the_log_and_keeps_why(twin):
    twin.log_drive("walker", True)
    file = on_a_full_disk(twin)
    twin.set_passive_joints("walker", {}, (ODOMETRY, time.monotonic()))

    assert twin.drive_logs == {}
    assert file.closed
    assert "No space left on device" in twin.drive_log_notes["walker"]


def test_a_log_the_disk_refuses_to_end_still_lets_the_readers_stop(twin):
    twin.log_drive("walker", True)
    file = on_a_full_disk(twin)
    twin.stop_passive()

    assert file.closed
    assert "No space left on device" in twin.drive_log_notes["walker"]


def test_an_off_that_lands_while_the_log_is_opening_still_ends_it(twin, monkeypatch):
    opening = threading.Event()
    release = threading.Event()
    start = drivelog.start

    def slow_start(*args):
        opening.set()
        release.wait(5.0)
        return start(*args)

    monkeypatch.setattr(drivelog, "start", slow_start)
    on = threading.Thread(target=twin.log_drive, args=("walker", True))
    on.start()
    opening.wait(5.0)
    off = threading.Thread(target=twin.log_drive, args=("walker", False))
    off.start()
    off.join(0.1)                                               # an off that does not wait is done by now
    release.set()
    on.join(5.0)
    off.join(5.0)

    assert twin.drive_logs == {}


def test_twin_episode_prints_the_summary_of_a_drive_log(twin, capsys):
    path = twin.log_drive("walker", True)
    walk(twin, time.monotonic())
    twin.log_drive("walker", False)

    assert twin_cli(["episode", str(path)]) == 0
    assert "droga wg odometrii 0.70 m" in capsys.readouterr().out


def test_twin_episode_never_takes_a_drive_log_for_a_strategy_run(twin):
    path = twin.log_drive("walker", True)
    twin.log_drive("walker", False)

    assert twin_cli(["episode", str(path.parent)]) == 1
