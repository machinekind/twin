"""The per-rig file `workspace/grasp.json`: reading, writing and the refusal reasons."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from lerobot_mp.paths import data_path
from lerobot_mp.twin.grasp import rig as rig_mod
from lerobot_mp.twin.grasp.contract import (
    FREEZE_DISABLED,
    TIP_CLEAR,
    TOUCH_EXPECTED_Z,
    WRIST_DIST_2026_09_26,
    WRIST_K_2026_09_26,
)
from lerobot_mp.twin.grasp.rig import DEFAULT_PATH, Rig, RigError

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc).timestamp()
ORDER = ["table not taught", "table touch implausible", "grasp point not taught", "table touch far",
         "teaching old", "object height not set", "wrist camera not set", "dot not checked", "dot off",
         "latency not measured", "latency high", "jaw stroke not measured", "gripper freeze undecided",
         "no shadow episodes"]


def stamp(hours_ago: float) -> str:
    t = datetime.fromtimestamp(NOW, timezone.utc) - timedelta(hours=hours_ago)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def complete() -> dict:
    """A rig file on which every mode may run."""
    return {
        "version": 1,
        "table": {"touch_tcp_z": 0.0342, "tcp": [0.01, 0.30, 0.0342], "joints": [90, 120, -80, 40, 0, 0],
                  "taught_at": stamp(1.0)},
        "grasp": {"tcp": [0.0, 0.33, 0.036], "yaw": 0.1, "joints": [90, 110, -75, 50, 0, 5],
                  "taught_at": stamp(0.5), "object_height": 0.030, "joint_offsets_deg": [0.0] * 6},
        "gripper": {"stroke_close_s_effort": 1.28, "stroke_close_s_operator": 1.35, "freeze_threshold": 100.0,
                    "gap_mm": {}},
        "wrist": {"source": "/dev/video2", "K": None, "dist": None, "upside_down": True, "dot_error_px": 3.1},
        "latency": {"camera_s": 0.045, "actuation_s": 0.03, "infer_p95_s": 0.02, "total_s": 0.11},
        "guard": {"effort_abs": 12.0},
        "overrides": {"shadow_done": "10 clean shadow episodes, 2026-09-27"},
    }


def write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def rig_from(data: dict) -> Rig:
    return Rig(**{k: copy.deepcopy(v) for k, v in data.items() if k != "version"})


def in_order(reasons: list[str]) -> bool:
    return [r for r in ORDER if r in reasons] == reasons and len(set(reasons)) == len(reasons)


def test_a_missing_file_is_an_empty_rig_with_that_path(tmp_path):
    path = tmp_path / "nowhere" / "grasp.json"
    rig = Rig.load(path)
    assert rig.path == path and rig.version == 1
    assert rig.to_dict() == {"version": 1, "table": {}, "grasp": {}, "gripper": {}, "wrist": {},
                             "latency": {}, "guard": {}, "overrides": {}}
    assert not path.parent.exists()


def test_the_default_path_is_workspace_grasp_json_through_data_path():
    assert DEFAULT_PATH == "workspace/grasp.json"
    assert data_path(DEFAULT_PATH).name == "grasp.json"


def test_save_and_load_round_trip_including_the_version(tmp_path):
    path = tmp_path / "ws" / "grasp.json"
    rig = rig_from(complete())
    rig.path = path
    rig.save()
    assert path.is_file() and not list(path.parent.glob("*.tmp"))
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["version"] == 1
    back = Rig.load(path)
    assert back.to_dict() == complete() == rig.to_dict()
    assert back.path == path


def test_values_measured_with_numpy_are_saved_as_plain_numbers(tmp_path):
    rig = Rig(path=tmp_path / "grasp.json")
    rig.table = {"touch_tcp_z": np.float32(0.0342), "tcp": np.array([0.01, 0.30, 0.0342]),
                 "joints": {"arm_joint1": np.float64(90.0)}, "taught_at": stamp(1.0)}
    rig.grasp = {"tcp": np.array([0.0, 0.33, 0.036]), "yaw": np.float64(0.1), "taught_at": stamp(0.5),
                 "object_height": 0.03, "joint_offsets_deg": np.zeros(6)}
    rig.wrist = {"source": np.int64(2), "upside_down": np.bool_(True), "dot_error_px": np.float32(3.0)}
    rig.save()
    back = Rig.load(rig.path)
    assert back.table["tcp"] == [0.01, 0.30, 0.0342] and back.table["joints"] == {"arm_joint1": 90.0}
    assert back.touch_tcp_z() == pytest.approx(0.0342, abs=1e-8) and back.wrist_source() == "2"
    assert back.grasp["joint_offsets_deg"] == [0.0] * 6 and back.upside_down() is True
    assert back.to_dict() == rig.to_dict()


def test_to_dict_is_a_copy():
    rig = rig_from(complete())
    d = rig.to_dict()
    d["table"]["touch_tcp_z"] = 1.0
    d["grasp"]["tcp"][0] = 5.0
    assert rig.touch_tcp_z() == 0.0342 and rig.grasp["tcp"][0] == 0.0


def test_an_unknown_top_level_key_raises(tmp_path):
    data = complete()
    data["reach"] = {}
    with pytest.raises(RigError, match="reach"):
        Rig.load(write(tmp_path / "grasp.json", data))
    assert issubclass(RigError, ValueError)


@pytest.mark.parametrize("version", [2, 0, "1", 1.5, True, None])
def test_a_file_with_another_version_raises(tmp_path, version):
    data = complete()
    data["version"] = version
    with pytest.raises(RigError):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_file_without_a_version_is_read_as_version_1(tmp_path):
    data = complete()
    del data["version"]
    rig = Rig.load(write(tmp_path / "grasp.json", data))
    assert rig.version == 1 and rig.to_dict() == complete()


def test_save_refuses_a_malformed_value_and_keeps_the_old_file(tmp_path):
    path = write(tmp_path / "grasp.json", complete())
    before = path.read_bytes()
    rig = Rig.load(path)
    rig.table["touch_tcp_z"] = float("nan")
    with pytest.raises(RigError):
        rig.save()
    rig = Rig.load(path)
    rig.version = 2
    with pytest.raises(RigError):
        rig.save()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grasp.json"]


@pytest.mark.parametrize("section,key,value", [
    ("table", "touch_tcp_z", "0.034"), ("table", "touch_tcp_z", float("nan")), ("grasp", "tcp", [0.0, 0.33]),
    ("grasp", "yaw", True), ("grasp", "taught_at", "yesterday"), ("wrist", "upside_down", "yes"),
    ("wrist", "K", [[1, 0, 0], [0, 1, 0]]), ("gripper", "stroke_close_s_effort", 0.0),
    ("grasp", "joint_offsets_deg", [0.0] * 5), ("latency", "total_s", "fast"),
])
def test_a_malformed_value_raises_when_loading(tmp_path, section, key, value):
    data = complete()
    data[section][key] = value
    with pytest.raises(RigError):
        Rig.load(write(tmp_path / "grasp.json", data))


def test_a_section_that_is_not_an_object_raises(tmp_path):
    data = complete()
    data["table"] = [0.034]
    with pytest.raises(RigError):
        Rig.load(write(tmp_path / "grasp.json", data))
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RigError):
        Rig.load(tmp_path / "bad.json")


def test_a_complete_rig_may_run_every_mode():
    rig = rig_from(complete())
    for mode in ("dry", "sim", "shadow", "real"):
        assert rig.missing(mode, simulated=False, now=NOW) == []
        assert rig.missing(mode, simulated=True, now=NOW) == []


def test_an_empty_rig_on_a_real_backend_misses_per_mode_in_the_listed_order():
    rig = Rig()
    assert rig.missing("dry", simulated=False, now=NOW) == [
        "table not taught", "grasp point not taught", "wrist camera not set"]
    assert rig.missing("shadow", simulated=False, now=NOW) == [
        "table not taught", "grasp point not taught", "object height not set", "wrist camera not set",
        "dot not checked", "latency not measured"]
    assert rig.missing("real", simulated=False, now=NOW) == [
        "table not taught", "grasp point not taught", "object height not set", "wrist camera not set",
        "dot not checked", "latency not measured", "jaw stroke not measured", "gripper freeze undecided",
        "no shadow episodes"]


def test_an_empty_rig_on_a_simulated_backend_misses_nothing():
    for mode in ("dry", "sim", "shadow", "real"):
        assert Rig().missing(mode, simulated=True, now=NOW) == []


def test_every_limit_at_once_gives_every_reason_in_the_listed_order():
    data = complete()
    data["table"].update(touch_tcp_z=0.06, tcp=[0.13, 0.30, 0.06], taught_at=stamp(13.0))
    data["grasp"].update(yaw=None, object_height=0.2)
    data["gripper"].update(stroke_close_s_effort=None, stroke_close_s_operator=None, freeze_threshold=None)
    data["wrist"].update(source=None, dot_error_px=9.0)
    data["latency"].update(total_s=0.2)
    data["overrides"] = {}
    rig = rig_from(data)
    want = ["table touch implausible", "grasp point not taught", "table touch far", "teaching old",
            "object height not set", "wrist camera not set", "dot off", "latency high",
            "jaw stroke not measured", "gripper freeze undecided", "no shadow episodes"]
    assert rig.missing("real", simulated=False, now=NOW) == want
    assert in_order(want)
    assert rig.missing("shadow", simulated=False, now=NOW) == want[:8]
    assert rig.missing("dry", simulated=False, now=NOW) == [
        "table touch implausible", "grasp point not taught", "table touch far", "teaching old",
        "wrist camera not set"]
    assert rig.missing("real", simulated=True, now=NOW) == []


def test_overrides_lift_the_matching_limits():
    data = complete()
    data["table"]["touch_tcp_z"] = 0.06
    data["table"]["taught_at"] = stamp(30.0)
    data["wrist"]["dot_error_px"] = 12.0
    data["latency"]["total_s"] = 0.3
    rig = rig_from(data)
    assert rig.missing("real", simulated=False, now=NOW) == [
        "table touch implausible", "teaching old", "dot off", "latency high"]
    rig.overrides.update(table="mount moved, checked by hand", taught="same day", dot="known offset",
                         latency="slow camera, accepted")
    assert rig.missing("real", simulated=False, now=NOW) == []
    rig.overrides["dot"] = ""                              # an empty note is no override
    assert rig.missing("real", simulated=False, now=NOW) == ["dot off"]
    rig.overrides.pop("shadow_done")
    assert rig.missing("real", simulated=False, now=NOW) == ["dot off", "no shadow episodes"]
    assert rig.missing("shadow", simulated=False, now=NOW) == ["dot off"]


def test_teaching_older_than_twelve_hours_needs_the_taught_override():
    data = complete()
    data["grasp"]["taught_at"] = stamp(11.9)
    assert rig_from(data).missing("dry", simulated=False, now=NOW) == []
    data["grasp"]["taught_at"] = stamp(12.1)
    rig = rig_from(data)
    assert rig.missing("dry", simulated=False, now=NOW) == ["teaching old"]
    assert rig.missing("dry", simulated=False) == []          # no clock, no age check
    rig.overrides["taught"] = "operator: point unchanged since yesterday"
    assert rig.missing("dry", simulated=False, now=NOW) == []
    data["grasp"]["taught_at"] = stamp(0.1)
    data["table"]["taught_at"] = None
    assert rig_from(data).missing("dry", simulated=False, now=NOW) == ["teaching old"]


def test_taught_age_reads_iso_stamps_with_or_without_a_zone():
    rig = rig_from(complete())
    ages = rig.taught_age_s(NOW)
    assert ages == {"table": pytest.approx(3600.0), "grasp": pytest.approx(1800.0)}
    rig.table["taught_at"] = "2026-09-27T11:00:00+00:00"
    rig.grasp["taught_at"] = "2026-09-27T10:00:00"            # naive stamps are UTC
    assert rig.taught_age_s(NOW) == {"table": pytest.approx(3600.0), "grasp": pytest.approx(7200.0)}
    assert Rig().taught_age_s(NOW) == {"table": None, "grasp": None}


def test_a_table_taught_far_from_the_grasp_point_is_refused():
    data = complete()
    data["table"]["tcp"] = [0.12, 0.33, 0.0342]                # 0.12 m from the grasp point in x
    assert rig_from(data).missing("dry", simulated=False, now=NOW) == ["table touch far"]
    data["table"]["tcp"] = [0.0, 0.33 - 0.09, 0.0342]
    assert rig_from(data).missing("dry", simulated=False, now=NOW) == []
    data["table"]["tcp"] = None                                  # a touch without its position
    assert rig_from(data).missing("dry", simulated=False, now=NOW) == ["table touch far"]


@pytest.mark.parametrize("height,ok", [(None, False), (0.004, False), (0.005, True), (0.03, True),
                                       (0.10, True), (0.11, False)])
def test_object_height_must_be_set_within_limits_for_shadow_and_real(height, ok):
    data = complete()
    data["grasp"]["object_height"] = height
    rig = rig_from(data)
    want = [] if ok else ["object height not set"]
    assert rig.missing("shadow", simulated=False, now=NOW) == want
    assert rig.missing("real", simulated=False, now=NOW) == want
    assert rig.missing("dry", simulated=False, now=NOW) == []


def test_the_believed_point_is_the_taught_xy_at_touch_plus_tip_clearance():
    rig = rig_from(complete())
    np.testing.assert_array_equal(rig.believed(), [0.0, 0.33, 0.0342 + TIP_CLEAR])
    assert TIP_CLEAR == 0.006
    assert rig.grasp_yaw() == 0.1 and rig.touch_tcp_z() == 0.0342
    rig.table["touch_tcp_z"] = None
    assert rig.believed() is None
    assert Rig(table={"touch_tcp_z": 0.034}).believed() is None


def test_upside_down_and_object_height_are_read_from_the_file(tmp_path):
    data = complete()
    data["wrist"]["upside_down"] = False
    data["grasp"]["object_height"] = 0.045
    rig = Rig.load(write(tmp_path / "grasp.json", data))
    assert rig.upside_down() is False and rig.object_height() == 0.045
    assert Rig().upside_down() is True and Rig().object_height() is None


def test_readers_give_defaults_and_plain_values():
    rig = rig_from(complete())
    assert rig.stroke_s() == 1.35 and rig.jaw_units_s() == pytest.approx(100.0 / 1.35)
    assert rig.freeze_threshold() == 100.0 and rig.wrist_source() == "/dev/video2"
    assert rig.camera_latency_s() == 0.045 and rig.effort_abs() == 12.0
    empty = Rig()
    assert empty.stroke_s() is None and empty.jaw_units_s() is None and empty.freeze_threshold() is None
    assert empty.camera_latency_s() == 0.0 and empty.effort_abs() == 12.0 and empty.wrist_source() is None
    assert empty.touch_tcp_z() is None and empty.grasp_yaw() is None
    assert Rig(wrist={"source": 2}).wrist_source() == "2"
    assert Rig(wrist={"source": "  "}).wrist_source() is None
    assert Rig(gripper={"stroke_close_s_effort": 1.3}).stroke_s() == 1.3


def test_intrinsics_are_the_contract_default_or_the_rigs_own_pair():
    K, dist = Rig().intrinsics()
    np.testing.assert_array_equal(K, WRIST_K_2026_09_26)
    np.testing.assert_array_equal(dist, WRIST_DIST_2026_09_26)
    K[0, 0] = 1.0
    assert WRIST_K_2026_09_26[0, 0] != 1.0
    own = Rig(wrist={"K": [[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]], "dist": [0.1, 0.0, 0.0, 0.0]})
    K, dist = own.intrinsics()
    assert K.shape == (3, 3) and K[0, 2] == 960 and dist.tolist() == [0.1, 0.0, 0.0, 0.0]
    with pytest.raises(RigError):
        Rig(wrist={"K": [[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]]}).intrinsics()


def test_for_sim_holds_every_sim_value_and_nothing_measured():
    rig = Rig.for_sim()
    assert rig.path is None
    assert rig.touch_tcp_z() == TOUCH_EXPECTED_Z == 0.0339
    assert rig.object_height() == 0.030
    assert rig.camera_latency_s() == 0.0
    assert "stroke_close_s_operator" in rig.gripper and rig.stroke_s() is None
    assert rig.freeze_threshold() == FREEZE_DISABLED
    assert rig.wrist_source() == "sim" and rig.wrist["dot_error_px"] == 0.0
    assert rig.latency["total_s"] == 0.0 and rig.overrides["shadow_done"] == "sim"
    assert rig.believed() is None and rig.grasp_yaw() is None
    for mode in ("dry", "sim", "shadow", "real"):
        assert rig.missing(mode, simulated=True, now=NOW) == []


def test_a_rig_without_a_path_is_never_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(rig_mod, "data_path", lambda rel: tmp_path / rel)
    monkeypatch.chdir(tmp_path)
    for rig in (Rig.for_sim(), Rig()):
        with pytest.raises(RigError):
            rig.save()
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_mode_raises():
    with pytest.raises(ValueError):
        Rig().missing("fast", simulated=False)


def test_workspace_twin_json_is_never_touched(tmp_path, monkeypatch):
    real = data_path("workspace/twin.json")
    before = (real.is_file(), hashlib.sha256(real.read_bytes()).hexdigest() if real.is_file() else None)
    monkeypatch.setattr(rig_mod, "data_path", lambda rel: tmp_path / rel)
    monkeypatch.chdir(tmp_path)
    rig = Rig.load()
    assert rig.path == tmp_path / "workspace" / "grasp.json"
    rig.table.update(touch_tcp_z=0.034)
    rig.save()
    Rig.load().missing("real", simulated=False, now=NOW)
    Rig.for_sim().missing("real", simulated=True)
    assert sorted(p.name for p in (tmp_path / "workspace").iterdir()) == ["grasp.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["workspace"]
    after = (real.is_file(), hashlib.sha256(real.read_bytes()).hexdigest() if real.is_file() else None)
    assert after == before
