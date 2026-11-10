"""A walking robot next to the arm, driven from the panel by base velocity; its joints are still only read."""

from __future__ import annotations

import contextlib
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from conftest import BOT, WALKER
from machinekind_twin_api.entry_points import BACKENDS
from test_other_robot_base import ON_THE_FLOOR
from test_twin_other_robots import FAKE_BUS, wait_for

from machinekind_twin import passive, robots
from machinekind_twin import stand as st
from machinekind_twin.backends import open_read_only
from machinekind_twin.backends.factory import BaseDriveArm
from machinekind_twin.clock import VirtualClock
from machinekind_twin.passive import PassiveReader
from machinekind_twin.robots import get_spec, robot_config
from machinekind_twin.runtime import Twin
from machinekind_twin.workspace import RobotRecord, Workspace

FORWARD = (0.2, 0.0, 0.0)
STILL = (0.0, 0.0, 0.0)

DRIVE_BUS = FAKE_BUS + '''

from machinekind_twin_api import base_drive_capable


class Walker(Arm):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.bus.ctx = ctx
        self.bus.sent = []

    def send_velocity(self, forward, left, turn):
        self.bus.sent.append((forward, left, turn))


@read_only_capable
@base_drive_capable
def create_walker(ctx):
    return Walker(ctx)
'''


@pytest.fixture
def drivebus(install_dist):
    """The fake bus as `drivebus`, which may drive the base, and `watchbus`, which may only read."""
    install_dist("drivebus-dist", {BACKENDS: {"drivebus": "drivebus_mod:create_walker",
                                              "watchbus": "drivebus_mod:create"}},
                 {"drivebus_mod": DRIVE_BUS})
    import drivebus_mod

    drivebus_mod.BUSES.clear()
    return drivebus_mod


@pytest.fixture
def clock(monkeypatch):
    """The readers' `time.monotonic`, moved by hand (`clock.now` [s])."""
    fake = SimpleNamespace(now=100.0)
    monkeypatch.setattr(passive, "time", SimpleNamespace(monotonic=lambda: fake.now))
    return fake


def station(walker: str, backend: str = "drivebus") -> Workspace:
    ws = Workspace(robot=BOT)
    ws.other_robots.append(RobotRecord("walker", walker, xyz=list(ON_THE_FLOOR), backend=backend, base_drive=True))
    return ws


@contextlib.contextmanager
def a_live_walker(walker: str, **kw):
    """A twin reading the walker live through `drivebus`; closed on exit."""
    twin = Twin(station(walker), **kw)
    try:
        twin.start_passive()
        wait_for(lambda: twin.passive_status()["walker"] == "odczyt na zywo")
        yield twin
    finally:
        twin.close()


def a_reader(log: list) -> PassiveReader:
    """A reader of a drivable robot whose plugin logs every base velocity and its disconnect."""
    plugin = SimpleNamespace(connect=lambda: None, disconnect=lambda: log.append("disconnect"), read_joints=dict,
                             send_velocity=lambda *v: log.append(v))
    return PassiveReader("walker", lambda: BaseDriveArm(plugin, "drivebus"), lambda joints, base: None)


def test_a_factory_not_marked_for_it_is_never_created_and_the_status_says_why(walker, drivebus):
    twin = Twin(station(walker, backend="watchbus"))
    try:
        twin.start_passive()
        wait_for(lambda: twin.passive_status()["walker"] != "laczenie...")
        status = twin.passive_status()["walker"]
    finally:
        twin.close()

    assert "nie umie jezdzic podstawa robota (base_drive)" in status
    assert drivebus.BUSES == []


def test_a_robot_without_base_max_vel_is_refused(walker, drivebus):
    spec = dataclasses.replace(get_spec(walker), base_max_vel=None)

    with pytest.raises(ValueError, match="nie ma base_max_vel"):
        open_read_only(robot_config(spec), spec, "drivebus", None, base_drive=True)


def test_the_plugin_is_opened_to_drive_the_base_not_read_only(walker, drivebus):
    spec = get_spec(walker)
    open_read_only(robot_config(spec), spec, "drivebus", None, base_drive=True)
    ctx = drivebus.BUSES[-1].ctx

    assert ctx.read_only is False
    assert ctx.base_drive is True


def test_a_robot_opened_read_only_still_has_nothing_to_send(walker, drivebus):
    spec = get_spec(walker)
    arm = open_read_only(robot_config(spec), spec, "drivebus", None)

    assert not hasattr(arm, "send_velocity")


def test_a_negative_base_speed_in_the_description_is_refused(walker):
    spec = dataclasses.replace(get_spec(walker), base_max_vel=(0.3, -0.2, 0.6))

    with pytest.raises(ValueError, match="base_max_vel"):
        robots._check(spec, WALKER / "robot.yaml")


def test_a_command_goes_out_every_cycle_until_it_expires_then_three_zeros_then_nothing(clock):
    log = []
    reader = a_reader(log)
    reader.drive(0.2, 0.0, 0.0, 1.0)
    for t in (100.0, 100.5, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4):
        clock.now = t
        reader.read_once()

    assert log == [FORWARD, FORWARD, FORWARD, STILL, STILL, STILL]


def test_the_reader_polls_fast_while_a_command_runs_even_without_a_base_pose(clock):
    reader = a_reader([])
    reader.drive(0.2, 0.0, 0.0, 1.0)
    live = reader.read_once()
    clock.now = 101.0
    idle = reader.read_once()

    assert live == pytest.approx(1 / 30)
    assert idle == pytest.approx(0.1)


def test_a_new_command_replaces_the_running_one(clock):
    log = []
    reader = a_reader(log)
    reader.drive(0.2, 0.0, 0.0, 2.0)
    reader.read_once()
    clock.now = 101.0
    reader.drive(0.0, 0.0, -0.5, 2.0)
    clock.now = 102.5                                   # past the first command's end, within the second's
    reader.read_once()

    assert log == [FORWARD, (0.0, 0.0, -0.5)]


def test_stopping_the_reader_of_a_driven_robot_sends_a_zero_before_disconnecting(clock):
    log = []
    reader = a_reader(log)
    reader.drive(0.2, 0.0, 0.0, 60.0)
    reader.read_once()
    reader.stop()

    assert log == [FORWARD, STILL, "disconnect"]


def test_speeds_are_clamped_to_base_max_vel(walker, drivebus):
    with a_live_walker(walker) as twin:
        twin.drive_other("walker", 5.0, -5.0, 5.0, 60.0)
        wait_for(lambda: drivebus.BUSES[-1].sent != [])
        first = drivebus.BUSES[-1].sent[0]

    assert first == (0.3, -0.2, 0.6)


def test_stop_other_zeroes_at_once(walker, drivebus):
    with a_live_walker(walker) as twin:
        twin.drive_other("walker", 0.2, 0.0, 0.0, 60.0)
        wait_for(lambda: FORWARD in drivebus.BUSES[-1].sent)
        twin.stop_other("walker")
        wait_for(lambda: drivebus.BUSES[-1].sent[-1] == STILL, timeout=0.5)     # long before the 60 s


def test_the_emergency_stop_stops_the_robot_with_no_arm_connected(walker, drivebus):
    with a_live_walker(walker) as twin:
        twin.drive_other("walker", 0.2, 0.0, 0.0, 60.0)
        wait_for(lambda: FORWARD in drivebus.BUSES[-1].sent)
        twin.estop()
        wait_for(lambda: drivebus.BUSES[-1].sent[-1] == STILL, timeout=0.5)


def test_driving_is_refused_while_the_emergency_stop_is_latched(walker, drivebus):
    with a_live_walker(walker, clock=VirtualClock(t0=0.0)) as twin:
        twin.connect("sim", threaded=False)
        twin.estop()

        with pytest.raises(ValueError, match="STOP awaryjny jest zatrzasniety"):
            twin.drive_other("walker", 0.2, 0.0, 0.0, 2.0)


def test_base_drive_round_trips_through_the_workspace_file(walker, tmp_path):
    path = station(walker).save(tmp_path / "twin.json")

    assert Workspace.load(path).other_robot("walker").base_drive is True


def test_base_drive_round_trips_through_a_stand_folder(walker, tmp_path):
    ws = station(walker)
    ws.save(tmp_path / "ws" / "twin.json")
    st.save(ws, tmp_path / "stand")

    assert st.load(tmp_path / "stand").workspace().other_robot("walker").base_drive is True


def test_a_command_never_runs_longer_than_the_twins_cap(walker, drivebus, clock):
    with a_live_walker(walker) as twin:
        twin.drive_other("walker", 0.1, 0.0, 0.0, 600.0)
        until = twin._readers["walker"]._until

    assert until == clock.now + twin.drive_max_s


def test_a_speed_that_is_not_a_number_is_refused(walker, drivebus):
    with a_live_walker(walker) as twin, pytest.raises(ValueError, match="skonczonymi liczbami"):
        twin.drive_other("walker", float("nan"), 0.0, 0.0, 2.0)
