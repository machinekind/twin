"""Shared test helpers: no config file from the shell, and a toy arm."""

from __future__ import annotations

import pytest

from lerobot_mp.twin.robots import RobotSpec


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch):
    """Testy licza na wartosciach domyslnych - plik z LEROBOT_MP_CONFIG w powloce by je zmienil."""
    # setenv zapamietuje stan sprzed testu, wiec to, co ustawi `lerobot-twin --config`
    # w os.environ, zniknie po tescie (samo delenv nieistniejacej zmiennej niczego nie zapamietuje).
    monkeypatch.setenv("LEROBOT_MP_CONFIG", "")
    monkeypatch.delenv("LEROBOT_MP_CONFIG")


#: Zabawkowe ramie: dwa stawy obrotowe, chwytak na suwaku (0..2 cm) i druga szczeka
#: sprzezona z nim rownoscia (`gripper_m = -gripper`), jak druga szczeka A1X.
TOY_MJCF = """<mujoco model="toy">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default>
    <joint damping="0.5" armature="0.005"/>
    <position kp="20" kv="1"/>
  </default>
  <worldbody>
    <body name="base">
      <geom name="base" type="cylinder" size="0.04 0.02" pos="0 0 0.02" mass="1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.015" mass="0.2"/>
        <body name="link2" pos="0 0 0.1">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.012" mass="0.1"/>
          <body name="hand" pos="0.12 0 0">
            <geom type="box" size="0.01 0.03 0.01" mass="0.05"/>
            <site name="tcp" pos="0.04 0 0"/>
            <body name="jaw1" pos="0.02 0 0">
              <joint name="gripper" type="slide" axis="0 1 0" range="0 0.02"/>
              <geom name="tip1" type="box" size="0.02 0.003 0.008" pos="0.02 0.005 0" mass="0.01"/>
            </body>
            <body name="jaw2" pos="0.02 0 0">
              <joint name="gripper_m" type="slide" axis="0 1 0" range="-0.02 0"/>
              <geom name="tip2" type="box" size="0.02 0.003 0.008" pos="0.02 -0.005 0" mass="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="gripper_m" joint2="gripper" polycoef="0 -1 0 0 0"/>
  </equality>
  <actuator>
    <position name="j1" joint="j1"/>
    <position name="j2" joint="j2"/>
    <position name="gripper" joint="gripper" kp="50" kv="2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="session")
def toy_spec(tmp_path_factory) -> RobotSpec:
    """Opis zabawkowego ramienia (`TOY_MJCF`) - dowod, ze blizniak nie zaklada SO-101."""
    path = tmp_path_factory.mktemp("toy") / "toy.xml"
    path.write_text(TOY_MJCF, encoding="utf-8")
    return RobotSpec(
        name="toy",
        title="Zabawka",
        mjcf=str(path),
        joints=("j1", "j2", "gripper"),
        gripper="gripper",
        tcp_site="tcp",
        hand_body="hand",
        base_body="base",
        home={"j1": 0.0, "j2": 20.0, "gripper": 50.0},
        tcp_approach=(1.0, 0.0, 0.0),
        tcp_closing=(0.0, 1.0, 0.0),
        fingertips=("tip1", "tip2"),
        wave_ranges={"j1": (-45.0, 45.0), "j2": (-30.0, 30.0)},
        jaw_bodies=("jaw1", "jaw2"),
        limits={"j1": (-90.0, 90.0), "j2": (-60.0, 60.0), "gripper": (0.0, 100.0)},
        max_vel={"j1": 90.0, "j2": 90.0, "gripper": 300.0},
        gripper_range_q=(0.0, 0.02),
        mirror_joints={"gripper_m": ("gripper", -1.0)},
    )
