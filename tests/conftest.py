"""Wspolne narzedzia testow: syntetyczna dlon i skrocona konfiguracja."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot_mp.config import AppConfig, load_config
from lerobot_mp.twin.robots import REGISTRY, RobotSpec
from lerobot_mp.vision.pose import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_POSE_LANDMARKS,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseSample,
)
from lerobot_mp.vision.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    HandSample,
)


def make_hand(
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.12,
    roll_deg: float = 0.0,
    pinch: float = 0.6,
    curl: float = 0.85,
    handedness: str = "Right",
    depth: float = 0.0,
    aspect: float = 16.0 / 9.0,
) -> HandSample:
    """Buduje syntetyczna dlon o zadanych cechach.

    Dlon jest plaska: nadgarstek w srodku, palce "do gory" (w kierunku
    malejacego Y obrazu), obrocone o `roll_deg`. `curl` to stosunek dlugosci
    palca do dlugosci dloni - 0.85 to dlon otwarta, 0.35 to piesc.

    Geometria jest budowana w jednostkach FIZYCZNYCH (wysokosc kadru = 1),
    a dopiero na koniec sciskana w osi X o proporcje kadru - tak samo, jak
    robi to kamera. Bez tego kroku obrocona dlon zmienialaby rozmiar, a testy
    mierzylyby wade wlasnej atrapy zamiast kodu.

    Lewa dlon jest lustrzanym odbiciem prawej - takze to musi byc w atrapie,
    bo na tym opiera sie normalizacja obrotu dla obu rak.
    """
    points = np.zeros((21, 3), dtype=np.float32)
    angle = math.radians(roll_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    mirror = -1.0 if handedness == "Left" else 1.0

    def place(index: int, along: float, across: float, z: float = 0.0) -> None:
        """`along` - w strone palcow, `across` - w poprzek dloni."""
        across = across * mirror
        x = across * cos_a - (-along) * sin_a
        y = across * sin_a + (-along) * cos_a
        points[index] = (center[0] + x * scale / aspect, center[1] + y * scale, z)

    place(WRIST, 0.0, 0.0)
    place(MIDDLE_MCP, 1.0, 0.0)              # dlugosc dloni = `scale`
    place(INDEX_MCP, 0.95, 0.30)
    place(PINKY_MCP, 0.85, -0.42)
    place(RING_MCP, 0.92, -0.16)

    # Palce sterujace sprzeglem: srodkowy, serdeczny, maly.
    for mcp, tip, across in ((MIDDLE_MCP, 12, 0.0), (RING_MCP, 16, -0.16), (PINKY_MCP, 20, -0.42)):
        base_along = {MIDDLE_MCP: 1.0, RING_MCP: 0.92, PINKY_MCP: 0.85}[mcp]
        place(tip, base_along + curl, across)
    for pip, mcp in ((10, MIDDLE_MCP), (14, RING_MCP), (18, PINKY_MCP)):
        points[pip] = (points[mcp] + points[{10: 12, 14: 16, 18: 20}[pip]]) / 2.0

    # Kciuk i palec wskazujacy: odleglosc miedzy koncowkami = `pinch` * scale.
    place(INDEX_TIP, 1.9, 0.30)
    place(THUMB_TIP, 1.9, 0.30 - pinch)
    for i in (1, 2, 3):
        points[i] = (points[WRIST] + points[THUMB_TIP]) / 2.0
    for i in (6, 7):
        points[i] = (points[INDEX_MCP] + points[INDEX_TIP]) / 2.0
    points[11] = points[10]
    points[15] = points[14]
    points[19] = points[18]

    points[:, 2] += depth
    # `world_landmarks` sa metryczne i bez sciskania osi X.
    world = points.copy()
    world[:, 0] *= aspect
    world[:, :2] -= world[WRIST, :2]
    return HandSample(points, world, handedness, 0.95)


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch):
    """Testy licza na wartosciach domyslnych - plik z LEROBOT_MP_CONFIG w powloce by je zmienil."""
    # setenv zapamietuje stan sprzed testu, wiec to, co ustawi `lerobot-twin --config`
    # w os.environ, zniknie po tescie (samo delenv nieistniejacej zmiennej niczego nie zapamietuje).
    monkeypatch.setenv("LEROBOT_MP_CONFIG", "")
    monkeypatch.delenv("LEROBOT_MP_CONFIG")


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def frame_size() -> tuple[int, int]:
    return (1280, 720)


@pytest.fixture(params=sorted(REGISTRY))
def spec(request) -> RobotSpec:
    """Kazde ramie z rejestru - testy geometrii i fizyki, ktore maja trzymac dla wszystkich."""
    pytest.importorskip("mujoco")
    return REGISTRY[request.param]


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


def make_pose(
    elevation_deg: float = -90.0,
    azimuth_deg: float = 0.0,
    elbow_deg: float = 0.0,
    side: str = "Right",
    visibility: float = 1.0,
    upper_arm: float = 0.30,
    forearm: float = 0.26,
) -> PoseSample:
    """Buduje syntetyczna sylwetke o zadanych katach ramienia.

    Operator stoi twarza do kamery. Uklad MediaPipe: X w prawo obrazu, Y w dol,
    Z w glab sceny - wiec strona LEWA operatora jest po prawej stronie obrazu,
    a bark jest nad biodrem, czyli ma MNIEJSZE Y.

    Katy sa zadane w tym samym ukladzie tulowia, ktorego uzywa
    `extract_arm_features`, wiec test moze porownac wejscie z wyjsciem.
    """
    world = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    world[LEFT_HIP] = (0.10, 0.0, 0.0)
    world[RIGHT_HIP] = (-0.10, 0.0, 0.0)
    world[LEFT_SHOULDER] = (0.20, -0.50, 0.0)
    world[RIGHT_SHOULDER] = (-0.20, -0.50, 0.0)
    world[NOSE] = (0.0, -0.70, -0.10)

    up = np.array([0.0, -1.0, 0.0])
    right = np.array([-1.0, 0.0, 0.0])       # w prawo OPERATORA
    forward = np.cross(up, right)            # w strone kamery
    outward = right if side == "Right" else -right

    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    upper_dir = (
        math.sin(elevation) * up
        + math.cos(elevation) * math.cos(azimuth) * outward
        + math.cos(elevation) * math.sin(azimuth) * forward
    )

    # Przedramie zgiete o `elbow_deg` - obracamy je wokol osi prostopadlej
    # do ramienia, wybranej tak, zeby zgiecie szlo "do przodu" jak u czlowieka.
    axis = np.cross(upper_dir, forward)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(upper_dir, up)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(elbow_deg)
    fore_dir = (
        upper_dir * math.cos(angle)
        + np.cross(axis, upper_dir) * math.sin(angle)
        + axis * np.dot(axis, upper_dir) * (1 - math.cos(angle))
    )

    shoulder_i = RIGHT_SHOULDER if side == "Right" else LEFT_SHOULDER
    elbow_i = RIGHT_ELBOW if side == "Right" else LEFT_ELBOW
    wrist_i = RIGHT_WRIST if side == "Right" else LEFT_WRIST
    world[elbow_i] = world[shoulder_i] + upper_dir * upper_arm
    world[wrist_i] = world[elbow_i] + fore_dir * forearm

    # Punkty obrazu sa tu tylko po to, zeby HUD mial co rysowac.
    image = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    image[:, 0] = 0.5 + world[:, 0] * 0.4
    image[:, 1] = 0.5 + world[:, 1] * 0.4

    vis = np.full(NUM_POSE_LANDMARKS, visibility, dtype=np.float32)
    return PoseSample(image, world, vis)
