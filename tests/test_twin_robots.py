"""Opis ramienia i kinematyka blizniaka.

Najwazniejszy test jest pierwszy: MJCF z MuJoCo Menagerie ma sie zgadzac
z niezaleznym modelem Articulusa co do polozenia kazdego czlonu w dowolnej
pozie. Articulus jest tym, na czym stoi podglad 3D aplikacji i przeliczenia
backendu `feetech`, wiec zgodnosc znaczy, ze katy z prawdziwego ramienia
wpisane w symulacje daja w niej te sama poze. Bez tego blizniak klamie.
Dalej: chwytak blizniaka ma ten sam kat co serwo.
"""

from __future__ import annotations

import numpy as np
import pytest

# Przed importem blizniaka: kinematics importuje mujoco, a bez niego (instalacja
# bez [twin]) caly plik ma sie pominac, a nie przerwac zbieranie testow.
mujoco = pytest.importorskip("mujoco")

from lerobot_mp.twin.kinematics import RobotKinematics, inverse  # noqa: E402
from lerobot_mp.twin.robots import SO101  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics(SO101)


def random_joints(kin: RobotKinematics, rng: np.random.Generator) -> dict[str, float]:
    spec = kin.spec
    joints = {
        name: float(rng.uniform(np.degrees(lo), np.degrees(hi)))
        for name, lo, hi in zip(spec.joints, kin.lo, kin.hi)
    }
    if spec.gripper:
        joints[spec.gripper] = float(rng.uniform(0.0, 100.0))
    return joints


def test_mjcf_matches_the_articulus_reference_in_every_pose(kin):
    preview = pytest.importorskip("lerobot_mp.preview.model")
    art = preview.load_model()
    if art is None:
        pytest.skip("brak modelu podgladu 3D w repozytorium")

    base = art.link_names.index("base")
    links = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        joints = random_joints(kin, rng)
        Ta = art.link_transforms(art.from_lerobot(joints))
        for name in links:
            ref = (inverse(Ta[base]) @ Ta[art.link_names.index(name)])[:3, 3]
            worst = max(worst, float(np.linalg.norm(ref - kin.body(name, joints)[:3, 3])))
    assert worst < 1e-5, f"czlony rozjezdzaja sie o {worst * 1000:.3f} mm"


def test_gripper_angle_is_the_servo_angle_of_the_feetech_backend(kin):
    """0..100 chwytaka ma byc w blizniaku TYM SAMYM katem szczeki, co na serwie.

    Wczesniej 0..100 szlo liniowo na caly zakres MJCF (-10..100 st.), a backend
    `feetech` rozciaga je na tiki 1986..2670 (60 st. skoku serwa) - jednostka
    chwytaka byla w symulacji 1,8 raza wieksza niz na ramieniu.
    """
    from lerobot_mp.config import load_config
    from lerobot_mp.robot.feetech import FeetechArm

    cfg = load_config(overrides={"robot": {"backend": "feetech", "port": "COM_TEST"}})
    arm = FeetechArm(cfg, bus=object())                   # tylko przeliczenia, bez portu
    rc = cfg.robot
    k = SO101.joints.index("gripper")
    for ticks in (rc.gripper_closed_ticks, 2100, 2300, rc.gripper_open_ticks):
        servo_deg = (ticks - rc.center_ticks) * 360.0 / 4096
        units = arm._to_units("gripper", ticks)
        assert np.degrees(kin.to_q({"gripper": units})[k]) == pytest.approx(servo_deg, abs=1e-6)
        assert kin.from_q(kin.to_q({"gripper": units}))["gripper"] == pytest.approx(units, abs=1e-6)
    # Poza zakresem stawu MJCF kat jest przyciety, a nie zawiniety.
    assert kin.to_q({"gripper": 1000.0})[k] == pytest.approx(kin.hi[k])
    assert kin.to_q({"gripper": -1000.0})[k] == pytest.approx(kin.lo[k])
