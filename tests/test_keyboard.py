"""Tryb `keys`: bezpieczenstwo prowadzenia koncowki chwytaka klawiszami.

Zostaly tylko testy tego, co zatrzymuje ramie: wygasanie ruchu po puszczeniu
klawisza (OpenCV nie daje zdarzenia "puszczono") oraz sprzeglo SPACJA -
rozlaczenie zatrzymuje ruch, a ponowne zalaczenie startuje z pozy zmierzonej.
"""

from __future__ import annotations

import pytest

from lerobot_mp.config import JOINT_NAMES
from lerobot_mp.control.keyboard import KeyboardPilot

DT = 1.0 / 30.0
W = ord("w")


def measured(cfg) -> dict[str, float]:
    return dict(cfg.safety.home)


def drive(pilot: KeyboardPilot, cfg, key: int | None, steps: int = 10):
    """Trzyma klawisz przez `steps` krokow petli i zwraca ostatni wynik."""
    out = None
    for _ in range(steps):
        if key is not None:
            pilot.press(key)
        out = pilot.update(DT, engaged=True, measured=measured(cfg))
    assert out is not None
    return out


def tip(pilot: KeyboardPilot, targets: dict[str, float]) -> tuple[float, float, float]:
    return pilot.kinematics.forward(
        targets["shoulder_pan"],
        targets["shoulder_lift"],
        targets["elbow_flex"],
        targets["wrist_flex"],
    )


def test_motion_stops_when_repeats_stop_arriving(cfg):
    """Bez zdarzenia "puszczono" ruch ma wygasnac sam - i potem stac."""
    pilot = KeyboardPilot(cfg)
    drive(pilot, cfg, W)

    # Ruch trwa jeszcze `hold_timeout` po ostatnim powtorzeniu - to jest cel
    # tego progu. Sprawdzamy dopiero to, co dzieje sie PO jego uplynieciu.
    coasting = int(cfg.keyboard.hold_timeout / DT) + 2
    for _ in range(coasting):
        pilot.update(DT, engaged=True, measured=measured(cfg))
    settled = tip(pilot, pilot.update(DT, engaged=True, measured=measured(cfg)).targets)

    for _ in range(30):
        out = pilot.update(DT, engaged=True, measured=measured(cfg))
    assert tip(pilot, out.targets) == pytest.approx(settled, abs=1e-9)


def test_disengaging_stops_the_arm_and_re_engaging_does_not_jump(cfg):
    pilot = KeyboardPilot(cfg)
    drive(pilot, cfg, W, steps=20)

    idle = pilot.update(DT, engaged=False, measured=measured(cfg))
    assert idle.targets is None
    assert idle.engaged is False
    assert pilot.seeded is False

    # Po ponownym zalaczeniu cel siada na pozie ZMIERZONEJ, a nie na zapamietanej.
    again = pilot.update(DT, engaged=True, measured=measured(cfg))
    for name in JOINT_NAMES:
        assert again.targets[name] == pytest.approx(cfg.safety.home[name], abs=0.5)
