"""Kalibracja kamer z karty w chwytaku: pelna sesja.

Cala sesja - ramie, fala, render, detekcja, dopasowanie, bramki - w symulacji,
wzgledem prawdziwych poz kamer i karty, ktorych sesja nie widzi.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.calib.simulate import run  # noqa: E402


# ----------------------------------------------------------------- sesja
@pytest.mark.render
def test_full_session_in_simulation_calibrates_the_camera():
    """Cala sesja: fala, render, detekcja, dopasowanie i werdykt - wzgledem prawdy."""
    res = run(seed=0, n_cameras=1)
    cam = next(iter(res.trusted))
    assert res.trusted[cam], res.reasons[cam]
    # Zaufane sesje na ziarnach 0..5 daja 0,3-1,7 mm i 0,006-0,19 st. (srednio ok. 0,5 mm);
    # ziarno 0 jest najgorsze z nich. Prog 1 mm przy jednym ziarnie lamal sie od samej zmiany
    # drogi fali (inny kat szczeki przy chwytaku 0) - stary uklad tez mial 1,15-1,17 mm na
    # ziarnach 8 i 12. Do chwytania 25-milimetrowej kostki 2 mm to wciaz zapas.
    assert res.err_mm[cam] < 2.0
    assert res.err_deg[cam] < 0.25
    assert res.poses < 40, "fala ma sie zmiescic w kilkudziesieciu pozach, nie w maksimum"
