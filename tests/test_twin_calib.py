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


# ----------------------------------------------------------- zestawy polozen
def test_step2_results_are_kept_as_presets_and_newest_loads_by_default(tmp_path):
    """Kazdy zapis kroku 2 to nowy plik; `load` bierze najnowszy, wybrany starszy zostaje."""
    from lerobot_mp.twin.workspace import CameraRecord, Workspace

    def pose(x):
        return [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]

    ws = Workspace(path=tmp_path / "twin.json")
    ws.add_camera(CameraRecord("front", T_cam2base=pose(0.1),
                               calibration={"trusted": True, "time": "2026-09-26T10:00:00"}))
    ws.save()                                           # stare stanowisko: jego poza staje sie pierwszym zestawem
    assert ws.presets() == ["20260926-100000"] and ws.calib_preset == "20260926-100000"

    ws.camera("front").T_cam2base = pose(0.2)
    ws.camera("front").calibration = {"trusted": True, "time": "2026-09-26T11:00:00"}
    ws.save_preset()
    ws.save_preset()                                    # ten sam wynik drugi raz - bez nowego pliku
    ws.save()
    assert ws.presets() == ["20260926-110000", "20260926-100000"]
    assert (tmp_path / "calib" / "20260926-100000.json").exists(), "starszy zapis nie znika"

    again = Workspace.load(tmp_path / "twin.json")
    assert again.calib_preset == "20260926-110000" and again.camera("front").T_cam2base[0][3] == 0.2
    assert again.load_preset("20260926-100000") == ["front"]
    assert again.camera("front").T_cam2base[0][3] == 0.1
    again.save()
    assert Workspace.load(tmp_path / "twin.json").calib_preset == "20260926-100000", "wybor starszego zostaje"

    (tmp_path / "twin.json").write_text(
        (tmp_path / "twin.json").read_text().replace('"calib_preset": "20260926-100000"', '"calib_preset": null'))
    assert Workspace.load(tmp_path / "twin.json").camera("front").T_cam2base[0][3] == 0.2, "bez wyboru - najnowszy"
