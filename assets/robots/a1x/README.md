# Galaxea A1X (MJCF)

Model ramienia Galaxea A1X dla MuJoCo, wygenerowany skryptem
[`scripts/import_a1x_model.py`](../../../scripts/import_a1x_model.py) z URDF producenta.

## Zrodlo

* `a1x.urdf` i siatki `assets/*.STL` (10 plikow, 2 555 790 B) - URDF i siatki producenta
  (Galaxea), skopiowane bez zmian z pakietu `ros2_ws/src/galaxea_a1xy_description`
  repozytorium galaxeo-manipulators. Pakiet podaje jako zrodlo "userguide-galaxea/URDF",
  a w `package.xml` licencje `Proprietary`; warunki licencji producenta nie sa nigdzie
  opisane. **To nie jest Apache-2.0** - patrz `NOTICE` w korzeniu repozytorium.
* `a1x.xml` - przetworzenie, generowane przez skrypt (droga z `sim/urdf2mjcf.py`
  galaxeo-manipulators: masy, bezwladnosci, osie i zakresy stawow z URDF, serwa
  pozycyjne, proste bryly kolizyjne szczek, sprzezenie szczek rownoscia, keyframe `home`).
  Nie edytowac recznie.

## Reguly przetworzenia (wzgledem galaxeo)

1. Staw `gripper_finger_joint1` nazywa sie `gripper` (staw, aktuator, odwolanie w rownosci).
   Druga szczeka zostaje `gripper_finger_joint2`, sprzezona `q2 = -q1`. Keyframe `home` bez zmian.
2. Aktuatory nazywaja sie jak ich stawy (`arm_joint1..6`, `gripper`).
3. Skrzynki kolizyjne zagietych czubkow szczek maja nazwy `finger1_tip` / `finger2_tip`.
4. `<option integrator="implicitfast" timestep="0.002" cone="elliptic" impratio="10"/>` - krok
   z `sim/scene.xml` galaxeo, stozek i impratio jak SO-101 z Menagerie. `noslip_iterations`
   z galaxeo celowo pominiete (scena blizniaka go nie kopiuje, MuJoCo Warp go nie wspiera).
5. Site `tcp` jak z generatora: `x` - kierunek podejscia, `y` - os zamykania szczek.

Kazda regula ma test w `tests/test_twin_a1x_sim.py`, podobnie jak to, ze skrypt odtwarza
zatwierdzony `a1x.xml` bajt w bajt.

## Regeneracja

    python scripts/import_a1x_model.py                          # z zaimportowanego a1x.urdf
    python scripts/import_a1x_model.py --urdf <galaxeo>/ros2_ws/src/galaxea_a1xy_description/urdf/a1x.urdf \
        --meshes <galaxeo>/ros2_ws/src/galaxea_a1xy_description/meshes     # z nowszego URDF producenta

Sprawdzone z mujoco 3.14.0 (galaxeo budowalo model na 3.13.0).
