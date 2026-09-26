# SO-101 — etap moderate

Eksperymentalny checkpoint i wyniki symulacji. Wymaga kodu z PR #2 oraz adaptera percepcji typu, wymiarów i pozy obiektu przed wdrożeniem na fizycznym ramieniu.

Źródło checkpointu: `block-003-moderate`. Ewaluacja kompletna: `True`.

| Profil / obiekt | Sukcesy | Średnie takty | Jerk poleceń RMS [rad/s³] |
|---|---:|---:|---:|
| nominal/cube | 5/5 | 39.0 | 43.80 |
| nominal/flat | 4/5 | 89.8 | 59.84 |
| nominal/triangle | 5/5 | 38.2 | 39.35 |
| nominal/cylinder | 5/5 | 39.2 | 43.77 |
| nominal/pliers | 5/5 | 74.0 | 56.39 |
| nominal/screw | 2/5 | 124.6 | 56.79 |
| nominal/nut | 5/5 | 60.8 | 51.65 |
| nominal/washer | 3/5 | 105.8 | 56.94 |
| nominal/l_bracket | 5/5 | 59.2 | 54.09 |
| nominal/t_bracket | 5/5 | 57.4 | 52.44 |
| nominal/capsule | 5/5 | 56.2 | 52.83 |
| nominal/wedge | 3/5 | 109.6 | 55.19 |
| moderate/cube | 5/5 | 33.8 | 62.15 |
| moderate/flat | 4/5 | 113.4 | 79.14 |
| moderate/triangle | 5/5 | 76.0 | 79.21 |
| moderate/cylinder | 5/5 | 33.4 | 54.22 |
| moderate/pliers | 5/5 | 59.4 | 74.15 |
| moderate/screw | 1/5 | 194.4 | 98.77 |
| moderate/nut | 5/5 | 41.4 | 64.39 |
| moderate/washer | 4/5 | 115.4 | 80.97 |
| moderate/l_bracket | 4/5 | 73.2 | 64.75 |
| moderate/t_bracket | 4/5 | 96.8 | 73.40 |
| moderate/capsule | 4/5 | 59.6 | 70.43 |
| moderate/wedge | 5/5 | 135.4 | 82.08 |

Modele geometryczne są parametrycznymi przybliżeniami. Typ i rozmiar obiektu są wejściami, a nie wynikiem wyuczonego detektora obrazu. Wyniki walidacji służą do wyboru checkpointu; tylko raport final używa osobnego zbioru testowego.
