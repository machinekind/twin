# SO-101 — etap stress

Eksperymentalny checkpoint i wyniki symulacji. Wymaga kodu z PR #2 oraz adaptera percepcji typu, wymiarów i pozy obiektu przed wdrożeniem na fizycznym ramieniu.

Źródło checkpointu: `block-004-stress`. Ewaluacja kompletna: `True`.

| Profil / obiekt | Sukcesy | Średnie takty | Jerk poleceń RMS [rad/s³] |
|---|---:|---:|---:|
| nominal/cube | 5/5 | 44.2 | 54.62 |
| nominal/flat | 1/5 | 175.2 | 68.27 |
| nominal/triangle | 5/5 | 42.0 | 39.45 |
| nominal/cylinder | 5/5 | 35.6 | 37.34 |
| nominal/pliers | 5/5 | 69.4 | 53.63 |
| nominal/screw | 2/5 | 142.2 | 59.90 |
| nominal/nut | 5/5 | 44.4 | 50.83 |
| nominal/washer | 3/5 | 154.4 | 63.62 |
| nominal/l_bracket | 5/5 | 68.2 | 61.62 |
| nominal/t_bracket | 5/5 | 71.8 | 51.45 |
| nominal/capsule | 5/5 | 43.6 | 44.65 |
| nominal/wedge | 5/5 | 84.6 | 56.86 |
| moderate/cube | 5/5 | 33.4 | 58.54 |
| moderate/flat | 4/5 | 108.0 | 77.71 |
| moderate/triangle | 5/5 | 39.2 | 67.37 |
| moderate/cylinder | 5/5 | 32.2 | 51.65 |
| moderate/pliers | 5/5 | 59.8 | 65.49 |
| moderate/screw | 1/5 | 167.0 | 87.50 |
| moderate/nut | 5/5 | 46.4 | 63.05 |
| moderate/washer | 1/5 | 186.8 | 78.81 |
| moderate/l_bracket | 5/5 | 45.2 | 60.95 |
| moderate/t_bracket | 4/5 | 71.4 | 69.17 |
| moderate/capsule | 3/5 | 84.0 | 66.72 |
| moderate/wedge | 5/5 | 83.6 | 69.83 |

Modele geometryczne są parametrycznymi przybliżeniami. Typ i rozmiar obiektu są wejściami, a nie wynikiem wyuczonego detektora obrazu. Wyniki walidacji służą do wyboru checkpointu; tylko raport final używa osobnego zbioru testowego.
