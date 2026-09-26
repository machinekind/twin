# SO-101 — etap final

Eksperymentalny checkpoint i wyniki symulacji. Wymaga kodu z PR #2 oraz adaptera percepcji typu, wymiarów i pozy obiektu przed wdrożeniem na fizycznym ramieniu.

Źródło checkpointu: `selected`. Ewaluacja kompletna: `True`.

| Profil / obiekt | Sukcesy | Średnie takty | Jerk poleceń RMS [rad/s³] |
|---|---:|---:|---:|
| nominal/cube | 50/50 | 36.3 | 46.30 |
| nominal/flat | 40/50 | 108.2 | 69.83 |
| nominal/triangle | 50/50 | 40.1 | 49.63 |
| nominal/cylinder | 49/50 | 37.0 | 40.02 |
| nominal/pliers | 46/50 | 83.3 | 59.31 |
| nominal/screw | 37/50 | 97.4 | 57.94 |
| nominal/nut | 50/50 | 42.2 | 49.10 |
| nominal/washer | 47/50 | 71.5 | 53.86 |
| nominal/l_bracket | 40/50 | 92.4 | 56.64 |
| nominal/t_bracket | 49/50 | 62.5 | 56.86 |
| nominal/capsule | 41/50 | 57.0 | 48.69 |
| nominal/wedge | 41/50 | 94.6 | 64.34 |
| moderate/cube | 50/50 | 45.7 | 65.91 |
| moderate/flat | 45/50 | 98.4 | 80.51 |
| moderate/triangle | 49/50 | 50.3 | 65.80 |
| moderate/cylinder | 49/50 | 44.0 | 58.27 |
| moderate/pliers | 47/50 | 77.7 | 74.37 |
| moderate/screw | 42/50 | 97.7 | 76.73 |
| moderate/nut | 50/50 | 44.6 | 62.43 |
| moderate/washer | 43/50 | 91.2 | 71.23 |
| moderate/l_bracket | 48/50 | 66.7 | 69.37 |
| moderate/t_bracket | 46/50 | 74.9 | 73.00 |
| moderate/capsule | 38/50 | 56.6 | 68.80 |
| moderate/wedge | 44/50 | 82.0 | 70.34 |
| stress/cube | 49/50 | 54.1 | 92.58 |
| stress/flat | 32/50 | 131.9 | 102.48 |
| stress/triangle | 47/50 | 72.7 | 91.35 |
| stress/cylinder | 44/50 | 52.4 | 92.66 |
| stress/pliers | 47/50 | 87.4 | 94.39 |
| stress/screw | 31/50 | 127.6 | 100.11 |
| stress/nut | 47/50 | 64.2 | 93.38 |
| stress/washer | 32/50 | 115.7 | 95.70 |
| stress/l_bracket | 42/50 | 95.2 | 102.13 |
| stress/t_bracket | 45/50 | 91.0 | 97.92 |
| stress/capsule | 37/50 | 70.0 | 99.91 |
| stress/wedge | 43/50 | 99.5 | 99.42 |

Modele geometryczne są parametrycznymi przybliżeniami. Typ i rozmiar obiektu są wejściami, a nie wynikiem wyuczonego detektora obrazu. Wyniki walidacji służą do wyboru checkpointu; tylko raport final używa osobnego zbioru testowego.
