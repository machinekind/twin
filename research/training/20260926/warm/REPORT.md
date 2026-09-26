# SO-101 — etap warm

Eksperymentalny checkpoint i wyniki symulacji. Wymaga kodu z PR #2 oraz adaptera percepcji typu, wymiarów i pozy obiektu przed wdrożeniem na fizycznym ramieniu.

Źródło checkpointu: `block-002-warm`. Ewaluacja kompletna: `True`.

| Profil / obiekt | Sukcesy | Średnie takty | Jerk poleceń RMS [rad/s³] |
|---|---:|---:|---:|
| nominal/cube | 5/5 | 43.6 | 50.24 |
| nominal/flat | 2/5 | 161.8 | 71.90 |
| nominal/triangle | 5/5 | 39.0 | 43.31 |
| nominal/cylinder | 5/5 | 36.4 | 43.53 |
| nominal/pliers | 5/5 | 94.2 | 59.65 |
| nominal/screw | 1/5 | 183.2 | 68.50 |
| nominal/nut | 5/5 | 38.2 | 42.04 |
| nominal/washer | 4/5 | 107.6 | 65.98 |
| nominal/l_bracket | 4/5 | 91.2 | 62.90 |
| nominal/t_bracket | 5/5 | 42.4 | 48.56 |
| nominal/capsule | 5/5 | 54.8 | 53.53 |
| nominal/wedge | 3/5 | 137.8 | 67.93 |
| moderate/cube | 5/5 | 34.4 | 60.25 |
| moderate/flat | 4/5 | 152.0 | 91.39 |
| moderate/triangle | 5/5 | 51.4 | 82.30 |
| moderate/cylinder | 5/5 | 34.2 | 61.13 |
| moderate/pliers | 5/5 | 68.4 | 80.29 |
| moderate/screw | 2/5 | 170.8 | 102.26 |
| moderate/nut | 5/5 | 41.4 | 71.85 |
| moderate/washer | 2/5 | 151.2 | 91.16 |
| moderate/l_bracket | 5/5 | 36.2 | 68.09 |
| moderate/t_bracket | 4/5 | 109.8 | 82.88 |
| moderate/capsule | 4/5 | 50.8 | 72.99 |
| moderate/wedge | 4/5 | 88.8 | 90.86 |

Modele geometryczne są parametrycznymi przybliżeniami. Typ i rozmiar obiektu są wejściami, a nie wynikiem wyuczonego detektora obrazu. Wyniki walidacji służą do wyboru checkpointu; tylko raport final używa osobnego zbioru testowego.
