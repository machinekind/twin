# Eksperyment odporności lift

Skrypt `scripts/train_robust_lift.py` doucza dostarczone `lift-v3` i porównuje
checkpointy z bazą w zwykłym MuJoCo na CPU. Trening korzysta z MuJoCo Warp
na GPU. Zadanie, definicja sukcesu oraz funkcja nagrody pozostają takie jak
w pliku bazowej polityki. Sukces w raporcie wymaga zakończenia epizodu serią
10 kolejnych udanych taktów, a nie tylko chwilowego podniesienia kostki.

## Uruchomienie lokalne

Na tym komputerze Python treningowy znajduje się w
`D:\digital-twin-training\.venv\Scripts\python.exe`. Środowisko ma własne
wersje torch 2.11.0+cu126 i MuJoCo/MuJoCo Warp 3.14.0, a pozostałe dostępne
pakiety może dziedziczyć z Pythona systemowego (`--system-site-packages`).

Z katalogu repozytorium:

```powershell
D:\digital-twin-training\.venv\Scripts\python.exe -u scripts/train_robust_lift.py
```

Ponowny eksperyment należy zapisać do osobnego katalogu przez `--out`.
Samo zmierzenie bazy: `--baseline-only`. Liczby iteracji i prób można ustawić
przez `--iters`, `--validation-episodes` i `--test-episodes`.

## Protokół

1. Walidacja bazy: 100 epizodów na profil, seedy 20000–20099.
2. Douczanie: 150 iteracji, 4096 światów, horyzont 24, seed 260926;
   14 745 600 przejść. Checkpoint co 25 iteracji.
3. Walidacja każdego checkpointu na tych samych próbach co baza.
4. Wybór modelu wyłącznie na walidacji.
5. Test bazy i wybranego modelu: 500 epizodów na profil, seedy 50000–50499.
   Wyniki tego testu nie służą do ponownego wyboru checkpointu.

Profile to nominalny, standardowa randomizacja repo oraz `stress`:
1,7 raza szerszy rozrzut dynamiki wokół modelu Menagerie, opóźnienie akcji
0–2 takty, percepcji 0–4 takty, okres aktualizacji 1–4 takty, szum pozycji
kostki 4 mm i kątów 0,5°. Takty polityki trwają 50 ms. Dokładne zakresy
i wersje zapisuje `manifest.json`. To syntetyczny test odporności;
nie jest pomiarem konkretnego ramienia ani testem obrazu z kamer.

Do wyboru dopuszczane są checkpointy bez spadku skuteczności nominalnej
i standardowej względem bazy na walidacji. Dodatkowe warunki na tych dwóch
profilach: końcowa wysokość poniżej 160 mm, średnia poniżej 135 mm,
maksymalny zakres ruchu stawu po pierwszym sukcesie poniżej 55° oraz
liczba taktów przy limitach po chwycie nie większa niż u bazy (z tolerancją
do 0,4 taktu na epizod). Wśród dopuszczonych modeli wygrywa skuteczność
w `stress`, a remis rozstrzyga średnia długość epizodu. Baza też jest kandydatem.

## Artefakty

W `workspace/experiments/lift-robust/` znajdują się:

- `manifest.json`: konfiguracja, commit źródłowy i SHA-256 bazy;
- `training/progress.json`: postęp i historia treningu;
- `checkpoint-*/policy.pt`: zachowane polityki i krytyki;
- `validation-*.json`, `test-*.json`: agregaty i wynik każdego epizodu;
- `selection.json`: wybór na walidacji;
- `summary.json`: końcowe porównanie na niewidzianych seedach;
- `selected/policy.pt`: model wybrany na walidacji z wynikami testowymi.

Raport podaje przedział Wilsona 95% dla skuteczności. Nawet 500/500 sukcesów
nie dowodzi stuprocentowej niezawodności. Nie nazywamy modelu lepszym,
jeżeli końcowy test tego nie potwierdza.
