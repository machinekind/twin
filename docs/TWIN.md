# Cyfrowy bliźniak stanowiska z ramieniem robota

Symulacja MuJoCo, która wie o biurku to samo, co prawdziwe stanowisko: gdzie
stoi ramię, gdzie stoją kamery, jaką mają ogniskową i jak naprawdę odpowiadają
serwa. Kamera skalibrowana na biurku trafia do symulacji **dokładnie w swojej
pozie i ze swoją macierzą K**, a strategia sterowania jedzie w symulacji i na
prawdziwym ramieniu tą samą ścieżką — przez nadzór bezpieczeństwa — i widzi
kostkę tą samą percepcją. Bliźniak nie ma własnego ramienia: ramiona i ich
backendy dają zainstalowane pakiety. Trening i polityki są w pakiecie `soarm-lab`
(repozytorium so-arm101), a wytrenowana polityka jedzie jako strategia `policy`
w zakładce Strategie.

```
  prawdziwe stanowisko                        bliźniak (MuJoCo)
 ┌──────────────────────┐   kalibracja      ┌──────────────────────────┐
 │ ramię przez backend  │ ────────────────► │ ramię z robot.yaml       │
 │ kamery USB dookoła   │  karta w dłoni,   │ kamery w tych samych     │
 │ biurko, kostka       │  ChArUco, sysid   │ pozach, z tym samym K,   │
 └──────────▲───────────┘                   │ serwa jak zmierzone      │
            │                               └────────────┬─────────────┘
            │  strategia `policy` przez nadzór           │  lerobot-twin export:
            │  (ta sama ścieżka co w symulacji)          ▼  scene.xml + station.json
            └────────────────────────────────── trening w soarm-lab
```

## Stan

| | |
|---|---|
| **Gotowe i przetestowane w symulacji** | model, kinematyka, scena z kamerami o pełnym K, kolizje, **kalibracja wielu kamer z karty w chwytaku**, **intrynsyki z ChArUco**, mapa stołu na żywo, **wykrywanie kostki z kamer**, **wykrywanie przestawionej kamery**, **identyfikacja dynamiki serw**, **strategie przez nadzór** (`StrategyRunner`), **eksport stanowiska**, **panel webowy** |
| **Do sprawdzenia na sprzęcie** | wszystko, co dotyka prawdziwego ramienia i prawdziwych kamer — procedura krok po kroku niżej |
| **Drugie ramię: Galaxea A1X** | z pakietu galaxeo, instalowanego obok bliźniaka: model, kinematyka i kalibracja kamer w symulacji; backend `a1x` po CAN-FD przetestowany na symulowanej magistrali — sekcja [Galaxea A1X](#galaxea-a1x) |
| **Rozpoznanie** | rekonstrukcja 3D przedmiotów z biurka — [RESEARCH_3D.md](RESEARCH_3D.md) |

Szczegóły, decyzje i pułapki: **[HANDOFF.md](../HANDOFF.md)**.

---

## Instalacja

Skrót dla każdej platformy jest w README, „Szybki start na nowym komputerze”.
Najważniejsze zasady:

- Wersje sprawdzone razem są w `constraints.txt` (`-c constraints.txt`).
- Kontrakty wtyczek to osobny pakiet `lerobot-mp-api` w `packages/lerobot_mp_api`
  (patrz [Jak to jest zbudowane](#jak-to-jest-zbudowane)). Nie ma go na PyPI, więc
  instaluje się go z klonu w tym samym poleceniu co bliźniaka:
  `pip install -e packages/lerobot_mp_api -e ".[…]"`; samo `pip install -e ".[…]"`
  kończy się „No matching distribution found for lerobot-mp-api”. uv bierze tę
  ścieżkę z `[tool.uv.sources]` w `pyproject.toml`.

```bash
# Windows (PowerShell); Linux i macOS: python3.12 i source .venv/bin/activate
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass   # świeży Windows blokuje Activate.ps1; cmd.exe: .venv\Scripts\activate.bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e packages/lerobot_mp_api -e ".[twin,dev]" -c constraints.txt
pip install soarm    # ramię SO-101; robot przykładowy: pip install -e examples/robots/testbot
lerobot-twin check
lerobot-twin demo    # przykładowe stanowisko w symulacji z pierwszym zainstalowanym ramieniem -> workspace/twin.json
```

- Linux bez ekranu: `export MUJOCO_GL=egl` (NVIDIA) albo `osmesa` (CPU,
  `sudo apt install libosmesa6`).
- Bliźniak nie ma własnego ramienia. Ramię SO-101 i jego backendy (`feetech`,
  `lerobot`, most COM<->TCP) to pakiet `soarm` (repozytorium so-arm101),
  instalowany obok bliźniaka.

**Pliki stanowiska.** `workspace/twin.json` (kamery, kalibracja, port, stół,
dynamika) jest lokalny, poza repozytorium. Z klonu jego domyślna ścieżka
liczy się od katalogu repozytorium, niezależnie od katalogu uruchomienia; inny
plik: `--workspace <plik>`. Bez pliku bliźniak startuje z pustym stanowiskiem
(pierwsze zainstalowane ramię, backend `sim`, bez kamer). `lerobot-twin demo`
kopiuje `examples/twin.sim.json` i wpisuje w kopię pierwsze zainstalowane ramię
(inne: `--robot <nazwa>`) — dwie kamery symulowane, których kalibracja to ich
prawdziwa poza w scenie (zaufane), więc „kostka z kamer” działa od razu.
Istniejącego stanowiska nie nadpisze (`--force`, albo `--path <plik>`). Bez
zainstalowanego ramienia `demo` kończy się błędem i podaje, co zainstalować
(`pip install soarm` albo `pip install -e examples/robots/testbot`).

**Zapisane położenia kamer.** Każdy zapis kroku 2 (Kalibracja › „Zapisz
wynik”) trafia do osobnego pliku `workspace/calib/<data-godzina>.json`
z pozą i werdyktem wszystkich kamer, które ją mają; poprzednich zapisów nic
nie nadpisuje. `twin.json` trzyma kopię roboczą wgranego zestawu
(`calib_preset`); bez wskazania panel wgrywa najnowszy. Starszy zestaw
wraca z listy „Zapisane położenia kamer” pod krokiem 2 (werdykt „zaufana”
liczy się na nowo względem obecnego obiektywu). Stanowisko zapisane przed
tą zmianą dostaje pierwszy zestaw ze swojej dotychczasowej pozy przy
pierwszym zapisie.

**Konfiguracja ramienia.** Bliźniak czyta plik konfiguracji tylko wtedy, gdy
się go wskaże: `lerobot-twin --config configs/local.yaml ui` (opcja ustawia
zmienną `LEROBOT_MP_CONFIG`). Sekcje `safety:` i `robot:` (`backend`, `port`)
czyta bliźniak; każda inna sekcja należy do backendu o tej nazwie i sprawdza ją
sam backend. Klucze sekcji opisuje pakiet ramienia (SO-101: `feetech:` i
`lerobot:` w pakiecie `soarm`); Galaxea A1X (pakiet `galaxeo`) ma sekcję `a1x:`,
patrz [Galaxea A1X](#galaxea-a1x). Opis samego ramienia (stawy, limity, poza
domowa, zakres szczęki) leży w `robot.yaml` w folderze ramienia, w pakiecie
ramienia.

## Panel

```bash
lerobot-twin ui                  # http://localhost:8080; zajęty port: --port 8765
lerobot-twin ui --host 0.0.0.0   # także z sieci — panel nie ma hasła, tylko w zaufanej sieci
lerobot-twin ui --stream-port 8766   # plus podgląd kamer na osobnym porcie (MJPEG, tylko odczyt)
```

**Podgląd kamer** (`--stream-port`): kadry, które panel i tak pobiera, po HTTP na osobnym porcie —
`/` strona ze wszystkimi kamerami, `/<kamera>.mjpg` strumień, `/<kamera>.jpg` ostatni kadr,
`/cameras.json` lista. Nic tam nie steruje ramieniem, więc słucha domyślnie w całej sieci
(`--stream-host` zmienia). Osobny proces nie może otworzyć tych samych kamer (V4L2) —
dlatego podgląd siedzi w procesie panelu.

Duży **STOP** przyklejony u góry panelu (nie odjeżdża przy przewijaniu), pod
nim żółty **Skasuj STOP** (tylko przy aktywnym STOP-ie), plakietka stanu ramienia
(ROZŁĄCZONE / ROZBROJONE / UZBROJONE / STERUJE: … / STOP AKTYWNY) i **Przewodnik**:
zawsze widoczne jedno **Teraz:** („Krok N/5 (tytuł): zakładka > **przycisk**”,
albo co właśnie jedzie i jak to przerwać; aktywny STOP z przyczyną), a pod nim
zwinięty folder **Kroki 0–5** z uwagami, których Teraz nie mówi (ostrzeżenia,
kamera bez kadru dłużej niż 3 s, przestawiona kamera; ich liczba w nazwie
folderu) i listą kroków liczoną z bieżącego stanu — ramię, kamery, K, pozy,
Sim-Real, dynamika. Na prawdziwym ramieniu dynamika zmierzona w sim nie zalicza
kroku 5. W symulacji prowadzi przez te same kroki z kamerami symulowanymi. Każda
kontrolka ma podpowiedź po najechaniu (co robi, czy rusza ramieniem, jednostki),
a każda zakładka zwiniętą notkę **Jak używać**. Logika przewodnika:
`ui/guide.py` (bez visera, `tests/test_twin_guide.py`). Każdy ruch — z suwaków, z uchwytu
w 3D, ze strategii, z fali kalibracyjnej, z identyfikacji — idzie przez
`runtime.Twin` i jego `SafetySupervisor`.

**Ramię ma zawsze jednego właściciela**: panel (UZBRÓJ: suwaki i uchwyt 3D), strategia,
fala kalibracyjna albo identyfikacja. Drugi nie ruszy, dopóki pierwszy nie
skończy — panel mówi, kto ma ramię. **STOP**, **Dom**, **Połącz** i **Rozłącz**
odbierają ramię każdemu, a ramię zostaje w **zmierzonej** pozie: nie dociska
do przeszkody, nie wraca skokiem do starego celu i nie cofa się (przy szybkim
ruchu staje między zmierzoną pozą a ostatnim rozkazem). **Chwytak, który
ściska, dalej ściska** — STOP ani Dom nie upuszczają trzymanej kostki; otwiera
się go ręcznie suwakiem. **Połącz / Rozłącz puszczają trzymany przedmiot** na
backendzie `feetech` (przy połączeniu serwa dostają cel = bieżąca pozycja) —
przed nimi odłóż kostkę albo przytrzymaj ją ręką; w `sim` chwyt przetrwa.
Każde uruchomienie na prawdziwym ramieniu wymaga świeżego potwierdzenia.
STOP albo Dom wciśnięte w chwili startu strategii, fali czy identyfikacji
wygrywają: zadanie nie weźmie ramienia pod aktywnym STOP-em i nie ruszy po
*Skasuj STOP*.

Pętla sama zatrzymuje ramię (STOP awaryjny, powód w pasku stanu), gdy:
- serwo ramienia zgłosi błąd (przeciążenie, przegrzanie, napięcie, czujnik kąta);
- staw ramienia jest ponad 25° od rozkazu dłużej niż 0,5 s — serwo, które
  zwiotczało bez bitu błędu (każdy właściciel ruchu, nie tylko strategia);
  jeśli prawdziwe serwa pod obciążeniem nie nadążą, podnieś
  `Twin.track_err_deg` / `track_err_s`;
- serwa nie odpowiadają przez 5 odczytów z rzędu, albo pętla stanie na > 0,5 s.

Krótsza przerwa łącza nie jest błędem: nadzór stoi, nic nie idzie do serw, a
po pierwszym świeżym odczycie rusza od zmierzonej pozy (po 0,9 s przerwy
największy krok celu 4,2° — wcześniej 16,7°). **Przeciążenie samego chwytaka**
(mocny chwyt) nie zatrzymuje ramienia: docisk jest zmniejszany do 8 jednostek
(~5°) ciaśniej niż zmierzona szczęka, w pasku stanu pojawia się ostrzeżenie;
STOP dopiero, gdy błąd chwytaka trwa 2 s.

Połączenie potrzebuje dwóch zgodnych odczytów (±2 jednostki) — przekłamana
ramka nie zostaje pierwszym rozkazem. Ramię stojące po włączeniu poza limitami
zostaje, gdzie jest; po UZBRÓJ wraca w zakres (najwyżej 90°/s).
Limity z konfiguracji są zawężane do limitów zapisanych w EEPROM serw.
Niefatalne zastrzeżenia (inne tiki chwytaka w backendzie, zero stawów w
kalibracji LeRobota, zmniejszony docisk) panel pokazuje jako **Uwaga** w pasku
stanu.

| zakładka | co robi |
|---|---|
| **Ramię** | połączenie: `sim` (bliźniak jest ramieniem), backend z pakietu ramienia (SO-101 z `soarm`: `feetech`, `lerobot`), `a1x` (Galaxea A1X po CAN, z pakietem galaxeo: port `can0` / `xcan` / `PCAN_USBBUS1`, przycisk *Włącz silniki A1X*) — zestawy w liście *Ramię* wpisują port; po *Połącz*: duży **UZBRÓJ / UZBROJONE**, *Uchwyt 3D* (IK), *Dom*, *Ustaw dom = tu*, prędkość, suwaki stawów z nazwami po polsku, *Rozłącz*; *Stanowisko (stół)* zwinięte na dole |
| **Kamery** | lista kamer z jedną plakietką stanu (SKALIBROWANA / DO KALIBRACJI / KALIBRACJA NIEPEWNA / PORUSZONA / BEZ OBRAZU / WYŁĄCZONA — kolory jak piramidy w 3D); wybrana kamera: podgląd, jedna linia „co dalej”, przycisk *Włączona / Wyłączona*, *Usuń kamerę* (drugie kliknięcie potwierdza); symulowana dodatkowo *Przesuwaj w 3D* (uchwyt) i *Postaw tu, skąd patrzysz w 3D*; liczby w zwiniętych *Szczegółach*; na dole *Dodaj symulowaną przed ramieniem* i *Szukaj kamer USB* (wybór i *Dodaj kamerę USB* pojawiają się po znalezieniu) |
| **Kalibracja** | zdanie „po co to”, każda kamera z plakietkami **OBIEKTYW** i **POŁOŻENIE**, linia **Teraz**. **Krok 1 — obiektyw każdej kamery** (tylko przy prawdziwej kamerze; w kodzie: intrynsyki, K): ponumerowana instrukcja, szachownica do druku, *Zacznij: pokazuj szachownicę kamerze …*, w trakcie licznik zdjęć, *Gotowe — policz obiektyw* i *Anuluj*; po zapisie przechodzi sam do następnej kamery. **Krok 2 — gdzie stoją kamery** (w kodzie: fala): ramię samo pokazuje kamerom kartę z kodem w ok. 20 miejscach; instrukcja, karta do druku, *Start: ramię pokaże kartę kamerom (…)* — zablokowany z powodem, gdy nie może ruszyć (m.in. przed krokiem 1), na prawdziwym ramieniu drugie kliknięcie potwierdza; po pomiarze tabela „dobrze / źle” i *Zapisz wynik* / *Odrzuć*. Poruszona kamera: pomarańczowy przycisk *Kamera … została poruszona — pokaż jej kartę jeszcze raz*. Symulacja: skrót *Symulacja: pomiń — wpisz prawdziwe położenie kamer* |
| **Mapa** | kadry wszystkich zaufanych kamer zszyte w mapę blatu (także na stole w 3D), wykrywanie kostki (kolor do wyboru); *Tag na biurku*: poza taga z kamer i *Jedź nad tag* |
| **Dynamika** | **identyfikacja dynamiki serw** na połączonym ramieniu i zapis jako dynamiki stanowiska (`lerobot-twin export` niesie ją w `station.json`) |
| **Strategie** | strategie z zainstalowanych pakietów (entry points `lerobot_mp.strategies`); formularz ustawień wybranej strategii, wypełniony z sekcji pliku konfiguracji o jej nazwie; *Uruchom* (odmawia przy nieznanym kluczu albo złym typie, zanim ramię drgnie) i *Zatrzymaj*; strategia jedzie przez `StrategyRunner`, tak jak strategia testowa |
| **Sim-Real** | kadr prawdziwej kamery z nałożonymi krawędziami renderu bliźniaka z tej samej pozy i K, mediana rozjazdu w pikselach |

---

## Pierwszy test na prawdziwym stanowisku — krok po kroku

Wszystko poniżej jest przećwiczone na bliźniaku (`sim`) — ta sama ścieżka,
te same przyciski. Przed każdym krokiem na sprzęcie: ramię na stole, wolne
miejsce dookoła, zasilacz podłączony, ręka przy STOP.

**0. Ramię do komputera z bliźniakiem.** Połączenie zależy od ramienia
i opisuje je jego pakiet: SO-101 (port szeregowy, most COM<->TCP) — pakiet
`soarm` (repozytorium so-arm101, `docs/first-contact.md`); Galaxea A1X (CAN) —
rozdział [Galaxea A1X](#galaxea-a1x). Sekcję backendu wpisz w
`configs/local.yaml`; panel uruchamiaj wtedy z `--config configs/local.yaml`.

W zakładce Ramię: backend ramienia, port, **Połącz** (bez jazdy do domu).
Suwaki mają pokazywać to, co ramię. UZBRÓJ + mały ruch jednym suwakiem.

**1. Kamery.** Na maszynie wirtualnej/zdalnej najpierw przepuść
kamery USB w kliencie. W panelu *Szukaj kamer USB* → *Dodaj kamerę USB*. Kamera, która się otwiera, ale nie daje klatek, to prawie
zawsze brak przepuszczenia (patrz README).

**2. Obiektyw każdej kamery** (w kodzie: intrynsyki). Kalibracja, krok 1:
*Pobierz szachownicę do druku*, wydrukuj w skali 100%, **zmierz bok kwadratu
linijką**, wpisz. *Zacznij: pokazuj szachownicę kamerze …* i pokazuj ją w
rogach obrazu, bliżej i dalej, pochyloną — 12 zdjęć, pokrycie 55%. *Gotowe —
policz obiektyw*; panel sam przechodzi do następnej kamery. Szachownica
trzymana równolegle do obiektywu nie wyznacza ogniskowej — sesja bez pochyleń
(rozrzut < 35°) albo z ogniskową nie do wiary wychodzi jako **niepewna**, a
kamera z takim obiektywem nie przejdzie kroku 3. *Anuluj* niczego nie liczy
ani nie zapisuje.
Obiektyw zmierzony gdzie indziej (eksport K i dystorsji do pliku tekstowego,
jak `assets/calib/logitech_brio_4k_calib.txt`) wczytuje
*Gotowy obiektyw z pliku* → *Wczytaj obiektyw z pliku* — do wybranej kamery,
tylko z tej samej rozdzielczości; plik trafia do `assets/calib/`.

**3. Położenie kamer** (w kodzie: fala). Kalibracja, krok 2: *Pobierz kartę do
druku*, wydrukuj, zmierz bok kodu, zegnij kartę, włóż wolny koniec w szczęki
(ok. 7 cm ma wystawać). *Start: ramię pokaże kartę kamerom (…)*; przycisk
pyta, czy karta jest w chwytaku, a stół pusty — drugie kliknięcie w ciągu 8 s
rusza. Przycisk jest zablokowany, dopóki któraś prawdziwa kamera nie ma
zmierzonego obiektywu. Ramię zbiera ~20 póz: dopóki kamera nie zobaczy karty
co najmniej dwa razy, fala zamiata stawem podstawy sektorami po obu stronach
stołu (nie błądzi losowo); potem co druga poza celuje w kamerę, której najbardziej
brakuje. Kamera, której się nie udało, dostaje „źle” z powodem. *Zapisz wynik* (zapisuje też użyty bok kodu —
zmieniony po pomiarze blokuje zapis z powodem) albo *Odrzuć*. Wyjmij kartę.

**4. Sprawdzenie Sim-Real.** Zakładka Sim-Real, każda kamera: krawędzie renderu
mają leżeć na krawędziach ramienia i stołu. Mediana > 3 px = coś jest źle
(zmierzony tag? stół w bliźniaku nie na wysokości podstawy?).

**5. Dynamika serw.** Zakładka Dynamika → *Identyfikuj na połączonym ramieniu*
(~20 s ruchu po 12°; do pozy startowej ramię jedzie 30°/s) → *Zapisz jako
dynamikę stanowiska*. `lerobot-twin export` niesie ją w `station.json`, więc
trening randomizuje wokół zmierzonego ramienia, nie katalogu. Z tego ruchu
wyznaczalne są **tłumienie, armatura i opóźnienie** — na ruchu z panelu (5 stawów, 20 s), dla 13 syntetycznych ramion spoza siatki
startowej i szumu 0,05°: tłumienie do 10,5%, armatura do 8,2%, opóźnienie do
1,9 ms; podawany przedział niepewności zawsze obejmował prawdziwy błąd. Przy
czasach jak na prawdziwym ramieniu (30 Hz z jitterem, odczyt co drugi takt)
nagranie idzie z taktów pętli (`Twin.add_tick_listener`), więc wynik zostaje
w tych granicach: tłumienie do 5%, armatura do 8%, opóźnienie do 1,3 ms
(wcześniejsze znakowanie zegarem nagrywającego dawało do 38% i 4,4 ms). Wynik
panel pokazuje z przedziałem przy każdej wartości i z backendem, na którym
nagrano. Dopasowanie, które niczego nie wyjaśnia (błąd niewiele mniejszy od
modelu nominalnego albo przedział > 30% / nieskończony), ma wyłączone *Zapisz*
— zapis tylko po zaznaczeniu *Zapisz mimo to*, i taki wynik nie zalicza kroku 5.
Wzmocnienia serw i tarcia suchego ten ruch nie rozróżnia (kp 0,63 dopasowuje
się tak samo dobrze jak 1,0) — zostają z modelu, panel pisze przy nich
„z modelu”, a opis wyniku podaje dla nich szerokie zakresy treningu: kp
×0,6–1,5, tarcie ×0,5–2,0 (`dynamics.UNFITTED_RANGES`). Dopasowanie trwa
7–11 s; STOP, *Połącz* i *Rozłącz* przerywają tylko nagrywanie — gotowe
nagranie dopasowuje się do końca; przerwane świadomie (STOP, Dom, Połącz)
panel pokazuje jako *Przerwane*, nie jako błąd. Nagranie
z martwą pętlą albo nieświeżymi odczytami jest odrzucane.

**Dalej: strategie.** Zakładka Strategie uruchamia strategie z zainstalowanych
pakietów; wytrenowana polityka to strategia `policy` z `soarm-lab`.

**Chwytak.** 0..100 w panelu to w bliźniaku położenie szczęki liniowo na
zakresie `gripper_range_q` z `robot.yaml` ramienia. Backend podaje swój zakres;
panel ostrzega po połączeniu, gdy oba się rozjeżdżają. Sprawdź na ramieniu, że
przy 0 szczęki się stykają. Tiki chwytaka SO-101 ustawia sekcja `feetech:`
backendu z pakietu `soarm`.

---

## Galaxea A1X

Galaxea A1X (opis ramienia, model MuJoCo, przykładowe stanowisko w symulacji
i backend `a1x` po CAN-FD) jest w pakiecie `galaxeo` z repozytorium
[galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators).
Bliźniak nie zna go z nazwy: znajduje ramię i backend przez entry points
pakietu, który instaluje się obok bliźniaka:

```bash
pip install "galaxeo @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
# macOS z przejściówką XCAN: "galaxeo[xcan] @ git+https://github.com/machinekind/galaxeo-manipulators.git@main"
lerobot-twin demo --robot a1x      # stanowisko A1X w symulacji -> workspace/twin.json
```

Sprzęt i łącze CAN, sekcja `a1x:` konfiguracji, pierwsza sesja krok po kroku
i bezpieczeństwo (A1X nie ma hamulców) opisuje
[docs/TWIN.md w galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators/blob/main/docs/TWIN.md).

---

## Polecenia

| | |
|---|---|
| `lerobot-twin check` | czy środowisko jest gotowe (bez renderu) |
| `lerobot-twin demo` | przykładowe stanowisko w symulacji z pierwszym zainstalowanym ramieniem → `workspace/twin.json` (`--robot`, `--force`, `--path`) |
| `lerobot-twin demo --robot NAZWA` | to samo z innym zainstalowanym ramieniem; pakiet ramienia może mieć własne stanowisko w folderze ramienia (`demo.json`), np. `a1x` z galaxeo |
| `lerobot-twin ui` | panel w przeglądarce |
| `lerobot-twin --config configs/local.yaml <polecenie>` | konfiguracja ramienia z pliku (sekcje backendów, np. `a1x:`) |
| `lerobot-twin card --tag-mm 50` | arkusz karty kalibracyjnej (PNG, A4) |
| `lerobot-twin board --square-mm 28` | tablica ChArUco do intrynsyk (PNG, A4) |
| `lerobot-twin calib-sim --n 5 --cameras 2` | kalibracja w symulacji, oceniana względem prawdy, na pierwszym zainstalowanym ramieniu; `--robot NAZWA` na innym zainstalowanym ramieniu |
| `lerobot-twin workspace` | co wiadomo o stanowisku |
| `lerobot-twin export <folder>` | stanowisko jako dane: `scene.xml` + `assets/` + `station.json` — dla pakietów, które jadą na zmierzonym stanowisku bez importu bliźniaka (trening) |

```bash
pytest -q                                          # całość, z renderem
pytest -q -m "not render"                          # maszyna bez OpenGL
pytest -q -m "not cli"                             # bez testów CLI w podprocesie (kilka sekund)
pytest -q -n auto --dist loadfile                  # równolegle, plik na jednym procesie (pytest-xdist)
```

Testy domu etapami i `Twin.move` jadą na zegarze wirtualnym
(`twin/clock.py`): bliźniak bez wątku pętli (`connect(threaded=False)`),
`VirtualClock` przesuwa `step(dt)`, a `move` czyta ten sam zegar
(`Twin.clock`). Rampy i dojazd liczą się w sekundach symulowanych: dziesięć
sekund ruchu to ułamek sekundy procesora, a czasy powtarzają się co do taktu.
Stanowisko testowe: `tests/twinbench.py`
(`Bench.pump(until=..., sim_s=..., max_sim_s=10.0, dt=0.02)`). W aplikacji
wszystko jedzie na `WallClock` jak dotąd.

Najwięcej czasu zjada planowanie fali kalibracyjnej w `tests/test_twin_calib.py`
(ok. 35 s procesora, nie czekania). `--dist loadfile` trzyma plik na jednym
procesie, więc bliźniak o zasięgu modułu powstaje raz na plik; xdist nie jest
w `addopts` — podaj go, gdy chcesz.

---

## Liczby orientacyjne

| | RTX A4500 (4 rdzenie CPU) | laptop bez GPU (Intel HD 620) |
|---|---|---|
| budowa sceny | 111 ms | 293 ms |
| 1 s fizyki (CPU) | 2,6 ms | 6,9 ms |
| render 640×480 (cienie, MSAA) | **2,1 ms** | 164 ms |
| render 640×480 bez cieni i MSAA | 1,25 ms | 70 ms |
| render 224×224 / 128×128 | 0,8 / 0,7 ms | – |
| pętla sterowania | 48–50 Hz | 28 Hz |

---

## Kalibracja kamer

Metoda z [galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators/pull/3):
**jedna karta z dwoma AprilTagami, ściśnięta w chwytaku**. Ramię nią macha,
kamery patrzą, a solver wyznacza *jednocześnie* pozę każdej kamery względem
podstawy i pozę karty w dłoni — więc nie trzeba wiedzieć, jak krzywo ktoś ją
włożył. Karta jest wspólna dla wszystkich kamer.

Przed kalibracją położenia każda prawdziwa kamera dostaje **intrynsyki z
tablicy ChArUco** (`calib/intrinsics.py`) — inaczej poza jest dopasowywana do
nominalnego K z 65° pola widzenia i wychodzi pewna siebie i przesunięta.
Tablica to ArUco 5×5, inny słownik niż tagi karty (36h11).

### Wynik w symulacji (`lerobot-twin calib-sim --n 5 --cameras 2`)

| | przed poprawką półpiksela | **po** |
|---|---|---|
| zaufane kamery | 10/10 | 10/10 |
| błąd położenia kamery (mediana) | 0,345 mm | **0,161 mm** |
| błąd obrotu kamery (mediana) | 0,053° | **0,013°** |
| residuum (mediana) | 0,22 px | 0,21 px |

Z panelu, na żywym bliźniaku, dwie kamery: 0,19 mm / 0,009° i 0,31 mm / 0,005°
względem prawdy. Intrynsyki z symulowanej tablicy: ogniskowe z dokładnością
~0,1–1 px, residuum 0,1 px.

### Przestawiona kamera

Po zapisaniu kalibracji panel zapamiętuje kadr odniesienia każdej kamery
i co 2 s porównuje z nim bieżący — korelacją fazową, z **ramieniem wyciętym
maską z bliźniaka** (ruch ramienia daje 0,4 px „przesunięcia”, obrót kamery
o 1° — 9,8 px). Oprócz przesunięcia łapie **obrót wokół osi kamery i zmianę
skali** (dopasowanie podobieństwa na piramidzie ¼–½ kadru); miarą jest
największe przesunięcie narożnika kadru, a dopasowanie jest odporne (wagi
Tukeya): ręka albo tułów w kadrze nie udają ruchu kamery (ręka na 10–20 %
kadru: ≤ 0,6 px; wcześniej do 59 px). Ruch ramienia daje ≤ 0,1 px (wcześniej
do ok. 1,3 px), przybliżenie o 1% — 3,9–4,5 px (prawda 4), obrót o 1° —
6,7–6,9 px (prawda 7). Alarm „przestawiona” wymaga **dwóch kolejnych
zgodnych** sprawdzeń. Sprawdzenie kosztuje ok. 15 ms na kamerę. Powyżej 3 px piramida robi się czerwona, a w zakładce
Kalibracja pojawia się pomarańczowy przycisk *Kamera … została poruszona* (krótki
pomiar tylko dla tej kamery). Klatka
starsza niż 1 s nie jest kadrem — zamrożony albo martwy strumień panel pokazuje
jako błąd kamery i otwiera go na nowo.

## Dokładność symulowanej kamery

Symulowana kamera dostaje pełną macierz K — ogniskowe w pikselach i punkt
główny. Test renderuje kulki w znanych punktach stołu i porównuje środki ich
**masek segmentacji** z rzutem przez K: przesunięcie systematyczne poniżej
0,15 px, dla trzech różnych K.

| | objaw | poprawka |
|---|---|---|
| znak punktu głównego w MuJoCo | cały kadr przesunięty o 62 px | „środek minus K” w obu osiach |
| ~~półpiksel w pionie~~ | render 0,5 px **niżej** niż rzut K | dawna „kompensacja” była błędem pomiaru: środki czerwonych plam podnosił cień dolnej połowy kulki; maski segmentacji (czysta geometria) pokazały, że to ona przesuwała kadr. Po usunięciu błąd obrotu w kalibracji spadł 4× |
| półpiksel w detektorze AprilTag | rogi o 0,5 px dalej niż w konwencji `projectPoints` | korekta w `tags.detect` |

---

## Mapa stołu i kostka

Każdy kadr przerysowany na płaszczyznę blatu, mapy zszyte z wagami — do
oglądania stanowiska (także na blacie w 3D).

Kostka (`perception.CubeDetector.detect_frames`):

1. **start** — część wspólna masek koloru z kamer na wysokości górnej ściany
   (ściana leży w tym samym miejscu dla każdej kamery, boki rozjeżdżają się);
2. **dopasowanie sylwetki** — (x, y, obrót), przy których rzut bryły kostki
   najlepiej pokrywa maski wszystkich kamer; działa z jedną kamerą i z wieloma,
   liczone w wycinku kadru (~17 ms);
3. **ramię to „nie wiem”** — piksele zasłonięte ramieniem (segmentacja z
   bliźniaka, w pozie z serw) nie głosują;
4. **bramka IoU ≥ 0,75** — model zakłada kostkę leżącą; kostkę w powietrzu da
   się „wcisnąć” w jakąś pozę na blacie z IoU 0,36–0,64 i błędem do 1 m;
5. **jedna kamera przy dłoni to za mało** — wykrycie potwierdzone przez jedną
   kamerę (`n_cameras`) blisko chwytaka albo przy kostce w dłoni jest pomijane:
   jedna kamera potrafi „położyć” na blacie kostkę uniesioną w szczękach.
   `n_cameras = -1` (z mapy) i `0` (ramię zasłania kostkę wszystkim) też są
   jednym świadkiem. Jeden świadek przy dłoni jest przyjmowany tylko przy
   pustej szczęce, gdy 3 wykrycia zgadzają się co do 1 cm, a TCP przesunął się
   o ≥ 1,5 cm (kostka potrącona) — albo gdy dłoń stoi, ale szczęka jest otwarta
   szerzej niż 0,40 rad (47 mm; kostka trzymana w szczękach to ok. 0,20 rad,
   więc duch kostki z dłoni tak nie przejdzie). Gdy kostka nie jest przyjęta,
   źródło w panelu mówi dlaczego i co zrobić („brak (…)”). Na prawdziwym ramieniu
   sprawdź, że kostka trzymana w szczękach daje kąt poniżej 0,40 rad. Kostka,
   która wypadła z dłoni, opada na blat
   („upuszczona”). Wykrycie ma czas kadru (`t`); stare i powtórzone nie
   przedłużają życia kostki.

Na blacie w symulacji: **1–3 mm, ~1°**. `CubeTracker` przejmuje kostkę, gdy
kamery jej nie widzą: szczęka **zablokowana** (nie dojeżdża do rozkazu i stoi)
→ kostka jedzie z TCP w pozie zapisanej w chwili chwytu.

## Jak to jest zbudowane

```
src/lerobot_mp/twin/
├── robots.py        opisy ramion z folderów (`robot.yaml`), tylko z wtyczek `lerobot_mp.robots`
├── config.py        konfiguracja z YAML: safety, robot, stawy; sekcje backendów i strategii surowe (`TwinConfig`)
├── kinematics.py    FK/IK przez MuJoCo; IK z priorytetem pozycji (5 osi SO-101)
├── scene.py         scena z MjSpec: stół, ramię, kamery z pełnym K, karta, obiekty, panele, czujniki chwytu; eksport do MJCF
├── collision.py     kolizje pozy i drogi do niej
├── workspace.py     stanowisko w JSON: ramię, port, stół, karta, kamery (+ prawda symulowanych), dynamika
├── cameras.py       wykrywanie kamer, strumienie bez lustra, kamery symulowane
├── camera_stream.py wątek kamery: zawsze najnowsza klatka, bez kolejki
├── runtime.py       pętla sterowania w wątku, nadzór, wątek renderujący
├── safety.py        nadzór: limity pozycji i prędkości, STOP, powrót do domu (`SafetySupervisor`)
├── strategy.py      `StrategyRunner`; kontrakt strategii (`Observation`, `Strategy`, `Action`) re-eksportowany z `lerobot_mp_api`
├── waypoints.py     strategia testowa: przejazd przez zadane pozy
├── strategies.py    strategie z wtyczek `lerobot_mp.strategies` i ich ustawienia (dataclass, budowana ściśle)
├── perception.py    mapa stołu na żywo, kostka z kamer, śledzenie kostki w dłoni
├── sysid.py         identyfikacja dynamiki serw
├── dynamics.py      dynamika zmierzona (`Dynamics`, zapis w stanowisku)
├── cli.py           lerobot-twin
├── calib/
│   ├── tags.py        AprilTag 36h11 + korekta półpiksela
│   ├── handeye.py     solver wielu kamer ze wspólną kartą
│   ├── card.py        karta: geometria, poza w szczękach, arkusz
│   ├── intrinsics.py  ChArUco: arkusz, zbieranie ujęć, K i dystorsja, sesja w symulacji
│   ├── session.py     fala: szukanie, celowanie, kadr, rozrzut, bramki
│   ├── simulate.py    sesja w symulacji względem prawdy
│   └── topdown.py     rzut kadru na blat
└── ui/
    ├── app.py         panel (viser)
    ├── bridge.py      scena MuJoCo w przeglądarce (węzeł na ciało)
    ├── guide.py       przewodnik: kroki procedury i następna akcja ze stanu panelu
    ├── jobs.py        zadania w tle: kalibracje, identyfikacja
    ├── strategy_tab.py zakładka Strategie: lista, formularz ustawień, Uruchom i Zatrzymaj
    └── watch.py       wykrywanie przestawionej kamery

src/lerobot_mp/robot/
└── base.py, factory.py   kontrakt backendu (re-eksport z `lerobot_mp_api`) i rejestr wtyczek

packages/lerobot_mp_api/      pakiet lerobot-mp-api: kontrakty wtyczek, tylko numpy
├── strategy.py           `Observation`, `Action`, `ActionLike`, `RunInfo`, `ObjectPose`, `Frame`, protokół `Strategy`
├── backend.py            `BackendContext`, protokół `BackendPlugin`, `RobotBackend`, `RobotInfo`
└── entry_points.py       grupy entry points: `ROBOTS`, `BACKENDS`, `STRATEGIES`
```

**Kontrakty wtyczek.** Typy kontraktu leżą w pakiecie `lerobot-mp-api`
(`packages/lerobot_mp_api/`), a nie w bliźniaku. Wtyczka — pakiet ramienia (folder z `robot.yaml`
i backend) albo pakiet strategii — zależy od niego, nigdy od `lerobot-mp`
(MuJoCo, viser, OpenCV). Bliźniak re-eksportuje te same klasy z `lerobot_mp.twin.strategy`
i `lerobot_mp.robot.base`, więc dotychczasowe importy i sprawdzenia `isinstance` działają
bez zmian. Gotowe wzory: `examples/robots/testbot` i `examples/strategies/swing`.
