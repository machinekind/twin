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
| **Drugie ramię: Galaxea A1X** | model, kinematyka i kalibracja kamer w symulacji; backend `a1x` po CAN-FD przetestowany na symulowanej magistrali — sekcja [Galaxea A1X](#galaxea-a1x) |
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
`lerobot:` w pakiecie `soarm`); Galaxea A1X ma sekcję `a1x:`, patrz
[Galaxea A1X](#galaxea-a1x). Opis samego ramienia (stawy, limity, poza domowa,
zakres szczęki) leży w `robot.yaml` w folderze ramienia, w pakiecie ramienia
(A1X: `src/lerobot_mp/robot/a1x/robot/`).

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
| **Ramię** | połączenie: `sim` (bliźniak jest ramieniem), backend z pakietu ramienia (SO-101 z `soarm`: `feetech`, `lerobot`), `a1x` (Galaxea A1X po CAN: port `can0` / `xcan` / `PCAN_USBBUS1`, przycisk *Włącz silniki A1X*) — zestawy w liście *Ramię* wpisują port; po *Połącz*: duży **UZBRÓJ / UZBROJONE**, *Uchwyt 3D* (IK), *Dom*, *Ustaw dom = tu*, prędkość, suwaki stawów z nazwami po polsku, *Rozłącz*; *Stanowisko (stół)* zwinięte na dole |
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

Drugie ramię bliźniaka: 6 stawów i chwytak z wysuwanymi szczękami, sterowane
po CAN-FD. Ten sam panel, ten sam nadzór i ta sama procedura co przy SO-101.
Różnice siedzą w trzech miejscach. Stanowisko ma `"robot": "a1x"`
(`workspace/twin.json`), więc bliźniak bierze model `src/lerobot_mp/robot/a1x/robot`
i opis ramienia `src/lerobot_mp/robot/a1x/robot/robot.yaml`. Ze sprzętem rozmawia backend `a1x`
(`src/lerobot_mp/robot/a1x/backend.py`). Protokół i łącze CAN dostarcza pakiet
`galaxeo` z repozytorium galaxeo-manipulators — to samo źródło, z którego
korzystają narzędzia galaxeo (`jog_a1x.py`, `so101_bridge.py`). Szczegóły
protokołu i pomiary: `docs/PROTOCOL.md`, `docs/STEERING.md`, `docs/HARDWARE.md`
i **`docs/SAFETY.md`** w galaxeo-manipulators. SAFETY.md przeczytaj przed
pierwszym włączeniem zasilania.

### Sprzęt i łącze CAN

- Zasilacz **24 V** (nie 48 V). W skrzynce CAN przełączniki terminacji
  **R1 i R2 w górnej pozycji**. Przejściówka XCAN / PEAK USB-CAN FD
  (USB `0c72:0012`).
- **Jeden nadawca na magistrali.** Gdy działa panel z ramieniem `a1x`, nie
  uruchamiaj `ARM_APP`, `so101_bridge.py`, `jog_a1x.py`, `teleop2.py` ani
  sterownika ROS 2 z nadawaniem. Dwa strumienie na `0x050` przeplatają się
  i ramię nie jedzie za żadnym.

| system | łącze | pole Port w panelu |
|---|---|---|
| Linux | SocketCAN, jądro `peak_usb`; w Pythonie nic więcej | `can0` (pusty = `can0`) |
| macOS | sterownik w przestrzeni użytkownika (`galaxeo.xcan_usb`, libusb): `brew install libusb` + `pyusb` | `xcan`, przy dwóch przejściówkach `xcan:<adres USB>` (pusty = `xcan`) |
| Windows | PEAK PCAN-Basic (sterownik PEAK) przez `python-can` | `PCAN_USBBUS1` — **wpisz jawnie**, pusty oznacza `can0` |

**Linux.** Łącze podnosi `./can_up.sh` z galaxeo-manipulators. Skrypt działa
w kontenerze Dockera tamtego repozytorium (bez `sudo` na hoście) i wybiera przejściówkę po ruchu na magistrali.
Bez kontenera, ręcznie (czasy producenta):

```bash
sudo ip link set can0 type can \
    bitrate 1000000 sample-point 0.875 \
    dbitrate 5000000 dsample-point 0.875 \
    fd on restart-ms 100
sudo ip link set can0 txqueuelen 65535
sudo ip link set can0 up
ip -det -s link show can0     # stan łącza, bus-off, liczniki błędów
```

Nie dodawaj `berr-reporting on`: `pcan_usb_fd` go nie obsługuje i całe
`ip link set` się nie udaje. **Po każdym przełożeniu wtyczki łącze wstaje
wyłączone** — trzeba je podnieść od nowa. Dwie przejściówki mają ten sam
numer seryjny; które ramię wisi na którym łączu, pokaże `python3 which_arm.py
can0 can1` z galaxeo (tylko odczyt).

**macOS.** Tu nie ma SocketCAN, a biblioteka PCBUSB otwiera przejściówkę, ale
nie odbiera z niej ramek. Dlatego `galaxeo.xcan_usb` rozmawia z nią wprost
przez libusb. Sprawdzenie bez nadawania:

```bash
python -m galaxeo.xcan_usb --normal     # wersja firmware, potem 0x052 z ok. 200 Hz; nic nie wysyła
```

Bez `--normal` przejściówka słucha bez potwierdzania ramek. Ramię powtarza
wtedy każdą ramkę i `0x052` przychodzi z ok. 6 kHz z powtórzoną treścią, co
nie jest usterką. Przejściówkę może mieć otwartą **jeden proces naraz**: zamknij
sprawdzenie, zanim panel się połączy.

### Instalacja

```bash
pip install -e packages/lerobot_mp_api -e ".[twin,a1x,dev]" -c constraints.txt
```

Extra `a1x` pobiera pakiet `galaxeo` z gita (`galaxeo-manipulators@main`).
**Pakiet jest na razie tylko na gałęzi `pkg`** — `main` nie ma jeszcze
`pyproject.toml`, więc dopóki `pkg` nie jest scalona, instalacja z `.[a1x]`
się nie uda (zainstaluj bez `a1x`). Do tego czasu zainstaluj go z lokalnego
klonu na gałęzi `pkg`:

```bash
pip install -e ../galaxeo-manipulators            # Linux: sama biblioteka standardowa
pip install -e "../galaxeo-manipulators[xcan]"    # macOS: + pyusb (i brew install libusb)
pip install -e "../galaxeo-manipulators[pcan]"    # Windows: + python-can
python -c "from galaxeo import protocol; protocol.selftest(); import lerobot_mp.robot.a1x"
```

Extra `a1x` nie dociąga `pyusb` ani `python-can` — na macOS i Windows doinstaluj
je jak wyżej albo przez `pip install pyusb` / `pip install python-can`. Bez pakietu
`galaxeo` bliźniak działa normalnie. Dopiero *Połącz* z ramieniem `a1x`
odpowiada błędem z poleceniem instalacji.

### Stanowisko i konfiguracja

**Stanowisko.** O ramieniu decyduje pole `robot` pliku stanowiska, a panel go
nie zmienia. Najprościej zacząć od przykładu:

```bash
lerobot-twin demo --robot a1x          # examples/twin.a1x.sim.json -> workspace/twin.json (stół 1,2 × 0,8 m, 2 kamery sim)
lerobot-twin workspace                 # "ramie: a1x, backend: sim, ..."
```

Na prawdziwe stanowisko usuń potem w zakładce Kamery obie kamery symulowane
(*Usuń kamerę*) i dodaj kamery USB. Plik stanowiska `{"robot": "a1x"}` też
wystarczy (pusty stół i bez kamer). Backend i port zapisuje w stanowisku
dopiero *Połącz*.

**`configs/local.yaml`.** Backend `a1x` czyta sekcję `a1x:`, której nie ma
w `configs/default.yaml`. Napisz samą sekcję (reszta bierze wartości
domyślne, klucze są w tabeli niżej), a panel uruchamiaj z
`lerobot-twin --config configs/local.yaml ui`:

```yaml
a1x:
  joint_offsets_deg: [0.0, 0.0, 1.49, 0.0, 0.0, 0.0]   # zmierz swoje (niżej)
  # chwytak: domyślne końce i sprzężenie są zmierzone (2026-09-27, niżej); inne ramię - zmierz swoje
```

`robot.backend` i `robot.port` z pliku bliźniak **pomija**: backend i port
wybiera panel (pole Ramię i Port), a zapamiętuje stanowisko.

| klucz | domyślnie | znaczenie |
|---|---|---|
| `kp` | 20,0 | wzmocnienie wysyłane w `0x050`. Na tym łączu ramię go ignoruje (działa tylko `p_des`), ale przy `kp` 0 przestaje słuchać rozkazów — konfiguracja odrzuca `kp <= 0` |
| `kd` | 1,0 | tłumienie w `0x050` i `0x051`; bez wpływu na ruch |
| `gripper_kp` | 25,0 | `kp` ramki chwytaka `0x051`; też musi być > 0 |
| `gripper_open_pdes` | −1,95 | `p_des` chwytaka [rad] przy 100 (otwarty); bardziej ujemne = szerzej. Włos przed otwartym ogranicznikiem (−1,992), żeby się o niego nie opierał |
| `gripper_closed_pdes` | 0,08 | `p_des` [rad] przy 0 (zamknięty): 0,04 rad docisku za zamkniętym ogranicznikiem (+0,039). **Nie +0,6** — patrz „Chwytak A1X: pomiar” niżej |
| `gripper_fb_closed` / `gripper_fb_open` | 0,039 / −1,992 | pozycja grupy 7 ramki `0x052` [rad] przy 0 i przy 100 (ograniczniki zmierzone 2026-09-27). Podaj obie albo żadną (różne od siebie). `null` w obu = pętla otwarta: panel pokazuje **rozkaz** chwytaka, nie pomiar (ostrzeżenie „sprzężenie niezmierzone”) |
| `gripper_start` | 0,0 | co backend zgłasza jako chwytak przed pierwszym rozkazem, gdy sprzężenie niezmierzone. 0 = zamknięty, zgodnie z umową stanowiska „dom = szczęki zamknięte” (*Ustaw dom = tu*) |
| `gripper_rate_pdes_s` | 2,0 | największa prędkość `p_des` chwytaka [rad/s]; pełny skok 2,03 w ok. 1,0 s (`gripper_travel_s` dolicza 0,2 s zapasu) |
| `gripper_force` | 1,2 | o tyle \|effort\| grupy 7 przy zamykaniu może przekroczyć spoczynek, zanim cel chwytaka zamarznie (chwyta, nie miażdży). Spoczynek to mediana \|effort\| z pierwszych 0,1 s każdego zamykania (w tym oknie cel nie zamarza); liczy się od nowa po każdym otwieraniu. Jednostka empiryczna, miażdżenie widziane od ok. 6 |
| `joint_offsets_deg` | 6 × 0,0 | zero stawów ramienia [°]: odejmowane przy odczycie, dodawane przy wysyłce |
| `stream_hz` | 200,0 | częstotliwość strumienia `0x050` / `0x051`; ramię odpowiada `0x052` co 5 ms |
| `max_step_deg_s` | 90,0 | własny limit prędkości `p_des` backendu [°/s], druga warstwa pod nadzorem |
| `stale_ms` | 150,0 | brak `0x052` dłużej niż tyle = łącze milczy: wysyłka staje, cele przepadają, bliźniak robi STOP |
| `frozen_frames` | 50 | tyle identycznych ramek `0x052` z rzędu = telemetria zamrożona (ramię zwolnione FF 2) |
| `deaf_err_deg` / `deaf_s` | 3,0 / 1,0 | rozkaz dalej od pomiaru niż `deaf_err_deg` przez `deaf_s` przy stojącym stawie (< 0,3°/s) = ramię głuche na `0x050` albo zablokowane → błąd i STOP |
| `enable_on_connect` | `false` | sekwencja FF 1 → 5 → 6 zaraz po *Połącz*. Zostaw `false` i używaj przycisku w panelu, bo FF 5 na chwilę puszcza silniki |
| `gripper_over_xcan` | `true` | chwytak przez XCAN (macOS) działa (zmierzone 2026-09-27, niżej); `false` = odmowa „no gripper”. Zmienna środowiskowa `A1X_GRIPPER_XCAN=1` wymusza `true` |

**Zero stawów (`joint_offsets_deg`).** Na ramieniu z galaxeo J3 w spoczynku
czyta ok. +1,49° — o tyle za własnym limitem 0°. To przesunięcie zera, nie błąd
dekodowania. Zmierz swoje: ramię złożone, oparte na ogranicznikach, zasilone,
*Połącz* bez jazdy do domu z `joint_offsets_deg` samych zer. Odczytaj kąty
stawów opartych o ogranicznik (J2 i J3) z suwaków albo z logu (`A1X na …: poza
startowa …`). Wartość J3 (np. `1.49`) wpisz na trzecie miejsce, a pozostałe
zostaw 0, chyba że masz dla nich pewne odniesienie. Wpisana wartość jest
odejmowana od odczytu, więc po restarcie panelu J3 ma czytać ok. 0.

**Chwytak A1X: pomiar z 2026-09-27.** Na prawdziwym A1X przez XCAN
(galaxeo `python -m galaxeo.xcan_usb --grip-test`: `0x051` z 200 Hz, kp 20,
kd 1, 3 s na każdy krok) wyszło:

| `p_des` [rad] | grupa 7 `0x052` | \|effort\| szczyt | |
|---|---|---|---|
| −1,50 | −85,60° = −1,494 rad | 0,25 | swobodny ruch |
| 0,00 | −0,40° = −0,007 rad | 0,21 | swobodny ruch |
| −2,00 | −114,14° = −1,992 rad | 3,69 | na otwartym ograniczniku albo tuż za nim |
| +0,60 | +2,21° = +0,039 rad | 7,80 | **zamknięty ogranicznik, 0,56 rad za nim** |

Wnioski:

- **Grupa 7 ramki `0x052` to pozycja szczęk**, w tych samych jednostkach co
  `p_des` (rad, `surowe / 4700`). W swobodnym ruchu pomiar = rozkaz (−1,494 przy
  −1,5; −0,007 przy 0,0). Backend czyta ją jako `pos[6]` i przelicza przez
  `gripper_fb_closed/open` na 0..100. Gdy chwytak jest sterowany, **chwytak jest
  zmierzony** (`gripper_measured()`): panel, szczęki w 3D i rozpoznawanie chwytu
  („w dłoni”) widzą szczęki, nie rozkaz.
- **W spoczynku po włączeniu zasilania grupa 7 pokazuje dowolną wartość**, np.
  −164,9° (−2,88 rad), choć szczęki stały tam, gdzie zostawił je poprzedni
  przebieg (ok. −85°). Za `p_des` idzie dopiero, gdy płynie `0x051` (−85,6° przy
  −1,5; −0,4° przy 0,0). Dlatego backend liczy grupę 7 jako pomiar dopiero od
  4 ramek `0x052` po pierwszej wysłanej `0x051` w tym połączeniu (i od nowa po
  utracie łącza, bo ramię mogło zostać wyłączone). Wcześniej zgłasza
  przekonanie jak bez sprzężenia: `gripper_start` (0), *Ustaw dom = tu* działa
  (`assume_gripper`), a pierwsze otwieranie idzie od razu na cel. Reguła
  „od strumienia”, a nie „wartość w zmierzonym zakresie ± 0,15 rad”, bo dowolna
  wartość spoczynkowa może trafić w zakres, a strumień `0x051` to warunek, który
  pomiar naprawdę pokazał. Surową wartość spoczynkową backend loguje przy *Połącz*
  („A1X chwytak: grupa 7 w spoczynku …”).
- **Ograniczniki:** zamknięty ok. +0,039 rad (+2,2°), otwarty ok. −1,992 rad
  (−114°). To są domyślne `gripper_fb_closed` / `gripper_fb_open`.
- **Dlaczego nie +0,6.** Chwytak jest pozycyjny: przy `p_des` +0,6 szczęki stają
  na ograniczniku +0,039, a regulator przez cały czas pcha je w stronę celu
  0,56 rad dalej. Przy kp 20 to prąd utyku: \|effort\| 7,8 zamiast ok. 0,2
  w swobodnym ruchu. Ten prąd płynął przez cały czas trzymania „zamkniętego”
  i **przegrzał silnik chwytaka** (2026-09-27). Stop kontaktowy
  (`gripper_force`) nie chroni przed tym zawsze: gdy zamykanie zaczyna się przy
  szczękach już dociśniętych do ogranicznika, spoczynek z pierwszych 0,1 s to już
  prąd utyku, a `--grip-test` stopu kontaktowego w ogóle nie ma.
  Teraz zamknięty = +0,08: 0,04 rad docisku za ogranicznikiem, czyli ok. 7 %
  dawnego uchybu położenia. Przedmioty i tak łapie stop kontaktowy.
- **Otwarty:** −2,0 opierało się o ogranicznik (effort 3,7). Teraz −1,95,
  0,04 rad przed nim. Pomiar przy 100 czyta wtedy ok. 98, nie 100 — to
  zamierzone.

Czego jeszcze nie sprawdzono: czy 0,04 rad docisku wystarcza, żeby puste szczęki
naprawdę się zeszły, i czy −1,95 nie dotyka już ogranicznika. Jedna próba na
stole: zamknij, odczytaj \|effort\| i pozycję grupy 7 (ma stać ok. +0,039,
effort bliski spoczynku), potem otwórz i to samo przy ok. −1,95.

**Ponowny pomiar (inne ramię albo po serwisie chwytaka).** Najprościej
`python -m galaxeo.xcan_usb --grip-test` (macOS, XCAN; galaxeo PR 20): kilka
kroków `p_des` po 3 s z odczytem grupy 7 i szczytowego \|effort\|. Zamknij przed
nim panel (przejściówkę trzyma jeden proces). Szukaj punktów, w których pozycja
przestaje iść za `p_des`, a effort rośnie: to ograniczniki, wpisz je do
`gripper_fb_closed` / `gripper_fb_open`. `gripper_closed_pdes` ustaw kilka
setnych radiana za zamkniętym ogranicznikiem, `gripper_open_pdes` kilka setnych
przed otwartym. **Nie testuj długo daleko za ogranicznikiem** — prąd utyku grzeje
silnik; krok 3 s wystarcza.

Bez `--grip-test` (Linux, SocketCAN) pozycję grupy 7 czyta sam skrypt, który
nic nie wysyła. Mediana z 50 ramek `0x052`:

```python
from galaxeo import bus, protocol
b = bus.open_bus("can0")          # macOS: "xcan", Windows: "PCAN_USBBUS1"
try:
    pos = []
    while len(pos) < 50:          # 50 ramek 0x052 = 0,25 s
        f = b.recv(0.1)
        if f is not None and f.can_id == protocol.FB_ID and len(f.data) == protocol.FB_LEN:
            pos.append(protocol.decode_feedback(bytes(f.data)).pos[6])
    print(f"grupa 7: {sorted(pos)[25]:.4f} rad")
finally:
    b.close()
```

Pusty chwytak, UZBRÓJ, suwak chwytaka na 0, poczekaj, aż szczęki staną,
i uruchom skrypt; to samo przy 100. Czytaj tylko przy płynącym `0x051`: na
Linuksie skrypt może czytać równolegle z panelem (czytających może być wielu,
nadawca jeden). Na macOS przejściówkę trzyma panel, a po *Rozłącz* strumień
staje — taki odczyt nie jest pewny (w spoczynku po włączeniu zasilania grupa 7
bywa dowolna), więc tam użyj `--grip-test`.

### Pierwsza sesja krok po kroku

Przed sesją: ramię złożone i oparte, miejsce pod nim wolne, pod ręką wyłącznik
awaryjny zasilacza i STOP panelu, na magistrali nic innego nie nadaje. Łącze
sprawdzone: na Linuksie `ip -det -s link show can0`, na macOS
`python -m galaxeo.xcan_usb --normal`, potem je zamknij.

**0. Połączenie.** `lerobot-twin --config configs/local.yaml ui`, zakładka
Ramię: zestaw *Galaxea A1X* dla tego systemu (Port jak w tabeli wyżej), *Połącz*
— nigdy nie jedzie do domu. Nic nie idzie na magistralę, dopóki czegoś nie
ruszysz. Suwaki pokazują zmierzoną pozę, a przewodnik podpowiada ten sam krok.
*Połącz* kończy się błędem, gdy:

- przez 3 s nie przyjdzie żadna ramka `0x052` — łącze nie wstało albo ramię
  nie ma zasilania;
- telemetria jest zamrożona — ramię zostało zwolnione (FF 2). Podeprzyj je,
  wyłącz i włącz zasilanie. FF 1-5-6 tego nie naprawia, bo pozy zwolnionego
  ramienia nie da się odczytać.

Złożone ramię stoi zwykle **poza limitami bliźniaka** (J2 ok. −0,2°, J3 ok.
+1,5° przy limitach 3…177° i −187…−3°). To normalne: nadzór zostawia staw tam,
gdzie stoi (miękkie limity, jak ramię nr 2 SO-101). Pierwszy ruch suwakiem J2
albo J3 wyprowadza staw w zakres z prędkością 15°/s.

**1. Mały ruch.** UZBRÓJ, potem jeden suwak (np. J1 Podstawa) o kilka stopni. Strumień
`p_des` startuje od **zmierzonej** pozy. Jedzie najwyżej 90°/s (backend) i nie
szybciej niż pozwala nadzór. Sprawdź kierunki i to, że suwaki idą za ramieniem.
Ramię zgłasza się, ale nie jedzie? Po ok. 1 s pojawia się błąd „… st. od rozkazu
przez … s i stoi — ramię głuche na 0x050 … albo zablokowane” i STOP. Dopiero
wtedy (albo po zwolnieniu FF 2 i ponownym zasileniu) użyj **Włącz silniki A1X
(FF 1-5-6)**. Przycisk jest widoczny tylko przy połączonym `a1x`, wymaga
zaznaczenia potwierdzenia i odmawia przy uzbrojonym panelu albo gdy ramię ma
strategia, fala lub identyfikacja. W trakcie sekwencji backend strumieniuje
`p_des` = zmierzona poza. Świeżo zasilone ramię słucha od razu i przycisku nie
potrzebuje.

**2. Dom i wyłączenie.** *Dom* A1X to uniesiona poza z modelu
(J2 57°, J3 −92°, J4 34°, szczęki zamknięte). Pierwszy Dom to więc duży ruch w górę.
*Ustaw dom = tu* zapisuje zmierzoną pozę ramienia i **zawsze chwytak 0
(zamknięty)**: to umowa stanowiska, dom = szczęki zamknięte. Najpierw zamknij
puste szczęki naprawdę. Dopóki chwytak nie był w tym połączeniu sterowany (grupa 7
w spoczynku niepewna, patrz „Chwytak A1X: pomiar”) albo bez sprzężenia
(`gripper_fb_*` = `null`), bliźniak przyjmuje od tej chwili, że są zamknięte (panel, szczęki w 3D, następne „otwórz” jako rampa od 0).
Jeśli chwytak był już sterowany w tej sesji, przycisk tylko zapisuje dom, a stan
chwytaka (wtedy już zmierzony) zostaje (ostrzeżenie w logu „chwytak juz sterowany”): zamknij go suwakiem
i kliknij jeszcze raz. W `sim` chwytak sceny zamyka się od razu.
**Przed wyłączeniem zasilania złóż ramię** suwakami (J2 → 3°, J3 → −3°, reszta
→ 0) i podeprzyj je, bo A1X nie ma hamulców. Ostatnie 3° do ogranicznika to
zapas limitów bliźniaka. *Rozłącz* nic nie zwalnia: ramię trzyma pozę, w której
stoi.

**Dalej: kamery, intrynsyki, położenie kamer, Sim-Real, dynamika** — jak kroki
1–5 procedury wyżej, z tymi samymi potwierdzeniami. Na A1X karta w szczękach
wystaje tak samo (ok. 70 mm), a fala losuje pozy z `A1X.wave_ranges`.

### Czym A1X różni się od SO-101 (bezpieczeństwo)

- **Brak hamulców.** Odcięcie zasilania (także wyłącznik awaryjny, wypięty
  kabel) zrzuca ramię z miejsca, w którym jest. Przed wyłączeniem złóż je albo
  podeprzyj.
- **STOP trzyma zmierzoną pozę strumieniem**, nie zwalnia silników. Ramię,
  do którego nic nie jest wysyłane, też trzyma pozę i nie dryfuje. Zatrzymanie
  jest bezpieczne; niebezpieczny jest start z celem innym niż poza ramienia,
  i przed tym chroni start strumienia od pomiaru.
- **Bliźniak nigdy nie wysyła FF 2** (zwolnienie: ramię wiotczeje, opada
  i przestaje raportować). Sekcja `a1x:` nie ma opcji zdjęcia momentu przy
  rozłączeniu, heartbeat `0x023` nie jest wysyłany nigdy.
- **FF 5 na ułamek sekundy puszcza silniki.** Przycisk włączenia tylko przy
  podpartym ramieniu i rozbrojonym panelu; nigdy „na wszelki wypadek”.
- **`t_ff`, `kp`, `kd` nic nie robią** (zmierzone w galaxeo). Nie ma podatności,
  kompensacji grawitacji ani ustępowania przy kolizji. **Jedyną ochroną są
  limity i ograniczenie prędkości**: limity bliźniaka (URDF minus 3–5°),
  limity URDF w backendzie (poszerzone o pozę startową), nadzór i 90°/s
  backendu. Kolizję łapie dopiero wykrywanie rozjazdu (25° przez 0,5 s w nadzorze,
  3° przez 1 s przy stojącym stawie w backendzie).
- **Utrata łącza.** Brak `0x052` przez 150 ms albo zamrożona telemetria:
  wysyłka staje od razu, cele ramienia i chwytaka przepadają, bliźniak robi
  STOP. Po powrocie ramek nic nie jedzie do następnego rozkazu, a ten startuje
  od świeżego pomiaru.
- **Chwytak.** Celu chwytaka nie da się odczytać z ramienia, ale pozycję szczęk
  tak (grupa 7 `0x052`), ale tylko przy sterowanym chwytaku: od pierwszej `0x051`
  w połączeniu. Zamknięty `p_des` to +0,08, tuż za ogranicznikiem: +0,6
  trzymało silnik w utyku i go przegrzało. Przed pierwszą `0x051` i bez
  sprzężenia (`gripper_fb_*` = `null`) backend pracuje w pętli otwartej i panel
  pokazuje rozkaz. Strumień
  `0x051` rusza dopiero przy rozkazie różnym od zgłoszonej wartości (o > 0,5),
  więc *Połącz* niczego nie domyka. Przy zamykaniu cel zamarza, gdy |effort|
  przekroczy spoczynek z pierwszych 0,1 s zamykania o `gripper_force` (1,2);
  każdy rozkaz otwierania (albo równy celowi) zdejmuje zamrożenie. Spoczynek
  mierzy się od nowa dopiero po otwarciu o 0,3 p_des od najciaśniejszego punktu
  zamykania: po uchyleniu o kreskę szczęki dalej dociskają kostkę i docisk nie
  może zostać nowym spoczynkiem (próg rósłby z każdym cyklem). Bez zmierzonego
  sprzężenia stan szczęk po *Połącz* to `gripper_start` (zamknięty), założenie,
  nie pomiar. Otwarte szczęki zamknij suwakiem przed *Rozłącz* i przed *Ustaw
  dom = tu*. Dopóki stan jest tylko zgadnięty (`gripper_start` albo *Ustaw dom
  = tu*), pierwsze otwieranie idzie od razu na cel (pusty, zamknięty chwytak nic
  nie traci, otwarty nie drgnie, a przy złym zgadnięciu szczęki nie zatrzaskują
  się w pierwszej ramce), zamykanie zawsze rampą.
- **Chwytak przez XCAN (macOS) rusza.** galaxeo twierdziło, że chwytak
  ignoruje ramkę `0x051` z XCAN (10 bajtów idzie jako 12, DLC 9). CAN FD nie ma
  ramki 10-bajtowej, więc SocketCAN na Linuksie też wysyła 12 bajtów, a
  2026-09-27 pomiar to obalił: `python -m galaxeo.xcan_usb --grip-test`
  (galaxeo PR 20) przesunął szczęki przez XCAN („Chwytak A1X: pomiar” wyżej).
  Chwytak przez XCAN jest więc domyślnie włączony (`a1x.gripper_over_xcan:
  true`); `false` przywraca odmowę („no gripper”). Przejściówkę sprawdzisz tym
  samym testem przy zamkniętym panelu (jedną przejściówkę otwiera jeden proces).
  Od pierwszej `0x051` panel pokazuje, gdzie szczęki naprawdę są.

### Jednostki

| | aplikacja | bliźniak (MJCF) | ramię (CAN) |
|---|---|---|---|
| stawy ramienia | stopnie | radiany, te same kąty | radiany + `joint_offsets_deg` |
| chwytak | 0..100 (0 = zamknięty) | wysuw szczęki 0..0,05 m (0 = płytki zsunięte, czubki 3 mm od siebie); druga szczęka lustrzanie | `p_des` [rad] liniowo +0,08 (0) … −1,95 (100); pomiar grupy 7 `0x052` [rad]: +0,039 (0) … −1,992 (100) |

Chwytak A1X nie ma tików: klucze `center_ticks` i `gripper_*_ticks` sekcji
`feetech:` dotyczą tylko SO-101. Progi szczęk w metrach (`jaw` w `robot.yaml`): 20 mm wysuwu to
43 mm między czubkami, więcej niż przekątna kostki 3 cm (42 mm).

### Do sprawdzenia na pierwszej sesji

| pytanie | skąd wątpliwość | jak sprawdzić |
|---|---|---|
| ~~Czy chwytak słucha `0x051` przez XCAN (macOS)?~~ | **Tak**, zmierzone 2026-09-27 (`--grip-test`); domyślnie włączone (`a1x.gripper_over_xcan`, PR 18) | — |
| ~~Czy grupa 7 ramki `0x052` to pozycja chwytaka?~~ | **Tak**, w rad jak `p_des` (2026-09-27, „Chwytak A1X: pomiar” wyżej) | — |
| Docisk 0,04 rad przy zamknięciu i zapas przy otwarciu | końce ustawione z pomiaru ograniczników, nie sprawdzone jeszcze na stole | zamknij pusty: grupa 7 ok. +0,039, \|effort\| blisko spoczynku, szczęki się stykają; otwórz: ok. −1,95, effort blisko spoczynku |
| Jednostka effort i próg `gripper_force` 1,2 | skala `/600` bez znanej jednostki; próg empiryczny, miażdżenie od ok. 6 | chwyt kostki 3 cm: cel ma zamarznąć przy kontakcie. Sprawdź też, czy effort chwytaka nie stoi na 0,00 (chwytak głuchy — tego backend nie wykrywa) |
| Czy `kp` 20 nadąża przy 90°/s? | `kp` jest ignorowane (zmierzone przy małych skokach), szybkie ruchy nie były sprawdzane z bliźniakiem | Dom i fala: bez błędu rozjazdu (25° / 0,5 s), suwaki idą za ramieniem |
| Próg „głuche albo zablokowane” | 3° przez 1 s przy stojącym stawie; staw oparty o przeszkodę też go wyzwoli | czy nie strzela przy spokojnej pracy; w razie potrzeby `deaf_err_deg` / `deaf_s` |
| Zero J3 i chwytaka | +1,49° z ramienia galaxeo; zamknięty ogranicznik szczęk +0,039 rad (2026-09-27), ale czy płytki się przy nim stykają, nie sprawdzono | `joint_offsets_deg` wyżej; przy 0 szczęki mają się stykać |

### A1X w symulacji

Bez sprzętu:

```bash
lerobot-twin demo --robot a1x               # stanowisko z dwiema kamerami sim, panel: Ramię sim, Połącz
lerobot-twin calib-sim --robot a1x --n 3 --cameras 2
```

Model generuje `scripts/import_a1x_model.py` z URDF producenta. Reguły
przetwarzania są w `src/lerobot_mp/robot/a1x/robot/README.md`, każda z testem. Szczęka 2
idzie za szczęką 1 przez równość w MJCF, usztywnioną regułą 6 (`solref`,
`solimp`). Wyniki z `tests/test_twin_a1x_sim.py` i `tests/test_twin_calib.py`:

| | wynik |
|---|---|
| kalibracja kamer, `calib-sim --robot a1x --n 3 --cameras 2` | 6/6 zaufanych; mediana 0,24 mm / 0,017°, najgorzej 0,82 mm / 0,041°, residuum 0,21 px |
| sesja kalibracji, ziarno 0, 1 kamera | 16 póz, 0,20 mm, 0,017° |
| kamera z boku (obrót ~180° względem podstawy), ziarno 1, 2 kamery | 2/2 zaufane (przed poprawką mediany obrotów w `calib/handeye.py`: 0/2) |
| fala kalibracyjna: pozy z kartą nad stołem | 85 % (170/200) |

---

## Polecenia

| | |
|---|---|
| `lerobot-twin check` | czy środowisko jest gotowe (bez renderu) |
| `lerobot-twin demo` | przykładowe stanowisko w symulacji z pierwszym zainstalowanym ramieniem → `workspace/twin.json` (`--robot`, `--force`, `--path`) |
| `lerobot-twin demo --robot a1x` | to samo z ramieniem Galaxea A1X (`examples/twin.a1x.sim.json`) |
| `lerobot-twin ui` | panel w przeglądarce |
| `lerobot-twin --config configs/local.yaml <polecenie>` | konfiguracja ramienia z pliku (sekcje backendów, np. `a1x:`) |
| `lerobot-twin card --tag-mm 50` | arkusz karty kalibracyjnej (PNG, A4) |
| `lerobot-twin board --square-mm 28` | tablica ChArUco do intrynsyk (PNG, A4) |
| `lerobot-twin calib-sim --n 5 --cameras 2` | kalibracja w symulacji, oceniana względem prawdy, na pierwszym zainstalowanym ramieniu; `--robot a1x` na A1X |
| `lerobot-twin workspace` | co wiadomo o stanowisku |
| `lerobot-twin export <folder>` | stanowisko jako dane: `scene.xml` + `assets/` + `station.json` — dla pakietów, które jadą na zmierzonym stanowisku bez importu bliźniaka (trening) |

```bash
pytest -q                                          # całość, z renderem
pytest -q -m "not render"                          # maszyna bez OpenGL
pytest -q -m "not cli"                             # bez testów CLI w podprocesie (kilka sekund)
pytest -q -n auto --dist loadfile                  # równolegle, plik na jednym procesie (pytest-xdist)
```

Testy domu A1X i `Twin.move` jadą na zegarze wirtualnym
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
├── base.py, factory.py   kontrakt backendu (re-eksport z `lerobot_mp_api`) i rejestr wtyczek
└── a1x/                  pakiet A1X (wyjdzie do galaxeo)
    ├── robot/            robot.yaml, URDF i siatki producenta (NIE Apache-2.0, NOTICE), a1x.xml ze skryptu
    └── backend.py        backend a1x (CAN-FD przez pakiet galaxeo)

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
