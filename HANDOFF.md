# Handoff: cyfrowy bliźniak stanowiska (SO-101, Galaxea A1X)

Instrukcja uruchomienia i procedura testu na sprzęcie:
**[docs/TWIN.md](docs/TWIN.md)**.

## W skrócie

System jest gotowy do pierwszego testu na fizycznym stanowisku. Wszystko,
co da się sprawdzić bez ramienia i kamer, jest sprawdzone na bliźniaku —
tą samą ścieżką kodu, którą pójdzie sprzęt:

- **panel webowy** (`twin ui`) — ramię, kamery (także symulowane,
  stawiane i przeciągane w 3D), kalibracja, mapa na żywo, trening, polityki,
  porównanie sim-real; przećwiczony cały scenariusz w przeglądarce;
- **kalibracja**: intrynsyki z ChArUco, położenie wielu kamer z karty w
  chwytaku, szybka relokalizacja, wykrywanie przestawionej kamery;
- **trening na GPU**: MuJoCo Warp (tysiące światów) + PPO; polityki
  przechodzą na CPU (inny silnik) z 96–100% sukcesu;
- **kalibracja treningu sim-to-real**: identyfikacja dynamiki serw z nagrania
  ruchu, randomizacja wokół zmierzonej dynamiki;
- **percepcja**: kostka z kamer (część wspólna masek na wysokości górnej
  ściany), śledzenie w dłoni, gdy szczęki ją zasłaniają;
- **runner**: polityka na bliźniaku albo ramieniu przez `SafetySupervisor`,
  z zatrzymaniem przy rozjeździe i po wykonaniu zadania; `lift-v3` z kostką
  **wyłącznie z kamer** podnosi ją na bliźniaku 8/8 (`lift-v1`, uczona na
  prawdzie: 1/8);
- **polityki bazowe** w `assets/policies/` (`reach-v3`, `lift-v3`) — start do
  douczania na zmierzonej dynamice;
- **drugie ramię — Galaxea A1X**: opis ramienia i model z URDF producenta,
  kalibracja kamer, IK, fala i środowiska RL w symulacji; backend `a1x` po CAN-FD
  przetestowany na udawanym ramieniu — sprzęt jeszcze nie. Pakiet ramienia jest
  w galaxeo (`galaxeo/robot/`, galaxeo-manipulators), bliźniak znajduje go przez
  entry points;

## Repozytorium na innym komputerze

Klon ma wszystko do uruchomienia — instalacja per platforma:
README, „Szybki start na nowym komputerze”.

- **W repozytorium:** robot przykładowy (`examples/robots/testbot`),
  polityki bazowe (`assets/policies`),
  przykładowe stanowisko w symulacji (`examples/twin.sim.json`,
  `twin demo`), sprawdzone wersje pakietów (`constraints.txt`),
  licencje plików osób trzecich (`NOTICE`).
- **Poza repozytorium (per maszyna):** `workspace/` (stanowisko i własne
  polityki), `configs/local.yaml` (sekcje backendów, np. `a1x`, i reszta
  konfiguracji ramienia, `twin --config …`), pakiet `galaxeo` (ramię A1X, obok bliźniaka), `.claude/`, cache kerneli Warp.
- **Ramię SO-101** i jego backendy (`feetech`, `lerobot`, most COM<->TCP) to
  pakiet `soarm` (repozytorium so-arm101), instalowany obok bliźniaka.
- Domyślne ścieżki (`workspace/…`) z klonu liczą się od katalogu
  repozytorium (`src/machinekind_twin/paths.py`), więc katalog uruchomienia nie ma
  znaczenia; bez klonu — od katalogu bieżącego.
- `twin check` mówi osobno, czy komputer nadaje się do treningu
  (torch z CUDA, warp, mujoco_warp). Bez NVIDIA działa wszystko poza treningiem.

---

## 1. Co jest gotowe

| moduł | co robi | dowód |
|---|---|---|
| `scene.py` | + panele z obrazem, czujniki kontaktu szczęk; **punkt główny bez fałszywego półpiksela** | test na maskach segmentacji; kalibracja 4× dokładniejsza w obrocie |
| `calib/intrinsics.py` | ChArUco: arkusz, zbieranie ujęć z pilnowaniem różnorodności, K + dystorsja, bramki | K z renderu do ~1 px, rogi 0,002 px od rzutu |
| `perception.py` | mapa stołu na żywo, kostka z kamer, `CubeTracker` (w dłoni) | 1,2 mm / 0,6° z dwóch kamer |
| `rl/task.py` | obserwacja/akcja/nagroda — numpy i torch, te same wzory | test równości numpy/torch |
| `rl/env.py` | Gymnasium CPU, `check_env` | ekspert IK rozwiązuje `reach` 10/10 |
| `rl/batch.py` | MuJoCo Warp, randomizacja per świat + `set_const` | zgodność z CPU 1e-7 → 4e-3 po 60 krokach losowych |
| `rl/ppo.py`, `policy.py`, `evaluate.py` | PPO (także douczanie `--init` z zapisanym krytykiem), plik polityki z zadaniem, krytykiem i wynikami, ewaluacja na CPU | `reach-v3`, `lift-v3` w `assets/policies`; `tests/test_twin_policies.py` |
| `rl/sysid.py` | nagranie ruchu przez nadzór, dopasowanie tłumienia/armatury/opóźnienia (kp i tarcie suche z tego ruchu niewyznaczalne — zostają z modelu, trening losuje je szerzej) | ruch z panelu, 13 ramion spoza siatki startowej: tłumienie ≤ 10,5%, armatura ≤ 8,2%, opóźnienie ≤ 1,9 ms; przedział niepewności zawsze obejmował prawdziwy błąd |
| `rl/runner.py` | polityka na `Twin`, stop przy rozjeździe > 25° (w zmierzonej pozie), stop po wykonaniu zadania, czekanie na kostkę | test na bliźniaku; pełny łańcuch z kamerami w czasie symulowanym (`test_twin_lift_from_cameras`) |
| `ui/*` | panel (viser), zadania w tle, most sceny, strażnik kamer | test startu panelu + scenariusz w przeglądarce |
| `runtime.py` | pętla ramienia: jeden właściciel ruchu, STOP/Dom w zmierzonej pozie, reakcja na błędy serw i utratę łącza, render bez blokady pętli | `tests/test_twin_runtime.py` (czas symulowany) |
| `robots.py` (`RobotSpec`) | bliźniak niezależny od ramienia: stawy, limity, dom, progi szczęk, skala biurka, zakres chwytaka i szczęka lustrzana z opisu ramienia | SO-101 bit w bit bez zmian (łańcuch `lift` z kamer i hash epizodów środowiska przed i po); `tests/test_twin_home_stages.py` (dom etapami na robocie testowym) |
| `galaxeo/robot/` w galaxeo-manipulators | pakiet A1X poza bliźniakiem: opis ramienia, model z URDF producenta, przykładowe stanowisko, backend `a1x` po CAN-FD (strumień od zmierzonej pozy z limitem 90°/s, nic przed pierwszym rozkazem, nigdy FF 2 ani `kp` 0, włączenie FF 1-5-6 tylko przyciskiem) | testy modelu i backendu w galaxeo; tu `tests/test_a1x_in_twin.py` (z galaxeo obok bliźniaka): `Twin` na udawanej magistrali; kalibracja w symulacji 6/6 zaufanych (0,24 mm / 0,017° mediany), IK 20/20 (0,33 mm) |
| `tagbody.py` | **bryła z kilkoma AprilTagami**: układ tagów uczony z kadrów kamer stanowiska (pary w jednym kadrze + wspólne dopasowanie wszystkich rogów), poza bryły z kamer z pozą. Dziś nic w panelu go nie używa | `tests/test_twin_tagbody.py` (układ 0,1 mm, poza 1 mm / 1° od prawdy w symulacji) |

Testy: `pytest -q`. Zestaw to wyłącznie testy integracyjne (łańcuch przez kilka
modułów tak, jak jedzie produkcja) zachowań groźnych dla sprzętu i człowieka,
psujących stanowisko albo przeniesienie sim-real.

---

## 2. Decyzje i dlaczego

- **Jedna definicja zadania dla CPU, GPU i ramienia** (`rl/task.py` z `xp` =
  numpy albo torch). Dwie kopie „prawie tej samej” obserwacji to ciche źródło
  sim-2-real.
- **Akcja jako przyrost celu stawów, chwytak bezwzględnie**, oba ograniczone
  tak jak `SafetySupervisor` — w treningu polityka nie może robić tego, czego
  nadzór na ramieniu nie przepuści.
- **Trening w osobnym procesie** — CUDA przez kilkanaście minut; padnięcie
  sterownika nie może zabrać panelu ani pętli ramienia. Zatrzymanie plikiem
  `STOP` w katalogu przebiegu, polityka z tego miejsca zostaje zapisana.
- **Kamera symulowana ma prawdziwą pozę (`sim_pose`) osobno od kalibracji** —
  kalibruje się ją jak prawdziwą i widać błąd względem prawdy.
- **Kostka na mapie górnej ściany, część wspólna masek** — średnia kamer
  rozmazywała boki (12–17 mm), część wspólna daje 1–2 mm.
- **Śledzenie kostki w dłoni** — wizja gubi kostkę dokładnie przy chwycie;
  szczęka zatrzymana na obiekcie + kostka ostatnio przy szczękach = jedzie z TCP.
- **Jeden wątek renderujący** w `Twin` — kontekst GL może być bieżący tylko w
  jednym wątku; wcześniej każdy wątek (panel, fala, mapa) renderował sam.
- **Trening widzi kostkę tak jak percepcja, nie jak fizyka.** Zmierzone na
  CPU: `lift` uczona na prawdzie — 20/20 na prawdzie, 19/20 ze złożonym
  obrotem, 19/20 z szumem 3 mm, **7/20** z odświeżaniem 10 Hz i opóźnieniem
  150 ms. Stąd `Randomization.cube_*`; `lift-v2` jest uczona już z nimi.
- **Kostka z kamer = dopasowanie sylwetki bryły**, a nie sama część wspólna
  masek — ta druga potrzebuje dwóch kamer, a ramię zasłania kostkę jednej
  z nich w połowie epizodów.
- **Jeden właściciel ruchu ramienia** (`panel`, `polityka`, `kalibracja`,
  `identyfikacja`) zamiast wielu wątków piszących cel na zmianę. STOP, Dom,
  Połącz i Rozłącz odbierają ramię każdemu przez `Twin.preempt`.
- **Błąd serwa ramienia = STOP, błąd samego chwytaka = mniejszy docisk.**
  Weryfikacja pokazała, że STOP po przeciążeniu chwytaka puszczał chwyt i
  kostka spadała; teraz docisk maleje do ~5° ciaśniej niż zmierzona szczęka,
  STOP dopiero po 2 s błędu. STOP i Dom nie puszczają ściskającego chwytaka.
- **Strażnik rozjazdu dla każdego właściciela** (25° przez 0,5 s) — serwo,
  które zwiotczało bez bitu błędu, kończyło `move` „normalnie” 63° od celu.
- **Koniec epizodu po sukcesie liczony jak limit czasu** — PPO dalej
  bootstrapuje wartość stanu; zero na końcu uczyło polityki wisieć tuż pod
  progiem sukcesu.

---

## 3. Następne kroki

1. **Test na sprzęcie** według [docs/TWIN.md](docs/TWIN.md) — kroki 0–8.
   Najpierw połączenie i suwaki, potem intrynsyki, położenie, sim-real,
   identyfikacja, dopiero potem polityki.
2. **Po identyfikacji douczyć polityki** (`--init`, w panelu *Start z
   polityki*) — domyślne są uczone wokół modelu Menagerie. Douczanie zamiast
   uczenia od zera: `lift` z modelem percepcji od zera po 220 iteracjach miała
   0% sukcesu, a douczana z `lift-v1` startuje od umiejętności chwytu.
   **Przy pierwszym połączeniu sprawdzić zero chwytaka**: przy 0 szczęki mają
   się stykać. Na SO-101 tiki chwytaka ustawia sekcja `feetech:` backendu
   z pakietu `soarm`.
3. **Obserwacje obrazowe w RL** — renderer MuJoCo Warp (wsadowy, na GPU)
   zgadza się z OpenGL do 0,05 px; środowisko obrazowe to kolejny krok
   (mapa stołu z kamer jako wejście, jak w galaxeo).
4. **Kalibracja oko-w-dłoni** (`wrist_cam`) — solver odwrócony: karta na stole.
5. **Rekonstrukcja 3D przedmiotów** — [docs/RESEARCH_3D.md](docs/RESEARCH_3D.md);
   MuJoCo Warp ma render gaussian splats, co pozwala wstawić zeskanowany
   przedmiot do renderu treningowego.
6. **Pierwsza sesja na Galaxea A1X** według [galaxeo/robot/README.md w galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators/blob/main/galaxeo/robot/README.md),
   „Pierwsza sesja krok po kroku”: *Połącz* bez domu, zmierzyć `a1x.joint_offsets_deg`
   i `a1x.gripper_fb_closed/open`, mały ruch, Dom, złożenie przed wyłączeniem, potem
   kamery i fala. Po drodze rozstrzygnąć tabelę „Do sprawdzenia na pierwszej sesji”
   (chwytak przez XCAN, grupa 7 jako pozycja chwytaka, jednostka effort, nadążanie
   przy 90°/s).
7. **`pickplace` na GPU** (zadanie scalone, ekspert IK rozwiązuje je na CPU na obu
   ramionach): najpierw 5 linii próby `mujoco_warp.put_model` z
   `galaxeo/robot/a1x/a1x.xml` z galaxeo (równość szczęk, bryły z siatek) i pomiar
   `nconmax/njmax` dla A1X (w `batch.py` wpisano nie zmierzone 64/384; na CPU szczyt
   12 kontaktów / 49 wierszy), potem porównanie GPU z CPU
   (test `test_gpu_pickplace_matches_the_cpu_env` usunięty przy odchudzaniu — odtworzyć
   z commitu `b2ba4bb` na maszynie z CUDA; miał tolerancję 5e-3 dobraną bez
   sprawdzenia; `mocap_pos` per świat przez `wp.to_torch` też nie był sprawdzony),
   krótki przebieg `twin train --task pickplace --envs 256 --iters 3
   --eval-episodes 2` i dopiero pełny trening na SO-101 i A1X. Z polityką
   `pickplace-v1` w `assets/policies/` przestaje się pomijać
   `tests/test_twin_pickplace_from_cameras.py`.
8. Solver kalibracji bez odpornej funkcji straty — jedna zła obserwacja potrafi
   podnieść residuum do 1 px (ziarno 4 w `calib-sim`); warto dodać odrzucanie.

---

## 4. Pytania do rozstrzygnięcia

1. **Licencja galaxeo-manipulators** — `calib/tags.py`, `calib/handeye.py`
   i idea `calib/topdown.py` przeniesione z atrybucją; zakładamy zgodę machinekind.
2. Które zadania po `reach` i `lift`: pick-and-place, nalewanie, sortowanie?
3. **Licencja URDF i siatek A1X** i **`galaxeo.xcan_usb` (GPL-2.0)** — wyszły z tego
   repozytorium razem z pakietem A1X: bliźniak nie zawiera ani nie importuje niczego
   z galaxeo. Oba pytania zostają w galaxeo-manipulators.

---

## 5. Pułapki, które cicho psują sim-2-real

Każda jest pilnowana testem albo poprawiona w kodzie:

- **Znak punktu głównego w MuJoCo** — przy odwrotnym kadr przesuwa się o
  dwukrotność przesunięcia (62 px przy 30 px).
- **Fałszywy półpiksel w pionie** — dawna „kompensacja” wynikała z pomiaru
  na środkach czerwonych plam, które podnosi cień dolnej połowy kulki. Mierzyć
  geometrię trzeba na **maskach segmentacji**, nie na kolorze.
- **Półpiksel w detektorze AprilTag** — `CORNER_REFINE_APRILTAG` oddaje rogi
  w konwencji narożników pikseli. Dotyczy też galaxeo.
- **Zmiana masy w skompilowanym modelu bez `mj_setConst`** — stałe kontaktu
  (`body_invweight0`) zostają od starej masy; kostka ściśnięta między szczęką
  a blatem wylatywała z prędkością 10–180 m/s. Na GPU: `mjw.set_const`.
- **Torch z PyPI na Windows nie ma CUDA** — koło z indeksu PyTorcha.
- **`CameraStream` domyślnie odbija obraz lustrzanie** (`CameraConfig.mirror`);
  bliźniak otwiera kamery z `mirror=False`.
- **`MjSpec.attach` gubi opcje fizyki ramienia**, jeśli scena ich nie przejmie.
- **Nadzór dociągał ramię do limitów z konfiguracji** przy połączeniu (skok
  przy `shoulder_lift −101°`) — teraz ramię zostaje, a w zakres wraca powoli
  po włączeniu sprzęgła.
- **Chwytak 0..100 przeliczany inaczej w bliźniaku i w backendzie** — ta sama
  szczęka była w symulacji 1,8× szerzej otwarta; jedno przeliczenie przez tiki.
- **Obserwacja z kinematyki sprzed kroku fizyki** — TCP 7 mm od FK(q); po
  `mj_step` trzeba `mj_kinematics` (na GPU `mjw.kinematics`).
- **`cv2.phaseCorrelate` z oknem (OpenCV 5.0) mnoży wejście przez okno w
  miejscu** — kadr odniesienia strażnika kamer blakł co 2 s; przekazywać kopie.
- **Test w czasie zegarowym na obciążonej maszynie** — `lift` z kamer bywał
  czerwony pod obciążeniem; testy pętli idą teraz w czasie symulowanym
  (`Twin.step`, `PolicyRunner.step_once`).
- **Panel: poza każdego geomu 30×/s zrywała połączenie przeglądarki** (~3000
  komunikatów/s) — most sceny wysyła pozy ciał i tylko przy zmianie, a siatki
  do widoku są uproszczone (348 → 43 tys. trójkątów, 0,3 mm obrysu).
- **Viser w ukrytej karcie przeglądarki nic nie rysuje** — klient przetwarza
  komunikaty w pętli animacji; „pusty panel” w tle to nie błąd serwera.
- **Polityka nauczona na prawdziwej pozycji obiektu nie przeżywa opóźnienia
  percepcji** — patrz wyżej; każdy nowy obiekt z kamer musi mieć swój model
  percepcji w treningu.
- **Mediana obrotów po wektorach Rodriguesa w układzie globalnym** — dla kamery
  obróconej o ~180° względem podstawy (A1X, kamera z boku) przybliżenia leżą po
  obu stronach zawinięcia ±π (wektory ~+v i ~−v), a ich mediana to obrót ~180° od
  każdego z nich. Poza zgrubna w fali wychodziła odwrócona, żadna poza nie trafiała
  w kadr, 0/2 kamer zaufanych. `calib/handeye.py` liczy teraz medianę w przestrzeni
  stycznej jednego z wejść, potem drugi raz względem wyniku
  (`test_calib_session_trusts_a_side_camera_on_the_a1x`). Ten sam błąd jest w
  `sim/calib/handeye.py` galaxeo-manipulators.
- **Dopasowanie modelu „obiekt leży na blacie” do obiektu w powietrzu daje
  pewne siebie bzdury** — bramka na jakość dopasowania (IoU), a obiekt w dłoni
  z kinematyki.
