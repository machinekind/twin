# Kampania chwytania — 26 września 2026, 04:39–07:39 CEST

Cel: rozszerzyć politykę SO-101 na 12 rodzin przedmiotów, zwiększyć odporność
na rozjazdy obserwacji i dynamiki oraz premiować płynne polecenia ruchu.
Nie obiecujemy „maksymalnej” odporności bez pomiarów prawdziwego stanowiska.
Trening i testy tej kampanii dotyczą symulacji.

| Okno | Zadanie | Warunek oceny |
|---|---|---|
| 04:39–05:09 | Modele, obserwacje kształtu i wieku detekcji, randomizacja kalibracji, nagrody i testy CPU/GPU | Wszystkie geometrie kompilują się, pozostają stabilne; stare polityki zachowują format |
| 05:09–05:44 | Curriculum: kostka, płytka, trójkąt, walec, kombinerki i śruba; małe zakłócenia | Skuteczność osobno dla każdej rodziny, brak zaliczania przedmiotu opartego o stół |
| 05:44–06:39 | Wszystkie 12 rodzin; rozmiary 85%, 100%, 115%; średnie zakłócenia | Najgorsza rodzina, średnia skuteczność, płynność |
| 06:39–07:14 | Trudniejsze połączone zakłócenia, płynność i utrzymywanie chwytu | Checkpoint nie jest promowany wyłącznie za wysoką nagrodę treningową |
| 07:14–07:39 | Niewidziane seedy, niewidziany rozmiar 108%, profile nominalny/średni/trudny | Zapis wyników każdej próby, rzeczywistych pochodnych ruchu i ograniczeń |

Czas jest budżetem ściennym, nie obietnicą określonej liczby iteracji.
Proces kończy blok po najbliższej iteracji po terminie. Niepełna ewaluacja
zostaje oznaczona jako niepełna; nie dopisuje się brakujących sukcesów.

## Przedmioty — modele parametryczne, nie skany

| Rodzina | Przybliżone wymiary nominalne | Masa |
|---|---|---:|
| Kostka | 30 × 30 × 30 mm | 30 g |
| Płytka | 52 × 36 × 8 mm | 25 g |
| Trójkątny graniastosłup | 44 × 42 × 18 mm | 25 g |
| Walec stojący | Ø26 × 48 mm | 35 g |
| Zamknięte kombinerki | ok. 130 × 44 × 16 mm; sztywny model złożony | 110 g |
| Śruba leżąca | trzon Ø8 × 46 mm, łeb Ø16 × 6 mm | 18 g |
| Nakrętka | ok. 34 × 34 × 12 mm; rzeczywisty otwór w colliderze | 25 g |
| Podkładka | ok. Ø44 × 5 mm; otwór w colliderze | 12 g |
| Kątownik L | ok. 54 × 54 × 12 mm | 35 g |
| Element T | 50 × 60 × 14 mm | 35 g |
| Kapsułka leżąca | 68 × 24 × 24 mm | 30 g |
| Klin | 50 × 32 × 20 mm | 30 g |

Pozycja oraz obrót na stole są losowe. Śruba ma fizyczny łeb i poziomy
trzon: może się toczyć i przechylać podczas osiadania. Nie modelujemy gwintu
ani ruchomego przegubu kombinerek. Masa wariantu rozmiaru skaluje się z
objętością, a dodatkowa randomizacja masy obejmuje różne gęstości.

Polityka ma wspólny aktor dla rodzin, dodatkowe wejścia typu i wymiarów oraz
wieku detekcji. Pierwsze 33 wejścia odziedziczono z polityki kostki; nowe
kolumny wag startują od zera. To zachowuje początkową funkcję aktora,
ale nie oznacza, że potrafi od razu chwytać nowe przedmioty.

## Scenariusze sim-to-real

| Rozjazd | Co robimy w kampanii | Granica wniosku |
|---|---|---|
| Błąd położenia kamery / podstawy | Stały błąd XYZ na epizod: ±1, ±3, ±6 mm; profil ekstremalny ±10 mm | To zakłócenie wyjściowej pozy obiektu, nie pełna symulacja procesu kalibracji |
| Błąd orientacji układu | Stały obrót XYZ: ±0,25°, ±0,75°, ±1,5°; ekstremalnie ±3° | Nie naprawia źle skalibrowanej kamery |
| Błąd skali / intrynsyk | Skala oszacowanej pozycji ±1–2%, skrajnie ±3% | Nie modeluje wszystkich zniekształceń obiektywu |
| Drganie detekcji | Szum XY do 3 mm i Z do 2 mm w profilu trudnym | Brak rzeczywistych refleksów, rozmycia i tekstur obrazu |
| Nieaktualna obserwacja | Opóźnienie do 4 taktów, aktualizacja co 1–4 takty | Takty mają 50 ms; trzeba mierzyć opóźnienie prawdziwej wizji |
| Krótkie zasłonięcie obiektu | Losowe zaniki na 1–4 takty; wejścia wieku i świeżości | Dłuższa utrata wymaga zatrzymania, a nie zgadywania |
| Długa utrata detekcji | Po 0,3 s bez świeżej pozy zamrożenie celu wszystkich stawów, zachowanie chwytaka | Dotyczy środowiska nowej polityki; integracja z ogólnym detektorem pozostaje do wykonania |
| Opóźnienie napędów | Losowe opóźnienie akcji 0–2 takty; profil ekstremalny 0–3 | Brak dowolnych awarii komunikacji w tym modelu |
| Przesunięcia enkoderów | Stały błąd odczytu do ±0,3°, skrajnie ±0,6° | Przybliżenie błędu obserwacji, nie pełna nowa kinematyka ramienia |
| Inne serwa / tarcie / bezwładność | Randomizacja kp, tłumienia, armatury i tarcia suchego; zakresy repo poszerzone do 1,5× | Parametry mnożnikowe nie zastępują identyfikacji konkretnego ramienia |
| Inna masa i materiał przedmiotu | Masa i bezwładność skalowane razem, losowe tarcie wszystkich części obiektu | Wartości startowe są założeniami |
| Nietypowy chwyt i przewężenia | Obiekty złożone, otwory, różne yaw, rozmiary i pozycje | Sztywne przedmioty; brak deformacji, kabli, tkanin |
| Fałszywy sukces długiego obiektu | Wymagamy uniesienia najniższego punktu całego obiektu i kontaktu obu szczęk | Nie wystarczy wysoko położony środek przedmiotu |
| Przeskoki i wymachiwanie | Koszty prędkości, przyspieszenia i jerk poleceń, koszt ruchu po chwycie, kara przy limitach | Sprawdzamy też zmierzone ruchy w symulacji; nie tylko polecenia |
| Zmiana silnika fizyki | Trening Warp float32, walidacja MuJoCo CPU | Nie jest dowodem działania na sprzęcie |
| Zbyt trudne zakłócenia naraz | Stopniowe zwiększanie trudności; oddzielny profil ekstremalny do diagnozy | Nie wszystkie kombinacje są fizycznie chwytalne |
| Refleksy metalu, ciemne obiekty, brak kontrastu | Wymagają osobnej walidacji detektora obrazowego | NIE są objęte skutecznością tej polityki na zaszumionych pozach |
| Nieznany typ, błędny rozmiar lub błędna identyfikacja | Zapisana jako granica adaptera percepcji | Obecny trening dostaje poprawny typ i wymiary; nie uczy rozpoznawania obiektów |
| Stół na innej wysokości, ruchoma podstawa | Częściowo przez bias pozy; potrzebna ponowna kalibracja geometrii | Nie zmieniamy fizycznej wysokości stołu w treningu tej wersji |
| Kontakt z człowiekiem / nieznaną przeszkodą | Istniejący nadzór ruchu pozostaje wymagany | NIE objęte treningiem ani certyfikacją bezpieczeństwa |
| Przeciążenie, temperatura, awaria serwa lub łącza | Istniejące STOP, watchdog i nadzór; wymagają testów stanowiskowych | Nie ma symulatora uszkodzeń elektroniki |

Nowy format polityki jest jawnie blokowany w starym `PolicyRunner`, ponieważ
stary detektor rozpoznaje kostkę i nie dostarcza typu oraz wymiarów wszystkich
obiektów. Nie uruchamiamy na ramieniu wejść, których znaczenie się nie zgadza.

## Płynność i wybór modelu

Koszt ruchu liczymy z przyrostów **faktycznie wysłanych celów**, po opóźnieniu,
przycięciu do limitów i zatrzymaniu przy nieświeżej obserwacji. Prędkość,
przyspieszenie i jerk są w rad/s, rad/s² i rad/s³, z ustalonymi skalami
normalizacji. Chwytak wyłączono z tych trzech kar, żeby nie premiować
poluzowania chwytu. Niezależnie działają istniejące koszty zmian akcji,
bliskości limitów oraz ruchu po sukcesie.

Każdy checkpoint przechodzi walidację 12 rodzin. Ranking:
1. skuteczność najgorszej rodziny w warunkach nominalnych,
2. średnia skuteczność rodzin i profili,
3. niższy jerk poleceń jako rozstrzygnięcie remisu.

Test końcowy: nowe seedy 80000+, rozmiar 108% niewykorzystywany do treningu,
do 50 epizodów na każdą rodzinę i profil, łącznie 1800. Seedy 30000+ służą
wyłącznie do walidacji i doboru checkpointu. Nie zmieniamy doboru po wyniku
końcowym. Raportujemy osobno porażki, pauzy na brak detekcji, ruch przy
limitach, czas, jerk poleceń i jerk rzeczywistych stawów.

## Monitorowanie i filmiki

- Proces: `scripts/campaign_multi.py`; status i logi w
  `workspace/campaigns/20260926_3h/`.
- Luna monitoruje odczytowo etapy, błędy, zastój i GPU.
- Wspólny folder filmów: `D:\digital-twin-training\sukcesy`.
- Tylko wyraziste sukcesy: pierwszy dla danej rodziny/scenariusza lub ruch
  o co najmniej 20% niższym jerk niż poprzedni film tej kategorii. Maks. 80 klipów.
- Filmy są odtworzeniem rzeczywiście zarejestrowanych konfiguracji udanego
  epizodu CPU; mają podpis „SIMULATION”. Nie są generowane ani retuszowane.
- Plik `STOP` w katalogu kampanii kończy trening po bieżącej iteracji;
  checkpoint zostaje zapisany. Końcowa ewaluacja może jeszcze działać.

Podstawą podejścia jest [randomizacja dynamiki w sim-to-real](https://arxiv.org/abs/1710.06537).
Geometria kontaktowa respektuje ograniczenia wypukłych colliderów opisane
w [dokumentacji MuJoCo](https://mujoco.readthedocs.io/en/stable/computation/index.html).
Zakresy liczbowe powyżej są doborem dla tej kampanii, nie wartościami
udowodnionymi dla fizycznego stanowiska.
