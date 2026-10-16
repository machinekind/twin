"""Przewodnik panelu: gdzie operator jest w procedurze pierwszego testu i co zrobic teraz.

Czysta logika bez visera - `GuideSnapshot` (stan panelu i ramienia) na wejsciu,
`Guide` (kroki 0-5 z docs/TWIN.md, jedna nastepna akcja, blokady) na wyjsciu.
Panel (`app.TwinApp._tick_guide`) tylko sklada migawke i wyswietla `render_now`
(zawsze widoczne "Teraz" i uwagi, ktorych nie wolno schowac: brak hamulcow, blad
ramienia, backend lerobot) oraz - w zwinietym folderze "Uwagi (n) i kroki 0-5" -
`render_notes` (pozostale uwagi, tryb) i `render_steps`. `render_head` sklada obie
czesci (uzywa go `render`).

Kroki sa te same w symulacji i na prawdziwym ramieniu - w symulacji z kamerami
symulowanymi (znaja swoje K, wiec krok 2 odpada; Sim-Real zastepuje blad
wzgledem prawdy w zakladce Kamery).

Styl akcji: "Zakladka: ..., **Przycisk**" - nazwy przyciskow pogrubione jak w panelu;
"Teraz" z kroku zaczyna sie od "Krok N/5 (tytul):". Zakladka Kalibracja pokazuje akcje
krokow 1-3 we wlasnej linii "Teraz" (bez przedrostka zakladki).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DONE, CURRENT, TODO, SKIP = "gotowe", "teraz", "do zrobienia", "nie dotyczy"
#: Znacznik kroku w liscie (czysty tekst - markdown panelu nie ma pol wyboru).
MARK = {DONE: "`[x]`", CURRENT: "`[>]`", TODO: "`[ ]`", SKIP: "`[-]`"}
#: Mediana rozjazdu krawedzi Sim-Real [px], powyzej ktorej cos jest zle (TWIN.md, krok 4).
SIMREAL_MAX_PX = 3.0
PANEL = "panel"
LAST_STEP = 5
#: Dluzszy opis zrobionego kroku nie trafia do listy (jest w `Step.detail`).
DONE_DETAIL_MAX = 48

#: Jak zatrzymac zadanie, ktore ma ramie (wlasciciel `Twin.owner` / `TwinApp._busy`).
HOW_TO_STOP = {
    "kalibracja kamer": "Kalibracja > Zatrzymaj",
    "kalibracja": "Kalibracja > Zatrzymaj",
    "identyfikacja dynamiki": "~20 s nagrania; przerywa je STOP albo Dom",
    "identyfikacja": "~20 s nagrania; przerywa je STOP albo Dom",
    "strategia": "Strategie > Zatrzymaj",
}
WAVE_OWNERS = ("kalibracja kamer", "kalibracja")
SYSID_OWNERS = ("identyfikacja dynamiki", "identyfikacja")
#: Najwieksza niepewnosc dopasowanego parametru (wzglednie; 0,3 = +-30%), przy ktorej wynik
#: identyfikacji jest przydatny (kontrakt E2, `FitResult.useful`).
MAX_BAND = 0.3
#: Znacznik w `Dynamics.source` dynamiki zapisanej MIMO werdyktu "nie wyjasnia" (panel dopisuje go
#: przy zapisie) - przewodnik nie zalicza wtedy kroku 5.
USELESS_MARK = "NIEPRZYDATNA"


def dynamics_backend(source: str) -> str:
    """Backend, na ktorym zidentyfikowano dynamike, z `Dynamics.source` ("" = nie wiadomo).

    `sysid.fit` pisze "identyfikacja <czas> (<backend>); niepewnosc: ...".
    """
    m = re.match(r"\s*identyfikacja\b[^(;]*\(([\w-]+)\)", source or "")
    return m.group(1) if m else ""


def dynamics_useful(dyn: dict | None) -> bool:
    """Czy zapisana dynamika (`Workspace.dynamics`) cos wyjasnia - krok 5 liczy sie tylko wtedy.

    Zmierzone (verify2, s6): dopasowanie 0,55 -> 0,51 st. z armatura +-inf dalo sie zapisac
    i przewodnik zaliczal krok 5, a trening centrowal randomizacje na armaturze x0,45 przy
    prawdzie 1,0. Nieprzydatna: znacznik `USELESS_MARK` w zrodle albo niepewnosc (`band`,
    kontrakt E2) dopasowanego parametru nieskonczona lub ponad `MAX_BAND`.
    """
    if not dyn:
        return True
    if USELESS_MARK in str(dyn.get("source") or ""):
        return False
    band = dyn.get("band") or {}
    for name in dyn.get("fitted") or list(band):
        v = band.get(name)
        try:
            if v is not None and not float(v) <= MAX_BAND:
                return False
        except (TypeError, ValueError):
            return False
    return True


@dataclass(frozen=True)
class CameraSnap:
    name: str
    simulated: bool
    enabled: bool = True
    #: Kamera daje kadr (panel liczy ja za martwa dopiero po kilku sekundach bez kadru).
    has_frame: bool = True
    #: K wystarcza do zaufanej pozy (`CameraRecord.intrinsics_problem() == ""`).
    intrinsics_ok: bool = False
    intrinsics_problem: str = ""
    calibrated: bool = False
    trusted: bool = False
    #: Powod niezaufanej pozy (z kalibracji).
    reason: str = ""
    #: Straznik kamer: kadr przesunal sie od kalibracji.
    moved: bool = False
    #: Najlepsza mediana rozjazdu krawedzi w Sim-Real [px] od ostatniej zmiany pozy/stolu/K;
    #: None = nie mierzono, NaN = render bez krawedzi (nic do porownania).
    simreal_px: float | None = None


@dataclass(frozen=True)
class GuideSnapshot:
    connected: bool = False
    #: "sim", "feetech", "lerobot", "a1x" (albo "brak" przed pierwszym polaczeniem).
    backend: str = "brak"
    simulated: bool = True
    arm_error: str = ""
    estop: bool = False
    #: Kto INNY niz panel ma ramie ("" = nikt albo panel).
    owner: str = ""
    warnings: tuple[str, ...] = ()
    cameras: tuple[CameraSnap, ...] = ()
    #: Zrodlo zidentyfikowanej dynamiki; "" = model Menagerie (brak identyfikacji).
    dynamics: str = ""
    #: Backend, na ktorym ja zmierzono (`dynamics_backend`); "" = nie wiadomo.
    dynamics_backend: str = ""
    #: Zapisana dynamika cos wyjasnia (`dynamics_useful`); False - krok 5 nie jest zrobiony.
    dynamics_useful: bool = True
    #: Zadania w tle: "kalibracja kamer", "pomiar obiektywu", "identyfikacja".
    jobs: tuple[str, ...] = ()
    calib_result_pending: bool = False
    dyn_result_pending: bool = False
    #: Czekajacy wynik identyfikacji cos wyjasnia (`jobs.fit_verdict`) - inaczej "powtorz", nie "Zapisz".
    dyn_result_useful: bool = True
    #: Postep fali kalibracyjnej 0..1 (None = nie jedzie).
    calib_progress: float | None = None
    #: Kamera, z ktorej zbierane sa kadry intrynsyk ("" = nie trwa).
    intr_camera: str = ""


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    status: str
    detail: str = ""
    #: Co zrobic w tym kroku (zakladka > przycisk).
    action: str = ""


@dataclass(frozen=True)
class Guide:
    steps: tuple[Step, ...]
    now: str
    blockers: tuple[str, ...] = ()
    background: tuple[str, ...] = ()
    mode: str = ""

    @property
    def current(self) -> Step | None:
        return next((s for s in self.steps if s.status == CURRENT), None)


# ------------------------------------------------------------------ kroki
def _relevant(s: GuideSnapshot) -> list[CameraSnap]:
    """Kamery tego trybu: w symulacji symulowane, na prawdziwym ramieniu prawdziwe (wlaczone)."""
    return [c for c in s.cameras if c.enabled and c.simulated == s.simulated]


def _pose_ok(c: CameraSnap) -> bool:
    """Zaufana poza liczona z zaufanym K (stara kalibracja bez zapisanego K przy nominalnym K - nie)."""
    return c.trusted and (c.simulated or c.intrinsics_ok)


def _px(v: float) -> str:
    """Piksele zaokraglone do 0,5 - lista nie drga od dziesiatych czesci."""
    return f"{round(v * 2) / 2:.1f} px"


def _step_arm(s: GuideSnapshot) -> Step:
    title = "Ramie do maszyny i Polacz"
    if s.connected:
        det = f"polaczone: {s.backend}" + (" (blizniak jest ramieniem)" if s.simulated else "")
        return Step(0, title, DONE, det)
    if s.simulated:
        act = "Ramie: zestaw symulacja, **Polacz**"
    elif s.backend == "a1x":
        # A1X po CAN: interfejs, nie port szeregowy.
        act = "Ramie: zestaw Galaxea A1X dla tego systemu (port wpisuje sie sam), **Polacz** (bez ruchu)"
    else:
        act = ("Ramie: zestaw SO-101 przez port (feetech), Port = COMx / /dev/ttyACM0 (`lerobot-twin check` "
               "wypisuje porty; albo socket://adres:5555 z mostu), **Polacz** (bez ruchu)")
    det = "rozlaczone" + (f" ({s.arm_error})" if s.arm_error else "")
    return Step(0, title, TODO, det, act)


def _step_cameras(s: GuideSnapshot) -> Step:
    title = "Kamery"
    cams = _relevant(s)
    live = [c for c in cams if c.has_frame]
    if live:
        det = f"{len(live)} z kadrem" + (f", bez kadru: {', '.join(c.name for c in cams if not c.has_frame)}"
                                          if len(live) < len(cams) else "")
        return Step(1, title, DONE, det)
    if s.simulated:
        act = "Kamery: **Dodaj symulowana przed ramieniem** (dwa razy - dwie kamery to lepsza mapa)"
        if not s.cameras:
            # Gotowe stanowisko z pakietu (examples/twin.sim.json): dwie kamery z zaufana kalibracja,
            # "kostka z kamer" od razu - bez przechodzenia fali w symulacji.
            act += ("; albo gotowy przyklad: zamknij panel, w konsoli `lerobot-twin demo` "
                    "(examples/twin.sim.json), uruchom panel ponownie")
    else:
        act = "Kamery: **Szukaj kamer USB**, potem **Dodaj kamere USB**"
    det = ("kamery sa, ale zadna nie daje kadru" if cams else
           "brak kamer " + ("symulowanych" if s.simulated else "USB"))
    return Step(1, title, TODO, det, act)


def _step_intrinsics(s: GuideSnapshot) -> Step:
    title = "Obiektywy kamer (szachownica)"
    cams = _relevant(s)
    if s.simulated:
        return Step(2, title, SKIP, "kamery symulowane nie potrzebuja")
    if not cams:
        return Step(2, title, TODO, "najpierw kamery")
    bad = [c for c in cams if not c.intrinsics_ok]
    if not bad:
        return Step(2, title, DONE, f"zmierzone: {', '.join(c.name for c in cams)}")
    first = bad[0]
    if s.intr_camera:
        act = (f"pokazuj szachownice kamerze {s.intr_camera} w rogach obrazu, blisko, daleko i pochylona (12 zdjec), "
               f"potem Kalibracja: **Gotowe - policz obiektyw**")
    else:
        act = f"Kalibracja, krok 1: wydrukuj szachownice, zmierz kwadrat, **Zacznij** dla {first.name}"
    det = "; ".join(f"{c.name}: {'do zmierzenia' if 'nominal' in c.intrinsics_problem else 'do powtorzenia'}"
                    for c in bad)
    return Step(2, title, TODO, det, act)


def _step_poses(s: GuideSnapshot) -> Step:
    title = "Polozenie kamer (karta w chwytaku)"
    cams = _relevant(s)
    if not cams:
        return Step(3, title, TODO, "najpierw kamery")
    moved = [c for c in cams if c.calibrated and c.moved]
    bad = [c for c in cams if not _pose_ok(c)]
    if not bad and not moved:
        return Step(3, title, DONE, f"ustalone: {', '.join(c.name for c in cams)}")
    confirm = "" if s.simulated else " (karta w chwytaku; drugie klikniecie potwierdza)"
    if moved and not bad:
        c = moved[0]
        act = f"Kalibracja: kamera {c.name} poruszona - pomaranczowy przycisk{confirm}"
        return Step(3, title, TODO, f"poruszona: {', '.join(c.name for c in moved)}", act)
    if s.simulated:
        act = ("Kalibracja, krok 2: **Start**, potem **Zapisz wynik** (szybciej: **Symulacja: pomin**)")
    else:
        act = f"Kalibracja, krok 2: karta w chwytaku, **Start**{confirm}, potem **Zapisz wynik**"

    def why(c: CameraSnap) -> str:
        if not c.calibrated:
            return "do ustalenia"
        if c.trusted:                                   # zaufana, ale K nie jest zaufane
            return "ustalone przed zmierzeniem obiektywu - zrob krok 2 jeszcze raz"
        return c.reason or "niepewne"
    det = "; ".join(f"{c.name}: {why(c)}" for c in bad)
    return Step(3, title, TODO, det, act)


def _step_simreal(s: GuideSnapshot) -> Step:
    title = "Sprawdzenie Sim-Real"
    if s.simulated:
        return Step(4, title, SKIP, "kamery symulowane: blad wzgledem prawdy w zakladce Kamery")
    cams = [c for c in _relevant(s) if _pose_ok(c)]
    if not cams:
        return Step(4, title, TODO, "najpierw kalibracja kamer")

    def passed(c: CameraSnap) -> bool:
        return c.simreal_px is not None and not math.isnan(c.simreal_px) and c.simreal_px <= SIMREAL_MAX_PX
    todo = [c for c in cams if not passed(c)]
    if not todo:
        return Step(4, title, DONE, f"zgodne: {', '.join(c.name for c in cams)}")
    c = todo[0]
    if c.simreal_px is None:
        det = f"nie sprawdzone: {', '.join(x.name for x in todo)}"
    elif math.isnan(c.simreal_px):
        det = f"{c.name}: nie sprawdzone (render bez krawedzi - czy kamera widzi ramie i stol?)"
    else:
        det = (f"{c.name}: najlepsza mediana {_px(c.simreal_px)} > {SIMREAL_MAX_PX:.0f} px - sprawdz zmierzony "
               f"bok kodu na karcie i stol (Ramie > Stanowisko), potem krok 2 kalibracji jeszcze raz")
    act = (f"Sim-Real: Kamera = {c.name}, zaznacz **Krawedzie symulacji** - zolte krawedzie maja lezec na ramieniu "
           f"i stole (mediana <= {SIMREAL_MAX_PX:.0f} px)")
    return Step(4, title, TODO, det, act)


def _step_dynamics(s: GuideSnapshot) -> Step:
    title = "Dynamika serw (identyfikacja)"
    # W symulacji wystarczy proba na blizniaku; na prawdziwym ramieniu dynamika z sim
    # (zapisana przy probie procedury) nie jest dynamika serw.
    measured_here = s.dynamics and (s.simulated or s.dynamics_backend != "sim")
    if measured_here and s.dynamics_useful:
        return Step(5, title, DONE, f"zmierzona na {s.dynamics_backend}" if s.dynamics_backend else "zmierzona")
    act = ("Dynamika: " + ("" if s.simulated else "zaznacz 'przestrzen wokol wolna', ")
           + "**Identyfikuj** (~20 s ruchu), potem **Zapisz jako dynamike stanowiska**")
    if measured_here:
        # Zapisana mimo werdyktu "nic nie wyjasnia" - trening centrowalby sie na zgadywance.
        det = "zapisana dynamika nic nie wyjasnia - powtorz identyfikacje (albo Wroc do modelu Menagerie)"
    else:
        det = "zmierzona na sim - zmierz na ramieniu" if s.dynamics else "model Menagerie (nie zmierzona)"
    return Step(5, title, TODO, det, act)


STEP_BUILDERS = (_step_arm, _step_cameras, _step_intrinsics, _step_poses, _step_simreal, _step_dynamics)


# ------------------------------------------------------------------ calosc
def blockers(s: GuideSnapshot) -> list[str]:
    """Uwagi, ktorych NIE mowi "Teraz" (STOP i wlasciciel ramienia sa juz tam)."""
    out = []
    if s.estop and s.owner and s.owner != PANEL:
        # Przy STOP-ie "Teraz" mowi o STOP-ie - kto ma ramie, trzeba powiedziec osobno.
        how = HOW_TO_STOP.get(s.owner, "STOP")
        out.append(f"ramie ma: {s.owner} - UZBROJ, kalibracja kamer, identyfikacja i strategie sa odrzucane, "
                   f"dopoki nie skonczy ({how})")
    elif not s.estop and s.connected and s.arm_error:
        out.append(f"ramie: {s.arm_error}")
    out += [str(w) for w in s.warnings]
    if s.connected and s.backend == "lerobot":
        out.append("backend lerobot liczy katy inaczej niz blizniak - do blizniaka uzywaj feetech")
    if s.connected and s.backend == "a1x":
        out.append("backend a1x: brak hamulcow - STOP trzyma zmierzona poze; nie wylaczaj zasilania z ramieniem "
                   "w gorze")
    for c in s.cameras:
        if not c.enabled:
            continue
        if c.calibrated and c.moved:
            out.append(f"kamera {c.name} poruszona - Kalibracja > pomaranczowy przycisk tej kamery")
        if not c.has_frame:
            out.append(f"kamera {c.name} nie daje kadru" + ("" if c.simulated else
                                                           " (na maszynie wirtualnej/zdalnej: "
                                                           "przepusc kamere USB w kliencie)"))
    if s.cameras and not _relevant(s) and s.connected:
        kind = "symulowane" if s.simulated else "prawdziwe (USB)"
        out.append(f"ramie {s.backend}: do procedury potrzebne kamery {kind}")
    return out


def _step_now(st: Step) -> str:
    return f"Krok {st.number}/{LAST_STEP} ({st.title}): {st.action or st.detail}."


def _now(s: GuideSnapshot, steps: list[Step]) -> str:
    if s.estop:
        why = f" ({s.arm_error})" if s.arm_error else ""
        return f"**aktywny STOP**{why} - usun przyczyne, potem **Skasuj STOP** (zolty, pod STOP-em)."
    cur = next((x for x in steps if x.status == CURRENT), None)
    if not s.connected:
        return _step_now(steps[0])
    if s.owner in WAVE_OWNERS:
        prog = f" ({int(10 * s.calib_progress) * 10}%)" if s.calib_progress is not None else ""
        return (f"Ramie pokazuje karte kamerom{prog} - nie wchodz w jego zasieg; potem Kalibracja > **Zapisz "
                f"wynik** (przerwac: **Zatrzymaj**).")
    if s.owner in SYSID_OWNERS:
        return ("Identyfikacja nagrywa ruch (~20 s) - nie wchodz w obszar ramienia (przerywa ja STOP albo "
                "Dom).")
    if s.owner and s.owner != PANEL:
        return f"Ramie ma: {s.owner} - poczekaj na koniec albo STOP."
    if s.calib_result_pending:
        return "Kalibracja: sprawdz wynik i kliknij **Zapisz wynik**."
    if s.dyn_result_pending and not s.dyn_result_useful:
        return ("Dynamika: wynik identyfikacji nic nie wyjasnia (opis pod paskiem) - powtorz "
                "**Identyfikuj** z wolnym ramieniem, bez obciazenia i kontaktu; nie zapisuj go.")
    if s.dyn_result_pending:
        return "Dynamika: sprawdz wynik i kliknij **Zapisz jako dynamike stanowiska**."
    if "identyfikacja" in s.jobs:
        return "Dopasowanie dynamiki do nagrania (7-11 s, ramie juz wolne) - poczekaj na wynik w zakladce Dynamika."
    if cur is None:
        return ("Procedura przejdziona. Po kazdym poruszeniu kamery: Kalibracja (przycisk tej kamery) i Sim-Real; "
                "ramieniem steruja strategie (zakladka Strategie).")
    return _step_now(cur)


def build(s: GuideSnapshot) -> Guide:
    """Kroki 0-5 z docs/TWIN.md: pierwszy niezrobiony (i nie "nie dotyczy") to "teraz"."""
    raw = [b(s) for b in STEP_BUILDERS]
    steps, found = [], False
    for st in raw:
        if st.status in (DONE, SKIP):
            steps.append(st)
        elif not found:
            steps.append(Step(st.number, st.title, CURRENT, st.detail, st.action))
            found = True
        else:
            steps.append(st)
    # W tle tylko to, czego "Teraz" nie opisuje (fala i identyfikacja juz tam sa).
    bg = [j for j in s.jobs if j not in ("kalibracja kamer", "identyfikacja")]
    mode = ("symulacja - te same kroki i przyciski co na stanowisku" if s.simulated
            else f"prawdziwe ramie ({s.backend}) - reka przy STOP")
    return Guide(tuple(steps), _now(s, steps), tuple(blockers(s)), tuple(bg), mode)


def _always_visible(blocker: str) -> bool:
    """Uwagi, ktore musza byc na wierzchu: brak hamulcow (A1X), blad ramienia, inne katy
    backendu lerobot (suwak uzbrojonego panelu moze zadac inna poze niz widac)."""
    return "hamul" in blocker or blocker.startswith("ramie:") or "lerobot" in blocker


def render_now(g: Guide) -> str:
    """Zawsze widoczna linia "Teraz" (ten sam tekst dla tego samego stanu); uwaga o braku hamulcow
    (A1X) tez tu - nie wolno jej schowac."""
    lines = [f"**Teraz:** {g.now}"]
    lines += [f"- **Uwaga:** {b}" for b in g.blockers if _always_visible(b)]
    return "\n".join(lines)


def render_notes(g: Guide) -> str:
    """Zwijana czesc: pozostale uwagi i tryb (tryb tylko poza symulacja)."""
    lines = [f"- **Uwaga:** {b}" for b in g.blockers if not _always_visible(b)]
    if not g.mode.startswith("symulacja") or g.background:
        lines.append(f"Tryb: {g.mode}" + (f" | w tle: {', '.join(g.background)}" if g.background else ""))
    return "\n".join(lines)


def render_head(g: Guide) -> str:
    """"Teraz", uwagi, tryb - obie czesci razem."""
    return "\n".join(x for x in (render_now(g), render_notes(g)) if x)


def render_steps(g: Guide) -> str:
    """Lista krokow 0-5 (w panelu zwinieta): opis przy biezacym i krotki przy zrobionych."""
    lines = []
    for st in g.steps:
        if st.status == CURRENT:
            line = f"**{st.number}. {st.title}**" + (f" - {st.detail}" if st.detail else "")
        elif st.status == DONE and st.detail and len(st.detail) <= DONE_DETAIL_MAX:
            line = f"{st.number}. {st.title} - {st.detail}"
        elif st.status == SKIP:
            line = f"{st.number}. {st.title} (nie dotyczy)"
        else:
            line = f"{st.number}. {st.title}"
        lines.append(f"{MARK[st.status]} {line}  ")
    return "\n".join(lines)


def render(g: Guide) -> str:
    """Calosc jako jeden markdown (panel wysyla tylko zmiany, wiec ten sam stan = ten sam tekst)."""
    return render_head(g) + "\n\n" + render_steps(g)
