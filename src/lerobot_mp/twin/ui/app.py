"""Panel cyfrowego blizniaka w przegladarce (viser): ramie, kamery, kalibracja, mapa, trening, polityki.

    lerobot-twin ui                      # http://localhost:8080 (tez zdalnie: --host 0.0.0.0)

Jedna petla odswieza widok (poza sceny ~30 Hz, kadry ~5 Hz, mapa ~3 Hz),
a wszystko, co trwa dluzej - fala kalibracyjna, intrynsyki, identyfikacja,
trening, ewaluacja - idzie w tle (`jobs`), wiec zaden przycisk nie blokuje
interfejsu ani petli sterowania ramieniem.

Zasada bezpieczenstwa, ta sama co w reszcie aplikacji: kazdy ruch ramienia
- z suwakow, z uchwytu w 3D, z polityki, z fali - idzie przez
`runtime.Twin` i jego `SafetySupervisor`. Przycisk STOP jest nad zakladkami,
zawsze widoczny.
"""

from __future__ import annotations

import html as _html
import logging
import os
import threading
import time
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import viser
from viser import uplot

from .. import scene as sc
from ..calib.card import Card
from ..calib import intrinsics as intr
from ..calib.intrinsics import Board
from ..kinematics import inverse, pose
from ..perception import CubeDetector, CubeTracker, TableMapper
from ..runtime import Twin
from ..calib import tags as aptags
from ..pmd import is_pmd
from ..tagbody import OPTICAL_AXES, TagBody, learn_layout, locate, observe
from ...paths import REPO_ROOT
from ..workspace import CameraRecord, Workspace, nominal_K
from . import guide as gd
from . import jobs
from .bridge import SceneMirror, mat_to_wxyz
from .specs_panel import add_specs_folder
from .stream import FrameServer
from .watch import CameraWatch, arm_mask, distort_mask

#: Gotowe obiektywy z innych kalibracji (pliki tekstowe: K, dystorsja) - do wczytania w kroku 1.
LENS_DIR = REPO_ROOT / "assets" / "calib"
#: Co ile sekund panel probuje na nowo otworzyc kamere, ktorej strumien padl albo sie nie otworzyl.
HEAL_S = 10.0

logger = logging.getLogger(__name__)

#: Wlasciciel ramienia (`Twin.claim`), gdy steruje panel: suwaki i uchwyt TCP ze sprzeglem.
PANEL_OWNER = "panel"
POLICY_OWNER = "polityka"
#: Najwiekszy skok stawu [st.] miedzy kolejnymi rozwiazaniami IK uchwytu TCP. Zmierzone:
#: przeciagniecie uchwytu o 1-2 cm z shoulder_pan ~108 st. dawalo "poprawne" IK na
#: drugiej galezi - 178-180 st. zmiany, cale ramie przez stol w ~1,3 s.
GIZMO_MAX_STEP_DEG = 15.0
#: Po takiej przerwie w przeciaganiu uchwyt TCP wraca na koncowke (Dom, polityka,
#: suwaki, przebudowa - uchwyt nie moze zostac w starym miejscu jako "cel").
GIZMO_IDLE_S = 0.5
#: Kadr starszy niz tyle [s] nie jest kadrem "na zywo" (zamrozony strumien).
FRAME_MAX_AGE = 1.0
#: Przewodnik mowi "kamera nie daje kadru" dopiero po takiej przerwie [s] - przestoje
#: przekazania USB ponad `FRAME_MAX_AGE` przelaczaly uwage i krok 1 co chwile.
GUIDE_FRAME_LOST = 3.0
#: Detekcja kostki starsza niz tyle [s] nie trafia do polityki jako nowa.
CUBE_MAX_AGE = 0.5
#: Niepewnosc chwili kadru [s] wzgledem jego znacznika czasu (opoznienie kamery i USB).
MASK_WINDOW = (-0.15, 0.25)
#: Najdalsze przesuniecie [px] sylwetki ramienia w masce - dalej to raczej zla historia katow.
MASK_MAX_SHIFT_PX = 60.0
#: Co tyle [px] kopia maski na drodze ramienia - ogniwo ma w kadrze kilkadziesiat px, bez przerw.
MASK_STEP_PX = 6.0


def _skip_unchanged_markdown() -> None:
    """Viser wysyla tresc markdownu przy kazdym przypisaniu, takze tej samej.

    Panel odswieza kilkanascie opisow 5 razy na sekunde - bez tego lecialo to
    przez websocket stale, nawet gdy nic sie nie zmienialo.
    """
    from viser._gui_handles import GuiMarkdownHandle

    prop = GuiMarkdownHandle.content
    if getattr(prop.fset, "_dedup", False):
        return

    def fset(self, content: str) -> None:
        # Porownanie z tym, co naprawde WYSLANO: `add_markdown` wpisuje `_content` przed pierwszym
        # wyslaniem, wiec porownanie z `_content` gubilo kazdy staly tekst (pusty w przegladarce).
        if getattr(self, "_tw_sent", None) != content:
            prop.fset(self, content)
            self._tw_sent = content
    fset._dedup = True
    GuiMarkdownHandle.content = property(prop.fget, fset)


_skip_unchanged_markdown()

GREEN, ORANGE, GREY, BLUE, RED = (40, 190, 90), (240, 160, 30), (150, 150, 150), (60, 140, 255), (230, 50, 50)
CUBE_COLORS = {"czerwona": ((0, 120, 70), (12, 255, 255), (170, 120, 70), (180, 255, 255)),
               "zielona": ((40, 80, 60), (85, 255, 255), (40, 80, 60), (85, 255, 255)),
               "niebieska": ((95, 100, 60), (130, 255, 255), (95, 100, 60), (130, 255, 255)),
               "zolta": ((20, 100, 90), (35, 255, 255), (20, 100, 90), (35, 255, 255))}


#: "Jak uzywac" na gorze zakladek Mapa, Trening, Polityki i Sim-Real (Ramie, Kamery i Kalibracja prowadza same).
TAB_HELP = {
    "Mapa": (
        "Mapa dziala sama, gdy jest co najmniej jedna zaufana kamera (Kalibracja).\n"
        "1. Sprawdz pokrycie blatu pod mapa.\n"
        "2. Wybierz **Kolor kostki**, poloz kostke 3 cm przed ramieniem - ma sie pojawic ramka i polozenie.\n"
        "3. **lift** i **pickplace** z kamer (Polityki) potrzebuja wlaczonych **Mapy na zywo** i **Szukaj kostki**."),
    "Trening": (
        "1. **Dynamika serw**: polacz ramie, **Identyfikuj** (~20 s ruchu), **Zapisz jako dynamike stanowiska**.\n"
        "2. Najszybciej douczanie: **Start z polityki** reach-v3 / lift-v3, 100-300 iteracji, **Ucz**.\n"
        "3. Od zera: Zadanie, Iteracje (reach ~160, lift ~600), **Ucz**; postep na wykresie, "
        "**Zatrzymaj** zapisuje polityke.\n"
        "Trening idzie w osobnym procesie na GPU - ramieniem nie rusza."),
    "Polityki": (
        "1. Wybierz polityke - wyniki GPU i CPU pod lista; **Ewaluuj na CPU** nie rusza ramieniem.\n"
        "2. Najpierw na ramieniu sim: **Uruchom**. Prawdziwe ramie: potwierdzenie przed kazdym uruchomieniem.\n"
        "3. reach: przeciagaj zolta kulke celu w 3D; pickplace: kulka to cel odlozenia (rzutowany na blat). "
        "lift i pickplace na prawdziwym ramieniu: kostka z **kamer**.\n"
        "4. **Zatrzymaj** - ramie trzyma ostatni cel; **STOP** - zatrzymanie awaryjne."),
    "Sim-Real": (
        "1. **Kamera**: prawdziwa, z zaufana poza (Kalibracja).\n"
        "2. **Krawedzie symulacji**: zolte krawedzie renderu na kadrze i mediana rozjazdu - do 3 px dobrze.\n"
        "3. Wiecej: sprawdz bok taga i stol (Ramie > Stanowisko), powtorz fale.\n"
        "Kamery symulowane porownuje sie z prawda w zakladce Kamery."),
}


def wxyz_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def fov_of(K: np.ndarray, height: int) -> float:
    return float(2 * np.arctan(height / 2 / K[1, 1]))


def thumb(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


def _px(shift: float) -> str:
    """Przesuniecie kadru z `CameraWatch` do tabeli; nieskonczone = kadr nie pasuje do odniesienia."""
    return f"{shift:.1f} px" if np.isfinite(shift) else "kadr nie pasuje do odniesienia"


#: Parametry dynamiki w panelu: (pole `Dynamics`, etykieta).
DYN_PARAMS = (("kp", "kp"), ("damping", "tlumienie"), ("armature", "armatura"), ("frictionloss", "tarcie"),
              ("delay", "opoznienie"))


def fitted_names(dyn, sysid_default: bool = False) -> tuple[str, ...]:
    """Parametry, ktore identyfikacja naprawde dopasowala (`Dynamics.fitted`).

    Starsza `Dynamics` bez tego pola: dla swiezego wyniku identyfikacji - lista,
    z ktora ja liczono (`sysid.IDENTIFIED`); dla zapisanej dynamiki - nic (nie wiadomo).
    """
    names = getattr(dyn, "fitted", None)
    if names is None and sysid_default:
        from ..rl import sysid
        names = getattr(sysid, "IDENTIFIED", ())
    return tuple(names or ())


def band_text(name: str, width) -> str:
    """Niepewnosc parametru jak w `FitResult.band`: mnozniki wzglednie (0,6 -> "+-60%"), opoznienie w s."""
    try:
        w = float(width)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(w):
        return "+-inf"
    return f"+-{1000 * w:.0f} ms" if name == "delay" else f"+-{100 * w:.0f}%"


def dynamics_parts(dyn, fitted, band: dict | None = None) -> tuple[list[str], list[str]]:
    """(dopasowane, z modelu) jako teksty "tlumienie x0.95 (+-8%)", "opoznienie 25 ms".

    Panel pisal "Dopasowano: kp x1.00 ... tarcie x1.00" takze dla parametrow, ktorych
    identyfikacja NIE ruszala (kp i tarcie zostaja z modelu) - wygladalo to jak
    pomiar "kp idealnie jak w modelu", a to tylko wartosc startowa.

    `band` (niepewnosc z identyfikacji) przy kazdej dopasowanej wartosci: "armatura x0.45"
    bez "+-inf" wygladalo jak pomiar, a nie zgadywanka (verify2, s6).
    """
    fitted = set(fitted)
    band = band or {}
    a, b = [], []
    for name, label in DYN_PARAMS:
        v = float(getattr(dyn, name))
        txt = f"{label} {v / 20 * 1000:.0f} ms" if name == "delay" else f"{label} x{v:.2f}"
        if name in fitted and name in band and band_text(name, band[name]):
            txt += f" ({band_text(name, band[name])})"
        (a if name in fitted else b).append(txt)
    return a, b


def smear_mask(mask: np.ndarray, shifts: list[tuple[float, float]]) -> np.ndarray:
    """Suma maski i jej kopii przesunietych o `shifts` [px] - sylwetka na drodze ramienia."""
    out = mask.copy()
    if not shifts or not mask.any():
        return out
    h, w = mask.shape
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
    crop = mask[y:y + bh, x:x + bw]                         # tylko prostokat ramienia - kilka razy szybciej
    for dx, dy in {(int(round(dx)), int(round(dy))) for dx, dy in shifts}:
        x0, y0 = x + dx, y + dy
        sx, sy = max(0, -x0), max(0, -y0)
        ex, ey = min(bw, w - x0), min(bh, h - y0)
        if (dx or dy) and sx < ex and sy < ey:
            out[y0 + sy:y0 + ey, x0 + sx:x0 + ex] |= crop[sy:ey, sx:ex]
    return out


#: Suwaki stawow: krotka etykieta i pelna nazwa (do podpowiedzi) zamiast nazw z MJCF.
JOINT_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "so101": {"shoulder_pan": ("Podstawa", "Obrot podstawy"), "shoulder_lift": ("Bark", "Bark"),
              "elbow_flex": ("Lokiec", "Lokiec"), "wrist_flex": ("Nadg. zgiecie", "Nadgarstek - zgiecie"),
              "wrist_roll": ("Nadg. obrot", "Nadgarstek - obrot"), "gripper": ("Chwytak", "Chwytak")},
    "a1x": {"arm_joint1": ("J1 Podstawa", "J1 - obrot podstawy"), "arm_joint2": ("J2 Bark", "J2 - bark"),
            "arm_joint3": ("J3 Lokiec", "J3 - lokiec"), "arm_joint4": ("J4 Nadg. gora", "J4 - nadgarstek gora/dol"),
            "arm_joint5": ("J5 Nadg. bok", "J5 - nadgarstek na bok"), "arm_joint6": ("J6 Obrot", "J6 - obrot chwytaka"),
            "gripper": ("Chwytak", "Chwytak")},
}

#: Wyglad przycisku UZBROJ, gdy panel nie moze uzbroic: (etykieta, kolor, ikona) - przycisk wylaczony.
ARM_OVERRIDES = {
    "off": lambda who: ("UZBROJ - najpierw Polacz", "gray", viser.Icon.PLUG_CONNECTED_X),
    "estop": lambda who: ("UZBROJ - najpierw Skasuj STOP", "gray", viser.Icon.HAND_STOP),
    "busy": lambda who: (f"UZBROJ - ramie ma: {who}", "gray", viser.Icon.LOCK),
    "homing": lambda who: ("UZBROJ - czekaj, jedzie do domu", "gray", viser.Icon.HOME),
    "lost": lambda who: ("UZBROJ - petla padla: Rozlacz, Polacz", "gray", viser.Icon.PLUG_CONNECTED_X),
}
GIZMO_ON_ARMED = ("Uchwyt 3D: wl. - ciagnij strzalki", "blue", viser.Icon.ARROWS_MOVE)
GIZMO_ON_IDLE = ("Uchwyt 3D: wl. (bez UZBROJ nie rusza)", "blue", viser.Icon.ARROWS_MOVE)
GIZMO_OFF = ("Uchwyt 3D: wyl.", "gray", viser.Icon.ARROWS_MOVE)
GIZMO_ESTOP = ("Uchwyt 3D: STOP aktywny", "gray", viser.Icon.ARROWS_MOVE)
CAM_ON = ("Wlaczona - kliknij, by wylaczyc", "green", viser.Icon.EYE)
CAM_OFF = ("Wylaczona - kliknij, by wlaczyc", "gray", viser.Icon.EYE_OFF)
MOVE_ON = ("Przesuwanie w 3D: wl. - ciagnij uchwyt", "blue", viser.Icon.ARROWS_MOVE)
MOVE_OFF = ("Przesuwaj w 3D", "gray", viser.Icon.ARROWS_MOVE)
#: Tyle [s] czeka pytanie na przycisku z potwierdzeniem drugim kliknieciem (`_TwoClick`).
CONFIRM_S = 8.0

#: Styl panelu: STOP i plakietka stanu przyklejone u gory (widoczne przy kazdym przewinieciu), Skasuj STOP pod nimi,
#: duze przyciski UZBROJ / Uchwyt 3D, plakietka stanu. Kotwice `<span id="tw-...">` stoja tuz przed
#: przyciskiem, ktory stylujemy (`div:has(> #tw-x) + div` = nastepny element visera). Selektory
#: opieraja sie na stalych klasach Mantine i atrybutach data-dock visera 1.1.x; gdy sie nie
#: dopasuja, STOP zostaje zwyklym czerwonym przyciskiem, ktory dalej dziala.
PANEL_CSS = """<style>
div[style*="grid-template-rows"] > div:has(#tw-stop){overflow:clip !important}
/* siatka panelu rosla do najszerszego elementu (o szerokosc paska przewijania) - obciete prawe krawedzie */
div[style*="grid-template-rows"]{grid-template-columns:minmax(0,1fr)}
div:has(> #tw-stop) + div{position:sticky;top:0;z-index:60;background:var(--mantine-color-body);
  margin-inline:0 !important;padding:6px var(--mantine-spacing-xs) 6px !important;
  box-shadow:0 8px 10px -8px rgba(0,0,0,.7)}
div:has(> #tw-stop) + div > button{height:56px !important;font-size:1.35rem;font-weight:800;letter-spacing:.14em}
div:has(> #tw-stop) + div > button .mantine-Button-section > div{width:26px;height:26px}
div:has(> #tw-stop) + div > button .mantine-Button-section svg{width:26px;height:26px;position:static;transform:none}
div:has(> #tw-stop) + div > button .mantine-Button-section[data-position="left"]{margin-right:.6em !important}
div:has(> .tw-chip){position:sticky;top:68px;z-index:59;background:var(--mantine-color-body);padding-bottom:6px}
div:has(> #tw-clear) + div:has(> button){position:sticky;top:103px;z-index:58;background:var(--mantine-color-body);
  margin-inline:0 !important;padding:0 var(--mantine-spacing-xs) 6px !important}
div:has(> #tw-clear) + div:has(> button) > button{height:40px !important;font-weight:700}
.tw-chip{display:flex;align-items:center;gap:.6em;margin:0 var(--mantine-spacing-xs);border-radius:4px;
  padding:.3em .7em;font-size:.8rem}
.tw-word{font-weight:800;letter-spacing:.06em;font-size:.95rem}
.tw-meta{opacity:.8;margin-left:auto;text-align:right}
.tw-off{background:var(--mantine-color-dark-6);color:var(--mantine-color-dark-1)}
.tw-idle{background:var(--mantine-color-dark-5);color:var(--mantine-color-gray-3)}
.tw-armed{background:var(--mantine-color-orange-8);color:#fff}
.tw-busy{background:var(--mantine-color-blue-8);color:#fff}
.tw-estop{background:var(--mantine-color-red-9);color:#fff}
.tw-lost{background:var(--mantine-color-red-9);color:#fff}
.tw-homing{background:var(--mantine-color-orange-8);color:#fff}
div:has(> #tw-arm) + div > button{height:auto !important;min-height:3.2em;font-size:1.1rem;font-weight:800;
  letter-spacing:.04em}
div:has(> #tw-arm) + div > button .mantine-Button-label{white-space:normal;line-height:1.15}
div:has(> #tw-arm) + div > button:disabled{opacity:.9;font-weight:600}
div:has(> #tw-3d) + div > button{height:2.6em !important;font-weight:700}
/* zakladki w jednym rzedzie: nieaktywne tylko ikona (nazwa w dymku), aktywna z nazwa */
div:has(> .tw-big) + div > button{height:auto !important;min-height:3em;font-size:1.05rem;font-weight:800;
  letter-spacing:.03em}
div:has(> .tw-big) + div > button .mantine-Button-label{white-space:normal;line-height:1.15}
.tw-cams{display:flex;flex-direction:column;gap:4px;margin:0 var(--mantine-spacing-xs);min-width:0}
.tw-cam{display:flex;flex-wrap:wrap;align-items:center;gap:.3em .5em;padding:.3em .5em;border-radius:4px;min-width:0;
  background:var(--mantine-color-dark-6);font-size:.8rem}
.tw-cam b{font-size:.9rem}
.tw-src{opacity:.6}
.tw-badges{margin-left:auto;display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end}
.tw-badge{font-weight:800;letter-spacing:.04em;padding:.1em .45em;border-radius:3px;font-size:.72rem;
  white-space:nowrap}
.tw-ok{background:var(--mantine-color-green-8);color:#fff}
.tw-warn{background:var(--mantine-color-orange-8);color:#fff}
.tw-bad{background:var(--mantine-color-red-9);color:#fff}
.tw-none{background:var(--mantine-color-dark-4);color:var(--mantine-color-gray-3)}
.tw-sim{background:var(--mantine-color-blue-8);color:#fff}
.tw-empty{margin:0 var(--mantine-spacing-xs);padding:.6em .7em;border:1px dashed var(--mantine-color-dark-3);
  border-radius:4px;font-size:.85rem;opacity:.85}
[data-dock-strip] [data-dock-tab]{padding:0 .55em !important}
[data-dock-strip] [data-dock-tab] svg{width:1.35em;height:1.35em}
[data-dock-strip] [data-dock-tab][aria-selected="false"] > span{display:none !important}
[data-dock-strip] [data-dock-tab][aria-selected="false"] > div:first-child{margin-right:0 !important;opacity:.9}
</style>"""


def set_if(handle, attr: str, value) -> None:
    """Zapis wlasciwosci visera tylko przy zmianie (kazdy zapis to komunikat do przegladarki)."""
    if getattr(handle, attr) != value:
        setattr(handle, attr, value)


class _Toggle:
    """Wlacznik z przyciskow o semantyce checkboxa visera: `.value` (bool), zapis z kodu wola
    `on_update` ze zdarzeniem bez `client` (jak checkbox), klik - z prawdziwym zdarzeniem.

    Z przyciskiem `release` zamiar kliku jest BEZWZGLEDNY: `primary` = wlacz, `release` = wylacz,
    wiec klik na nieodswiezonej stronie nie odwroci zamiaru operatora (UZBROJ). Bez `release`
    jeden przycisk przelacza (Uchwyt 3D - tylko widok, nic nie rusza).
    """

    def __init__(self, primary, release=None, on_look=None, off_look=None):
        self.primary, self.release = primary, release
        self.on_look, self.off_look = on_look, off_look
        self._value, self._override, self._cbs = False, None, []
        self._t_off = 0.0                               # chwila ostatniego wylaczenia (blokada dwukliku)
        self._lock = threading.RLock()
        if release is None:
            primary.on_click(lambda e: self._clicked(e, None))
        else:
            primary.on_click(lambda e: self._clicked(e, True))
            release.on_click(lambda e: self._clicked(e, False))
        self._apply()

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, v) -> None:
        v = bool(v)
        with self._lock:
            if v == self._value:
                return                                  # jak viser: bez zmiany nie ma zdarzenia
            self._value = v
            if not v:
                self._t_off = time.monotonic()
            self._apply()
        ev = viser.GuiEvent(client=None, client_id=None, target=self.primary)
        for cb in list(self._cbs):
            cb(ev)

    def _clicked(self, event, want) -> None:
        if event.client is None:
            return
        with self._lock:
            want = (not self._value) if want is None else want
            if want == self._value:                     # nieodswiezona strona: tylko przemaluj
                self._apply()
                return
            if want and self.release is not None and time.monotonic() - self._t_off < 0.5:
                self._apply()                           # drugi klik dwukliku "rozbroj" trafil w UZBROJ
                return
            self._value = want
            if not want:
                self._t_off = time.monotonic()
            self._apply()
        for cb in list(self._cbs):
            cb(event)

    def on_update(self, fn):
        self._cbs.append(fn)
        return fn

    def set_override(self, look) -> None:
        """None albo (etykieta, kolor, ikona): przycisk zablokowany z powodem na etykiecie."""
        with self._lock:
            if look != self._override:
                self._override = look
                self._apply()

    @property
    def disabled(self) -> bool:
        return self._override is not None

    @property
    def hint(self):
        return self.primary.hint

    def _apply(self) -> None:
        """Same zapisy wlasciwosci - nigdy nie czyta stanu blizniaka i nigdy nie rzuca (po nim idzie
        zwolnienie sprzegla albo powiadomienie o STOP-ie)."""
        try:
            if self._override is not None:
                label, color, icon = self._override
                self._show(self.primary, label, color, icon, disabled=True)
                if self.release is not None:
                    set_if(self.release, "visible", False)
            elif self.release is not None:
                set_if(self.primary, "visible", not self._value)
                set_if(self.primary, "disabled", False)
                set_if(self.release, "visible", self._value)
                if not self._value:
                    self._show(self.primary, "UZBROJ", "gray", viser.Icon.LOCK, disabled=False)
            else:
                label, color, icon = self.on_look if self._value else self.off_look
                self._show(self.primary, label, color, icon, disabled=False)
        except Exception:
            logger.exception("Przemalowanie wlacznika")

    @staticmethod
    def _show(b, label, color, icon, disabled) -> None:
        set_if(b, "visible", True)
        set_if(b, "label", label)
        set_if(b, "color", color)
        set_if(b, "icon", icon)
        set_if(b, "disabled", disabled)


class _TwoClick:
    """Potwierdzenie drugim kliknieciem: pierwszy klik (`ask` -> False) zapamietuje zamiar `key`,
    przycisk pokazuje pytanie (`pending`); drugi klik tego samego zamiaru w ciagu CONFIRM_S s -> True.
    Uplyw czasu, inny zamiar albo `reset` (Polacz, start fali) - od nowa."""

    def __init__(self, window: float = CONFIRM_S):
        self.window = window
        self._key, self._t = None, 0.0
        self._lock = threading.Lock()

    def ask(self, key) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._key == key and now - self._t < self.window:
                self._key = None
                return True
            self._key, self._t = key, now
            return False

    def pending(self, key=None) -> bool:
        k, t = self._key, self._t
        return k is not None and time.monotonic() - t < self.window and (key is None or k == key)

    def reset(self) -> None:
        with self._lock:
            self._key = None


def no_frame_image(text: str = "brak kadru", size: tuple[int, int] = (320, 240)) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.putText(img, text, (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 60, 60), 2, cv2.LINE_AA)
    return img


class TwinApp:
    def __init__(self, workspace: str | Path | None = None, host: str = "127.0.0.1", port: int = 8080,
                 stream_port: int | None = None, stream_host: str = "0.0.0.0"):
        self.host = host
        #: Podglad kamer po HTTP (MJPEG) na osobnym porcie - tylko do odczytu, wiec domyslnie cala siec.
        self.stream: FrameServer | None = None
        if stream_port is not None:
            self.stream = FrameServer(self.stream_frames, host=stream_host, port=stream_port)
        self.ws = Workspace.load(workspace)
        self.ws_path = Path(self.ws.path).resolve() if self.ws.path else None
        self.twin = Twin(self.ws)
        self.watch = CameraWatch()
        self.frames: dict[str, np.ndarray] = {}
        #: Chwila wykonania kazdego kadru z `frames` (time.monotonic) - wiek detekcji kostki.
        self.frame_times: dict[str, float] = {}
        self._t_grab = 0.0
        self.frame_lock = threading.Lock()
        self.calib_job, self.intr_job = jobs.Job("kalibracja"), jobs.Job("intrynsyki")
        self.sysid_job, self.eval_job = jobs.Job("identyfikacja"), jobs.Job("ewaluacja")
        from ..rl.policy import DEFAULT_DIR
        self.policies_dir = DEFAULT_DIR.resolve()
        self.train = jobs.TrainingJob(self.policies_dir, self.ws_path)
        self.runner = None
        self.policy = None
        #: Licznik "operator zatrzymal ruch" (STOP, Dom, Polacz/Rozlacz, Zatrzymaj). Start polityki,
        #: fali i identyfikacji zapamietuje go na poczatku i bierze ramie tylko, gdy sie nie zmienil
        #: (`_motion_mark`, `_take_arm`) - STOP wcisniety w trakcie ladowania polityki ginal.
        self._motion_gen = 0
        self.cube_det = CubeDetector()
        self.cube_tracker = CubeTracker.for_spec(self.ws.spec())
        self.last_cube = None
        #: Chwila kadrow ostatniej detekcji podanej polityce - ta sama detekcja nie idzie drugi raz.
        self._cube_used_t = 0.0
        from ..kinematics import RobotKinematics
        self._kin_vision = RobotKinematics(self.ws.spec())
        # Punkty ramienia do przesuniecia maski w kadrze - wlasna kinematyka (maski licza
        # petla panelu i przyciski visera, a `_kin_vision` jest w watku polityki).
        self._kin_mask = RobotKinematics(self.ws.spec())
        self._mask_lock = threading.Lock()
        # Uchwyt TCP: wlasna kinematyka (callbacki visera ida z wielu watkow naraz,
        # a `ik` pisze do swojego MjData) i ostatnie przyjete rozwiazanie jako start IK.
        self._kin_gizmo = RobotKinematics(self.ws.spec())
        self._gizmo_lock = threading.Lock()
        self._gizmo_seed: dict[str, float] | None = None
        self._gizmo_t = 0.0
        self._engage_t = 0.0
        #: (t, katy) z ostatnich ~3 s - maska ramienia w pozie z chwili kadru, nie renderu.
        self._joint_hist: deque[tuple[float, dict[str, float]]] = deque(maxlen=120)
        self._goal_base: np.ndarray | None = None
        self._reach_task = None                         # obszar celow reach bez uruchomionej polityki
        self._map_pose_key = None
        self._intr_solve = threading.Event()
        self._dirty_rebuild = 0.0                       # kamera spoza modelu przeciagana - przebudowa po chwili
        self.mapper: TableMapper | None = None
        self._mapper_key = None
        self._stop = threading.Event()
        self.frustums: dict[str, Any] = {}
        self._frustum_key: dict[str, Any] = {}
        self._img_sent: dict[str, int] = {}
        #: Piramidy kamer tworza i usuwaja callbacki (inne watki), a obrazy wstawia petla -
        #: bez blokady petla trafiala w usuniety uchwyt i caly panel padal.
        self.scene_lock = threading.RLock()
        self._tick_errors: set[str] = set()
        self.estimates: dict[str, Any] = {}
        self._dirty_save = 0.0                          # przeciaganie kamery: zapis po chwili spokoju
        # Przewodnik: najlepsza mediana Sim-Real kazdej kamery od ostatniej zmiany pozy/stolu/K
        # (zatrzask - krok 4 nie migocze z szumem), co juz jechalo na danym backendzie
        # ("reach", "lift-kamery"), backend, na ktorym ruszyl biezacy `runner`, ostatni kadr
        # kazdej kamery (kamera "bez kadru" dopiero po GUIDE_FRAME_LOST s) i lista polityk
        # z ostatniego odswiezenia (bez czytania dysku w kazdym takcie).
        self._simreal_px: dict[str, float] = {}
        self._ran: dict[str, set[str]] = {}
        self._runner_backend = ""
        self._frame_seen: dict[str, float] = {}
        self._policy_list: list[dict] = []
        # Widok zakladki Ramie: klucz ostatniego przemalowania (`_render_arm`), stan suwakow,
        # ostatnia plakietka stanu - przemalowanie tylko przy zmianie.
        self._arm_key = None
        self._sliders_armed = None
        self._render_lock = threading.RLock()
        self._heal_t = 0.0                          # ostatnia proba `_heal_cameras`
        self._chip_last = ""
        #: Ostatni przewodnik (`_tick_guide`) - linia "Teraz" zakladki Kalibracja bierze z niego kroki 1-3.
        self._guide_last: gd.Guide | None = None
        t_start = time.monotonic()

        # Serwer powstaje DOPIERO teraz, gdy cale ciezkie przygotowanie (scena, torch,
        # stanowisko) jest za nami: viser slucha od chwili utworzenia, a przegladarka
        # polaczona w polowie budowy panelu potrafila zostac z pustym widokiem.
        # Potem od razu panel (lekki), a na koncu siatki ramienia (~350 tys. trojkatow).
        self.server = viser.ViserServer(host=host, port=port, label=f"Blizniak {self.ws.robot.upper()}",
                                        verbose=False)
        s = self.server
        s.gui.configure_theme(control_width="large", dark_mode=True, show_logo=False,
                              show_share_button=False, brand_color=(255, 196, 0))
        if hasattr(s.gui, "main_panel"):
            s.gui.main_panel.dock_right()
        s.scene.set_up_direction("+z")
        T = self.twin.scene.T_base2world
        s.initial_camera.position = tuple(T[:3, 3] + np.array([0.75, -0.65, 0.55]))
        s.initial_camera.look_at = tuple(T[:3, 3] + np.array([0.18, 0.0, 0.05]))

        self._build_header()
        tabs = s.gui.add_tab_group()
        with tabs.add_tab("Ramie", icon=viser.Icon.ROBOT):
            self._build_arm()
        with tabs.add_tab("Kamery", icon=viser.Icon.CAMERA):
            self._build_cameras()
        with tabs.add_tab("Kalibracja", icon=viser.Icon.TARGET):
            self._build_calibration()
        with tabs.add_tab("Mapa", icon=viser.Icon.MAP):
            self._build_map()
        with tabs.add_tab("Trening", icon=viser.Icon.BRAIN):
            self._build_training()
        with tabs.add_tab("Polityki", icon=viser.Icon.PLAYER_PLAY):
            self._build_policies()
        with tabs.add_tab("Sim-Real", icon=viser.Icon.ARROWS_DIFF):
            self._build_simreal()

        # Oswietlenie widoku (nie symulacji): rozproszone z nieba (+z) i jedno kierunkowe z gory.
        # three.js bierze kierunek swiatla polkuli z POLOZENIA swiatla (obrot nic nie zmienia):
        # polozenie (0, 0, 1) = niebo w +z. Obrot przy polozeniu (0, 0, 0) dawal zerowy
        # kierunek i plaskie pol na pol nieba i ziemi na kazdej powierzchni.
        s.scene.add_light_ambient("/swiatlo/otoczenie", intensity=0.5)
        s.scene.add_light_hemisphere("/swiatlo/niebo", sky_color=(235, 240, 255), ground_color=(70, 65, 60),
                                     intensity=1.6, position=(0.0, 0.0, 1.0))
        s.scene.add_light_directional("/swiatlo/gora", intensity=1.4, position=(0.6, -0.8, 2.5), cast_shadow=True)
        self.base_frame = s.scene.add_frame("/podstawa", axes_length=0.08, axes_radius=0.003,
                                            position=T[:3, 3], wxyz=mat_to_wxyz(T[:3, :3]))
        self.mirror = None
        self._mirror_version = -1
        self._sync_mirror()
        self.twin.cameras.sync()
        self._refresh_cameras()
        self._guard("przewodnik", self._tick_guide)
        logger.info("Panel gotowy w %.1f s", time.monotonic() - t_start)
        if self.stream is not None:
            self.stream.start()
        # SPECS_CONNECT=1: polacz z przekaznikiem od razu (tryb dry - ramie stoi), bez klikania w panelu.
        if os.environ.get("SPECS_CONNECT", "").strip().lower() in ("1", "true", "yes", "tak"):
            self._guard("spectacles", self.specs.start)

    # ================================================================ narzedzia
    def _safe(self, fn):
        """Wyjatek w przycisku ma trafic do uzytkownika, a nie zginac w watku visera."""
        @wraps(fn)
        def wrapper(event, *a, **kw):
            try:
                return fn(event, *a, **kw)
            except Exception as exc:
                logger.exception("Blad w panelu")
                self._notify(event, "Blad", f"{type(exc).__name__}: {exc}", error=True)
        return wrapper

    def _notify(self, event, title: str, body: str, error: bool = False) -> None:
        client = getattr(event, "client", None)
        targets = [client] if client is not None else list(self.server.get_clients().values())
        for c in targets:
            c.add_notification(title, body, auto_close_seconds=6.0 if not error else 12.0,
                               color="red" if error else "yellow")

    def _img(self, handle, key: str, img: np.ndarray) -> None:
        """Obraz do panelu/sceny tylko, gdy sie zmienil (stojaca scena = ten sam kadr)."""
        digest = hash(img.tobytes()[::97])
        if self._img_sent.get(key) != digest:
            self._img_sent[key] = digest
            handle.image = img

    def _save(self) -> None:
        self.ws.save(self.ws_path)
        self.ws_path = Path(self.ws.path).resolve()
        self.train.workspace_path = self.ws_path

    @property
    def T_b2w(self) -> np.ndarray:
        return self.twin.scene.T_base2world

    def _tab_help(self, tab: str) -> None:
        """Zwinieta notka "Jak uzywac" na gorze zakladki (`TAB_HELP`)."""
        with self.server.gui.add_folder("Jak uzywac", expand_by_default=False):
            self.server.gui.add_markdown(TAB_HELP[tab])

    # ================================================================ naglowek
    def _build_header(self) -> None:
        """Nad zakladkami: STOP przyklejony u gory, Skasuj STOP (tylko przy STOP-ie), plakietka stanu
        ramienia, wyjatki w statusie, jedna linia przewodnika "Teraz"; uwagi i kroki zwiniete."""
        g = self.server.gui
        # Styl + kotwica STOP-u; naglowek nigdy nie znika, wiec styl jest zawsze na stronie.
        self._css = g.add_html(PANEL_CSS + '<span id="tw-stop"></span>')
        stop = g.add_button("STOP", color="red", icon=viser.Icon.HAND_STOP,
                            hint="Zatrzymanie awaryjne, zawsze dostepne: ramie staje w zmierzonej pozie, polityka, "
                                 "fala i nagranie identyfikacji sie koncza, panel sie rozbraja. Sciskajacy chwytak "
                                 "dalej sciska. Ruszyc dalej: Skasuj STOP (zolty, pod STOP-em).")
        self.state_html = g.add_html(self._state_chip_html("off", ""))
        g.add_html('<span id="tw-clear"></span>')
        self.clear_btn = g.add_button("Skasuj STOP", color="yellow", icon=viser.Icon.RESTORE, visible=False,
                                      hint="Kasuje STOP awaryjny (takze od bledu serwa). Najpierw usun przyczyne. "
                                           "Ramie samo nie rusza - potem UZBROJ, zeby sterowac.")
        self.status_md = g.add_markdown("", visible=False)
        # Przewodnik: gdzie operator jest w procedurze (docs/TWIN.md, kroki 0-8) i jedna nastepna
        # akcja. Logika w `guide.py`, tu tylko wyswietlanie. Widoczna zawsze tylko linia "Teraz";
        # uwagi i lista krokow zwiniete - rozwiniete zabieraly ok. 190-400 px nad kazda zakladka.
        self.guide_md = g.add_markdown("")
        self.guide_folder = g.add_folder("Kroki 0-8", expand_by_default=False)
        with self.guide_folder:
            self.guide_notes_md = g.add_markdown("")
            self.guide_steps_md = g.add_markdown("")

        @self.clear_btn.on_click
        @self._safe
        def _(event):
            self.twin.clear_estop()
            self._notify(event, "STOP skasowany", "Ramie stoi. UZBROJ, zeby sterowac.")

        @stop.on_click
        async def _(event):
            # Asynchronicznie, w petli zdarzen visera, a sama robota od razu w nowym watku:
            # zwykly callback czekal w tej samej puli 32 watkow co reszta panelu - za
            # przebudowami sceny z przeciagania kamery STOP potrafil ruszyc po kilkudziesieciu s.
            threading.Thread(target=self._emergency_stop, args=(event,), name="STOP", daemon=True).start()

    def _emergency_stop(self, event=None) -> None:
        """STOP: najpierw nadzor (ramie staje w zmierzonej pozie), dopiero potem sprzatanie watkow."""
        self._motion_gen += 1                           # start w toku (ladowanie polityki) nie wezmie ramienia
        try:
            self.twin.estop("STOP z panelu")
        finally:
            if self.runner is not None:
                self.runner.stop("STOP z panelu")
            self.calib_job.stop()
            self.sysid_job.stop()                       # tylko nagranie - dopasowanie jest chronione
            self.arm_engage.value = False
            self._gizmo_seed = None
        if self._estop_active():
            self._notify(event, "STOP", "Ramie zatrzymane. Skasuj STOP (zolty, pod STOP-em), zeby ruszyc dalej.",
                         error=True)
        else:
            self._notify(event, "STOP", "Ramie niepolaczone - nic nie jedzie; zadania w tle przerwane.", error=True)

    # ------------------------------------------------ kto steruje ramieniem
    def _busy(self, me: str) -> str:
        """Kto INNY teraz rusza ramieniem ("" = nikt). Panel (sprzeglo) oddaje ramie sam - nie blokuje.

        Zadania patrzymy obok `Twin.owner`: fala bierze ramie dopiero przy pierwszym
        przejezdzie, a do tego czasu (przebudowa sceny, sprawdzacz kolizji) jest juz w toku.
        """
        if me != jobs.CALIB_OWNER and self.calib_job.running:
            return "kalibracja kamer"
        if me != jobs.SYSID_OWNER and self.sysid_job.running and "recording" not in self.sysid_job.data:
            return "identyfikacja dynamiki"
        if me != POLICY_OWNER and self.runner is not None and self.runner.status.running:
            return "polityka"
        owner = self.twin.owner
        if owner is not None and owner not in (PANEL_OWNER, me):
            return owner
        return ""

    def _release_panel(self) -> None:
        """Panel puszcza ramie (sprzeglo off) - tylko jesli wciaz je ma."""
        self.arm_engage.value = False
        self._gizmo_seed = None
        if self.twin.owner == PANEL_OWNER:
            self.twin.set_engaged(False)
            self.twin.release(PANEL_OWNER)

    def _motion_mark(self) -> tuple[int, int]:
        """(licznik panelu, `Twin.preempt_gen`) na poczatku startu zadania - patrz `_claim_kw`."""
        return self._motion_gen, self.twin.preempt_gen

    def _claim_kw(self, mark: tuple[int, int] | None) -> dict:
        """Warunki dla `Twin.claim`: od `mark` nikt nie wcisnal STOP/Dom/Polacz/Zatrzymaj.

        Zmierzone (verify2, s15/s15c/s15d): od klikniecia Uruchom do `claim` polityki mija
        108-142 ms (ladowanie polityki); STOP albo Dom w tym oknie trafialy w stary runner,
        a nowa polityka brala ramie pod STOP-em (po Skasuj STOP - 83,7 st. ruchu) albo
        przerywala rampe do domu.
        """
        if mark is None:
            return {}
        panel_gen, twin_gen = mark
        return {"gen": twin_gen, "guard": lambda: self._motion_gen == panel_gen}

    def _take_arm(self, owner: str, preempt=None, mark: tuple[int, int] | None = None) -> None:
        """Ramie dla zadania `owner` - albo RuntimeError z tym, kto je ma. Panel oddaje je bez pytania.

        `mark` (`_motion_mark` z poczatku startu zadania): STOP, Dom albo Polacz od tamtej chwili
        -> odmowa; aktywny STOP -> odmowa ("aktywny STOP", z `Twin.claim`).
        """
        busy = self._busy(owner)
        if busy:
            raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj (albo STOP)")
        self._release_panel()
        self.twin.claim(owner, preempt=preempt, **self._claim_kw(mark))

    def _panel_preempted(self) -> None:
        """Twin odebral ramie panelowi (Dom, STOP, polaczenie, petla padla) - sprzeglo w panelu off."""
        self.arm_engage.value = False
        self._gizmo_seed = None

    def _stop_motion(self, reason: str, wait: float = 3.0) -> None:
        """Konczy wszystko, co rusza ramieniem, i czeka na zadania - przed (roz)laczeniem.

        Bez tego fala albo identyfikacja z sim jechala dalej na swiezo polaczonym
        prawdziwym ramieniu (odtworzone: shoulder_pan 14,8 -> 40 -> -39 -> 33 st.
        bez zadnej akcji operatora), z pominieciem potwierdzen z ich startu.
        """
        self._motion_gen += 1                           # start w toku nie wezmie ramienia po (roz)laczeniu
        if self.runner is not None:
            self.runner.stop(reason)
        self.calib_job.stop()
        # Identyfikacja: przerywa tylko nagranie. Po nim zadanie jest chronione
        # (`Job.protect`) - dopasowanie ramienia nie rusza i konczy sie z wynikiem.
        self.sysid_job.stop()
        self.twin.preempt(reason)
        for job in (self.calib_job, self.sysid_job):
            if job.running and "recording" not in job.data and not job.wait(wait):
                logger.warning("Zadanie %s nie skonczylo sie w %.0f s (%s)", job.name, wait, reason)
        self._release_panel()

    def _reset_confirmations(self) -> None:
        """Kazdy przejazd prawdziwego ramienia wymaga swiezego potwierdzenia (karta, wolne miejsce)."""
        self.calib_ask.reset()
        for box in (self.dyn_confirm, self.pol_confirm):
            box.value = False

    def _status_text(self) -> str:
        """Same wyjatki (pusty tekst = nic sie nie dzieje); stan ramienia jest na plakietce."""
        st = self.twin.status
        parts = []
        moved = [c.name for c in self.ws.cameras if c.enabled and self.watch.moved(c.name)]
        if moved:
            parts.append(f"**kamera przestawiona: {', '.join(moved)}**")
        if self.runner is not None and self.runner.status.running:
            parts.append(f"Polityka: jedzie ({self.runner.status.step} krokow)")
        if self.train.running:
            p = self.train.progress() or {}
            parts.append(f"Trening: {p.get('iteration', 0)}/{p.get('iterations', '?')} it.")
        # Trwale ostrzezenia ramienia (kalibracja chwytaka niezgodna z blizniakiem, chwytak
        # poluzowany po przeciazeniu) - `RobotStatus.warnings`; starszy status ich nie ma.
        warns = [str(w) for w in (getattr(st, "warnings", None) or [])]
        if warns:
            parts.append(f"**Uwaga:** {'; '.join(warns)}")
        if st.error:
            parts.append(f"**blad ramienia:** {st.error}")
        return " | ".join(parts)

    # ------------------------------------------------------------ przewodnik
    def _guide_snapshot(self) -> gd.GuideSnapshot:
        """Stan panelu dla przewodnika - same odczyty z pamieci (bez dysku, renderu i kamer)."""
        st = self.twin.status
        connected = bool(st.connected)
        backend = st.backend if connected else self._preset_backend()
        simulated = bool(st.simulated) if connected else backend == "sim"
        estop = self._estop_active()
        now = time.monotonic()
        with self.frame_lock:
            live = set(self.frames)
        cams = []
        for c in self.ws.cameras:
            problem = c.intrinsics_problem()
            # "Bez kadru" dopiero po GUIDE_FRAME_LOST s bez kadru (nowa kamera liczy od pojawienia sie).
            seen = self._frame_seen.setdefault(c.name, now)
            if c.name in live:
                self._frame_seen[c.name] = seen = now
            cams.append(gd.CameraSnap(
                c.name, c.simulated, c.enabled, has_frame=now - seen < GUIDE_FRAME_LOST, intrinsics_ok=not problem,
                intrinsics_problem=problem, calibrated=c.calibrated, trusted=c.trusted,
                reason=str(c.calibration.get("reason", "") or ""), moved=self.watch.moved(c.name),
                simreal_px=self._simreal_px.get(c.name)))
        runner = self.runner
        running = runner is not None and runner.status.running
        vision = runner is not None and runner.cube_provider == self._vision_cube
        if runner is not None and runner.status.success and self._runner_backend:
            # Krok 7/8: polityka choc raz doszla do celu - zapisane pod backendem, na ktorym
            # RUSZYLA (`_start_policy`), nie pod obecnym: stary runner z sukcesem w sim po
            # Polacz feetech zaliczal "lift z kamer" prawdziwemu ramieniu, ktore nie drgnelo.
            self._ran.setdefault(self._runner_backend, set()).add(runner.task.name + ("-kamery" if vision else ""))
        jobs_on = (("kalibracja kamer", self.calib_job.running), ("pomiar obiektywu", self.intr_job.running),
                   ("identyfikacja", self.sysid_job.running), ("trening", self.train.running),
                   ("ewaluacja", self.eval_job.running))
        dyn = self.ws.dynamics
        dyn_src = str(dyn.get("source") or "zmierzona") if dyn else ""
        return gd.GuideSnapshot(
            connected=connected, backend=backend, simulated=simulated, arm_error=str(st.error or ""), estop=estop,
            owner=self._busy(PANEL_OWNER), warnings=tuple(str(w) for w in (getattr(st, "warnings", None) or [])),
            cameras=tuple(cams), dynamics=dyn_src, dynamics_backend=gd.dynamics_backend(dyn_src),
            dynamics_useful=gd.dynamics_useful(dyn),
            dyn_result_useful=bool(self.sysid_job.data.get("useful", True)),
            policies=tuple((p["name"], p["task"], bool(p.get("bundled"))) for p in self._policy_list),
            jobs=tuple(n for n, on in jobs_on if on), policy_running=runner.task.name if running else "",
            policy_from_cameras=running and vision, calib_result_pending=self._calib_pending,
            dyn_result_pending=bool(self.dyn_keep.visible),
            calib_progress=float(self.calib_job.progress) if self.calib_job.running else None,
            ran=frozenset(self._ran.get(backend, ()) if connected else ()),
            intr_camera=str(self.intr_job.data.get("camera", "")) if self.intr_job.running else "")

    def _tick_guide(self) -> None:
        note = getattr(self.twin, "home_note", "")
        if note and self.home_md.content != f"**{note}**":
            self._msg(self.home_md, f"**{note}**")
        # Przypisanie tej samej tresci nic nie wysyla (`_skip_unchanged_markdown`).
        guide = gd.build(self._guide_snapshot())
        self._guide_last = guide
        self.guide_md.content = gd.render_now(guide)
        notes = gd.render_notes(guide)
        self.guide_notes_md.content = notes
        self.guide_steps_md.content = gd.render_steps(guide)
        n = notes.count("**Uwaga:**")
        set_if(self.guide_folder, "label", f"Uwagi ({n}) i kroki 0-8" if n else "Kroki 0-8")

    # ================================================================== ramie
    def _build_arm(self) -> None:
        """Zakladka Ramie jak pilot jog: rozlaczone - zestaw i Polacz; polaczone - duzy UZBROJ,
        Uchwyt 3D, Dom, Ustaw dom = tu, predkosc, suwaki stawow, Rozlacz; stol zwiniety na dole."""
        g = self.server.gui
        presets = self._arm_presets()
        labels = tuple(presets)
        current = next((k for k, (b, p) in presets.items()
                        if b == self.ws.backend and (p is None or p == (self.ws.port or p))), labels[0])
        # --- rozlaczone: zestaw, port, Polacz (folder bez etykiety = bez ramki)
        self.conn_box = g.add_folder(None)
        with self.conn_box:
            self.arm_backend = g.add_dropdown("Ramie", labels, initial_value=current,
                                              hint="Gotowe zestawy dla ramienia tego stanowiska: symulacja albo "
                                                   "sprzet z domyslnym laczem (A1X: xcan na macOS, can0 na Linuksie, "
                                                   "PCAN na Windows; SO-101: feetech przez port). Zestaw wpisuje "
                                                   "Port, ktory mozna potem poprawic. Dziala po Polacz.")
            self.arm_port = g.add_text("Port", self.ws.port or (presets[current][1] or ""),
                                       hint="feetech/lerobot: COMx, /dev/ttyACM0 albo socket://adres:5555 "
                                            "(`lerobot-twin check` wypisuje porty). a1x: can0 (Linux), xcan "
                                            "albo xcan:6 (macOS), PCAN_USBBUS1 (Windows); pusty = domyslny. "
                                            "sim: nieuzywany.")
            connect = g.add_button("Polacz", icon=viser.Icon.PLUG_CONNECTED, color="green",
                                   hint="Laczy wybrane ramie BEZ ruchu: ramie zostaje tam, gdzie stoi, suwaki "
                                        "pokazuja zmierzona poze. Konczy wszystko, co jezdzi, i kasuje "
                                        "potwierdzenia. Backend i port zapisuje w stanowisku.")

        @self.arm_backend.on_update
        def _(event):
            port = presets.get(self.arm_backend.value, (None, None))[1]
            if port is not None:
                self.arm_port.value = port
            self._render_arm(force=True)

        # --- polaczone: sterowanie
        self.ctl_box = g.add_folder(None)
        with self.ctl_box:
            g.add_html('<span id="tw-arm"></span>')
            arm_on = g.add_button("UZBROJ", color="gray", icon=viser.Icon.LOCK,
                                  hint="Uzbrojony panel RUSZA ramieniem: suwaki i Uchwyt 3D ustawiaja cel od "
                                       "ZMIERZONEJ pozy. Odmowa przy STOP-ie albo gdy ramie ma polityka, fala, "
                                       "identyfikacja; STOP, Dom i Polacz rozbrajaja.")
            arm_off = g.add_button("UZBROJONE - kliknij, by rozbroic", color="orange", icon=viser.Icon.BOLT,
                                   visible=False,
                                   hint="Rozbraja panel: ramie trzyma poze, suwaki znow pokazuja zmierzone katy.")
            self.arm_engage = _Toggle(arm_on, arm_off)
            g.add_html('<span id="tw-3d"></span>')
            gz_btn = g.add_button("Uchwyt 3D: wyl.", color="gray", icon=viser.Icon.ARROWS_MOVE,
                                  hint="Strzalki na koncowce w 3D. Uzbrojony panel: przeciaganie rusza "
                                       "ramieniem (IK, sama pozycja); skok stawu > 15 st. jest odrzucany - "
                                       "ciagnij malymi krokami. Klik = pokaz/ukryj.")
            self.tcp_gizmo_on = _Toggle(gz_btn, on_look=GIZMO_ON_IDLE, off_look=GIZMO_OFF)
            self.tcp_md = g.add_markdown("", visible=False)
            self.home_btn = g.add_button("Dom", icon=viser.Icon.HOME, color="orange",
                                         hint="Jedzie do pozycji domowej stanowiska (Ustaw dom = tu) albo z modelu - "
                                              "RUSZA ramieniem, odbiera je kazdemu. A1X: etapami (lokiec, "
                                              "nadgarstek, bark), kazdy etap sprawdzony na kolizje w blizniaku; "
                                              "kolizja = ramie stoi, powod nizej. Sciskajacy chwytak zostaje "
                                              "zamkniety. Przy STOP-ie nie rusza.")
            self.home_md = g.add_markdown("", visible=False)
            self.set_home_btn = g.add_button("Ustaw dom = tu", icon=viser.Icon.MAP_PIN, color="gray",
                                             hint="Zmierzona poza, dokladnie ta, staje sie domem tego stanowiska "
                                                  "(zapis w twin.json); limity poszerzaja sie, zeby ja objac. Zrob "
                                                  "to w pozie, w ktorej ramie naprawde spoczywa (zlozone na "
                                                  "podstawie). Niczym nie rusza.")
            top = max((self.ws.spec().max_vel.get(n, 0.0) for n in self.ws.spec().arm_joints), default=90.0)
            self.arm_speed = g.add_slider("Predkosc [st./s]", 1.0, float(top), 1.0, float(min(20.0, top)),
                                          hint="Najwieksza predkosc stawow przy suwakach i uchwycie (jak 'speed' "
                                               "w jog_a1x). Dom jedzie po swojemu: 20 st./s etapami (A1X) albo "
                                               "rampa nadzoru.")
            spec = self.ws.spec()
            kin = self.twin.scene.kin
            names = JOINT_LABELS.get(self.ws.robot, {})
            self.sliders = {}
            self.slider_range = {}
            order = [n for n in spec.joints if n != spec.gripper] + ([spec.gripper] if spec.gripper in spec.joints
                                                                      else [])
            for name in order:
                k = list(spec.joints).index(name)
                if name == spec.gripper:
                    g.add_divider()
                    lo, hi = 0.0, 100.0
                    tip = ("Chwytak 0..100: 0 = szczeki zamkniete, 100 = otwarte (SO-101: tiki serwa jak w "
                           "feetech; A1X: skok palca 0..5 cm). Uzbrojony panel ustawia cel. Chwytak, ktory po "
                           "STOP-ie dalej sciska: Skasuj STOP, UZBROJ, potem ten suwak.")
                else:
                    # Zaokraglone do 0,5 st. do srodka zakresu MJCF - ladne liczby na suwaku, zadnej poza zakresem.
                    lo = float(np.ceil(np.degrees(kin.lo[k]) * 2) / 2)
                    hi = float(np.floor(np.degrees(kin.hi[k]) * 2) / 2)
                    tip = (f"Kat stawu [st.], suwak {lo:g}..{hi:g} (zakres modelu). Uzbrojony panel ustawia cel - "
                           f"nadzor przycina go do limitow z konfiguracji i EEPROM serwa; rozbrojony pokazuje "
                           f"zmierzony kat.")
                label, full = names.get(name, (name, name))
                s = g.add_slider(label, lo, hi, 0.5, float(np.clip(spec.home.get(name, 0.0), lo, hi)),
                                 hint=f"{full} ({name}). {tip}")
                self.sliders[name] = s
                self.slider_range[name] = (lo, hi)

                def on_slide(event, name=name):
                    if event.client is None or not self.arm_engage.value:
                        return                          # zmiana z petli (sprzezenie zwrotne) albo bez uzbrojenia
                    self._gizmo_seed = None             # cel zmienil sie obok uchwytu TCP
                    try:
                        self.twin.set_target({name: float(self.sliders[name].value)}, owner=PANEL_OWNER)
                    except RuntimeError:                # ramie odebrane panelowi w miedzyczasie
                        self.arm_engage.value = False
                s.on_update(on_slide)

            disconnect = g.add_button("Rozlacz", icon=viser.Icon.PLUG_CONNECTED_X, color="gray",
                                      hint="Konczy polityke, fale i identyfikacje, odbiera ramie wszystkim i "
                                           "rozlacza backend. Silnikow nie zwalnia - ramie trzyma poze. "
                                           "Zmiana zestawu: Rozlacz, potem Polacz.")
            # Galaxea A1X: sekwencja wlaczenia FF 1-5-6 tylko recznie (zasilone od nowa ramie slucha od
            # razu, a FF 5 na chwile puszcza silniki). Widoczne tylko z polaczonym backendem a1x.
            self.a1x_folder = g.add_folder("Silniki A1X", expand_by_default=False)
            with self.a1x_folder:
                self.a1x_confirm = g.add_checkbox(
                    "Prawdziwe ramie: rozumiem, ze silniki na chwile puszcza", False,
                    hint="Potwierdzenie przed kazdym Wlacz silniki A1X (gasnie po uzyciu). FF 5 na ulamek sekundy "
                         "zwalnia silniki - ramie bez hamulcow moze lekko opasc; podeprzyj je.")
                a1x_enable = g.add_button("Wlacz silniki A1X (FF 1-5-6)", icon=viser.Icon.BOLT,
                                          hint="Tylko gdy ramie zglasza sie, ale nie slucha rozkazow (blad 'gluche "
                                               "na 0x050'). Wysyla FF 1 -> 5 -> 6 z celem = zmierzona poza przed, "
                                               "miedzy i po kazdym kodzie. Odmowa, gdy panel jest uzbrojony albo "
                                               "ramie ma polityka / fala / identyfikacja.")
                self.a1x_md = g.add_markdown("")

        def show_a1x() -> None:
            on = self.twin.status.connected and self.twin.status.backend == "a1x"
            self.a1x_folder.visible = on
            if not on:
                self.a1x_confirm.value = False

        show_a1x()

        @a1x_enable.on_click
        @self._safe
        def _(event):
            st, b = self.twin.status, self.twin.backend
            if not (st.connected and st.backend == "a1x") or not hasattr(b, "enable"):
                raise RuntimeError("wlaczenie silnikow tylko dla polaczonego ramienia a1x")
            if self.arm_engage.value:
                raise RuntimeError("najpierw rozbroj panel (kliknij UZBROJONE)")
            busy = self._busy(PANEL_OWNER)
            if busy:
                raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj")
            if not self.a1x_confirm.value:
                raise RuntimeError("zaznacz potwierdzenie, ze silniki na chwile puszcza")
            self.a1x_confirm.value = False
            self.a1x_md.content = "wlaczanie: FF 1 -> 5 -> 6, cel = zmierzona poza..."

            def work():
                try:
                    self.a1x_md.content = f"**Silniki A1X:** {b.enable()}"
                except Exception as exc:                  # odmowa backendu (cisza na laczu, FF 2)
                    logger.exception("Wlaczanie silnikow A1X")
                    self.a1x_md.content = f"**Silniki A1X:** odmowa - {exc}"
            threading.Thread(target=work, name="a1x-wlacz", daemon=True).start()

        @connect.on_click
        @self._safe
        def _(event):
            try:
                self._connect(self._preset_backend(), self.arm_port.value.strip() or None, False)
            finally:
                show_a1x()
                self._render_arm(force=True)
            self._notify(event, "Polaczono", f"{self.twin.status.backend}: {self.twin.status.state}")

        @disconnect.on_click
        @self._safe
        def _(event):
            self._stop_motion("rozlaczenie")
            self._reset_confirmations()
            self.twin.disconnect()
            show_a1x()
            self._render_arm(force=True)

        # --- stol (zawsze, zwiniete)
        with g.add_folder("Stanowisko (stol)", expand_by_default=False):
            self._build_table()

        self.twin.set_speed(self.arm_speed.value)

        @self.arm_speed.on_update
        def _(event):
            self.twin.set_speed(float(self.arm_speed.value))

        @self.arm_engage.on_update
        def _(event):
            self._on_engage(event)

        @self.arm_engage.on_update
        def _(event):
            self._render_arm()                          # suwaki, uchwyt, plakietka od razu po zmianie

        @self.tcp_gizmo_on.on_update
        def _(event):
            self._render_arm()

        @self.home_btn.on_click
        @self._safe
        def _(event):
            self._msg(self.home_md, "")
            self._go_home()

        @self.set_home_btn.on_click
        @self._safe
        def _(event):
            h = self.twin.set_home_here()
            self._notify(event, "Dom zapisany", ", ".join(f"{names.get(k, (k, k))[0]} {v:.1f}" for k, v in h.items()))

        self._render_arm(force=True)

    # ---------------------------------------------------------- widok Ramie
    @staticmethod
    def _msg(md, text: str) -> None:
        """Markdown widoczny tylko, gdy ma tresc."""
        md.content = text
        set_if(md, "visible", bool(text))

    def _estop_active(self) -> bool:
        s = self.twin.safety_state                      # None bez nadzoru (rozlaczone)
        return s is not None and getattr(s, "value", s) == "ESTOP"

    def _arm_mode(self) -> tuple[str, str]:
        """(tryb, kto): jedno zrodlo dla plakietki i przycisku UZBROJ."""
        if self._estop_active():
            return "estop", ""
        if not self.twin.connected:
            # Petla padla (blad lacza), a backend wciaz otwarty: Rozlacz musi byc dostepne.
            return ("lost" if self.twin.backend is not None else "off"), ""
        o = self.twin.owner
        if o is not None and o != PANEL_OWNER:
            return "busy", str(o)
        st = self.twin.safety_state
        if o is None and getattr(st, "value", st) in ("HOMING", "STARTING"):
            return "homing", ""                          # rampa do domu bez wlasciciela (SO-101)
        return ("armed" if self.arm_engage.value else "idle"), ""

    def _render_arm(self, force: bool = False) -> None:
        """Przemalowanie zakladki Ramie do stanu blizniaka (tylko przy zmianie klucza; same zapisy
        wlasciwosci, zadnych ruchow). Wolane z petli (30 Hz), po Polacz/Rozlacz i po zmianie wlacznikow."""
        with self._render_lock:
            mode, who = self._arm_mode()
            estop = mode == "estop"
            conn = self.twin.connected or self.twin.backend is not None      # Rozlacz takze po padnieciu petli
            armed = self.arm_engage.value
            key = (mode, who, armed, conn, self.tcp_gizmo_on.value, self._preset_backend(),
                   bool(self.tcp_md.content), bool(self.home_md.content), self.twin.status.backend if conn else None)
            if key == self._arm_key and not force:
                return
            self._arm_key = key
            set_if(self.conn_box, "visible", not conn)
            set_if(self.ctl_box, "visible", conn)
            set_if(self.arm_port, "visible", self._preset_backend() != "sim")
            set_if(self.clear_btn, "visible", estop)
            look = ARM_OVERRIDES.get(mode)
            self.arm_engage.set_override(look(who) if look else None)
            self.tcp_gizmo_on.on_look = GIZMO_ON_ARMED if armed else GIZMO_ON_IDLE
            self.tcp_gizmo_on.set_override(GIZMO_ESTOP if estop else None)
            self.tcp_gizmo_on._apply()
            set_if(self.home_btn, "disabled", not conn)     # Dom takze pod STOP-em: sam go kasuje
            set_if(self.set_home_btn, "disabled", not conn)
            if armed != self._sliders_armed:
                self._sliders_armed = armed
                for s in self.sliders.values():
                    set_if(s, "disabled", not armed)
            for md in (self.tcp_md, self.home_md):
                set_if(md, "visible", bool(md.content))
            self._render_chip(mode, who)

    def _state_chip_html(self, mode: str, who: str) -> str:
        word = {"off": "ROZLACZONE", "idle": "ROZBROJONE", "armed": "UZBROJONE", "homing": "JEDZIE DO DOMU",
                "busy": f"STERUJE: {who.upper()}", "estop": "STOP AKTYWNY", "lost": "PETLA PADLA"}[mode]
        st = self.twin.status
        if mode == "lost":
            meta = (str(st.error) or "blad lacza") + " - Rozlacz"
        elif st.connected:
            backend = "symulacja" if st.backend == "sim" else str(st.backend)
            port = f" {self.ws.port}" if self.ws.port and st.backend != "sim" else ""
            meta = f"{backend}{port} - {round(float(st.loop_hz or 0.0) / 5) * 5:.0f} Hz"
        else:
            meta = "wybierz zestaw i Polacz"
        return (f'<div class="tw-chip tw-{mode}"><span class="tw-word">{_html.escape(word)}</span>'
                f'<span class="tw-meta">{_html.escape(meta)}</span></div>')

    def _render_chip(self, mode: str, who: str) -> None:
        s = self._state_chip_html(mode, who)
        if s != self._chip_last:
            self._chip_last = s
            self.state_html.content = s

    def _arm_presets(self) -> dict[str, tuple[str, str | None]]:
        """Zestawy Ramie -> (backend, port albo None = z pola). Pierwszy = domyslny dla tego systemu."""
        import platform
        sim = {"symulacja (blizniak jest ramieniem)": ("sim", "")}
        if self.ws.robot == "a1x":
            can = {"Galaxea A1X po xcan (macOS)": ("a1x", "xcan"),
                   "Galaxea A1X po can0 (Linux)": ("a1x", "can0"),
                   "Galaxea A1X po PCAN (Windows)": ("a1x", "PCAN_USBBUS1")}
            first = {"Darwin": "Galaxea A1X po xcan (macOS)", "Linux": "Galaxea A1X po can0 (Linux)",
                     "Windows": "Galaxea A1X po PCAN (Windows)"}.get(platform.system())
            ordered = {}
            if first:
                ordered[first] = can[first]
            ordered.update(can)
            ordered.update(sim)
            return ordered
        return {"SO-101 przez port (feetech)": ("feetech", None), **sim,
                "SO-101 przez LeRobota (lerobot)": ("lerobot", None)}

    def _preset_backend(self) -> str:
        return self._arm_presets().get(str(self.arm_backend.value), ("sim", None))[0]

    def _go_home(self) -> None:
        """Pozycja domowa: odbiera ramie kazdemu (panel, polityka, fala) - takze startowi w toku.
        Pod STOP-em najpierw go kasuje: po zatrzymaniu zawsze ma byc jedno klikniecie do domu
        (ramie bez hamulcow w gorze - operator nie ma szukac zoltego przycisku)."""
        self._motion_gen += 1
        if self._estop_active():
            self.twin.clear_estop()
        self.twin.home()
        self._gizmo_seed = None

    def _halt_policy(self) -> None:
        """Polityki > Zatrzymaj: takze polityka, ktora dopiero laduje sie po Uruchom, nie ruszy."""
        self._motion_gen += 1
        if self.runner is not None:
            self.runner.stop("zatrzymana z panelu")

    def _connect(self, backend: str, port: str | None, go_home: bool = False, **kw) -> None:
        """Polacz: najpierw koniec wszystkiego, co jezdzi, i nowe potwierdzenia (moze to byc inne ramie)."""
        self._stop_motion("ponowne laczenie")
        self._reset_confirmations()
        self.twin.connect(backend, port, go_home=go_home, **kw)
        self._save()

    def _on_engage(self, event) -> None:
        """Sprzeglo z panelu (tylko zmiany od uzytkownika - te z petli maja `client` None)."""
        if event.client is None:
            return                                      # zmiana z petli / z kodu panelu
        self._engage_t = time.monotonic()
        if not self.arm_engage.value:
            self._release_panel()
            return
        if not self.twin.connected:
            self.arm_engage.value = False
            self._notify(event, "Brak ramienia", "Najpierw polacz ramie (np. sim).", error=True)
            return
        # Sprzeglo = panel bierze ramie na wlasnosc. Gdy jedzie fala, identyfikacja albo
        # polityka - odmowa z powodem; wczesniej suwaki pisaly cel na zmiane z nimi.
        try:
            busy = self._busy(PANEL_OWNER)
            if busy:
                raise RuntimeError(f"ramie zajete: {busy}")
            self.twin.claim(PANEL_OWNER, preempt=self._panel_preempted)
        except RuntimeError as exc:
            self.arm_engage.value = False
            self._notify(event, "UZBROJ", f"{exc} - zatrzymaj je najpierw.", error=True)
            return
        self._gizmo_seed = None
        self._gizmo_t = 0.0                         # uchwyt TCP od razu na koncowce
        # Sprzeglo tylko, jesli ramie wciaz nalezy do panelu - sprawdzenie i wlaczenie razem
        # (pod blokada wlasnosci w `Twin`); odebrane (STOP, Dom) miedzy claim a sprzeglem - odmowa.
        if not self.twin.set_engaged(True, owner=PANEL_OWNER):
            self.arm_engage.value = False
            if self._estop_active():
                self._notify(event, "UZBROJ", "STOP aktywny - najpierw Skasuj STOP.", error=True)

    def _build_table(self) -> None:
        """Stol (zwiniety folder na dole zakladki Ramie) i uchwyt TCP w 3D."""
        g = self.server.gui
        t = self.ws.table_obj()
        g.add_markdown("Zmierz blat i polozenie podstawy. Kamery sa wzgledem PODSTAWY, zmiana stolu ich "
                       "nie rusza.")
        self.tab_size = g.add_vector2("Blat: szerokosc, glebokosc [m]", tuple(t.size), min=(0.2, 0.2),
                                      max=(3.0, 3.0), step=0.01,
                                      hint="Wymiary blatu [m] (0,2-3,0), zmierz na biurku. Dziala po Zastosuj.")
        self.tab_base = g.add_vector2("Podstawa od srodka blatu [m]", tuple(t.base_xy), min=(-1.5, -1.5),
                                      max=(1.5, 1.5), step=0.005,
                                      hint="Gdzie stoi podstawa ramienia wzgledem srodka blatu, x i y [m]. "
                                           "Dziala po Zastosuj.")
        self.tab_yaw = g.add_number("Obrot podstawy [st.]", float(np.degrees(t.base_yaw)), min=-180.0,
                                    max=180.0, step=1.0,
                                    hint="Obrot podstawy wokol osi pionowej [st.], -180..180. Dziala po Zastosuj.")
        self.tab_h = g.add_number("Wysokosc blatu nad podloga [m]", float(t.height), min=0.3, max=1.5,
                                  step=0.01, hint="Wysokosc blatu [m] (0,3-1,5); zle ustawiona psuje Sim-Real. "
                                                  "Dziala po Zastosuj.")
        apply_table = g.add_button("Zastosuj", icon=viser.Icon.CHECK, color="gray",
                                   hint="Zapisuje stol i przebudowuje scene blizniaka. Ramieniem nie rusza, "
                                        "kamer nie przestawia (sa wzgledem podstawy).")

        @apply_table.on_click
        @self._safe
        def _(event):
            self.ws.table.update(size=list(self.tab_size.value), base_xy=list(self.tab_base.value),
                                 base_yaw=float(np.radians(self.tab_yaw.value)), height=float(self.tab_h.value))
            self._save()
            self._simreal_px.clear()                    # inny stol - Sim-Real do sprawdzenia od nowa
            self.twin.rebuild()
            T_new = self.twin.scene.T_base2world
            self.base_frame.position, self.base_frame.wxyz = T_new[:3, 3], mat_to_wxyz(T_new[:3, :3])
            self._refresh_cameras()
            if self._goal_base is not None:           # cel reach jest w ukladzie podstawy - kula za nia
                self._set_goal(self._goal_base)
            self._notify(event, "Stanowisko", "Stol zapisany, scena przebudowana.")

        self.tcp_gizmo = self.server.scene.add_transform_controls("/uchwyt_tcp", scale=0.12, disable_rotations=True,
                                                                  visible=False)

        @self.tcp_gizmo_on.on_update
        def _(event):
            self.tcp_gizmo.visible = self.tcp_gizmo_on.value
            self._gizmo_seed, self._gizmo_t = None, 0.0

        @self.tcp_gizmo.on_update
        def _(event):
            self._on_tcp_gizmo()

    def _on_tcp_gizmo(self) -> None:
        """Przeciagniety uchwyt TCP -> cel stawow, tylko na tej samej galezi IK co ramie teraz.

        IK startuje WYLACZNIE z ostatniego przyjetego rozwiazania (albo z rozkazu, gdy
        przeciaganie sie zaczyna) i bez losowych startow; rozwiazanie dalej niz
        `GIZMO_MAX_STEP_DEG` od startu jest odrzucane - to druga galaz, nie ruch o 1 cm.
        """
        if not (self.arm_engage.value and self.tcp_gizmo_on.value and self.twin.connected
                and self.twin.owner == PANEL_OWNER):
            return
        p_base = (inverse(self.T_b2w) @ np.r_[self.tcp_gizmo.position, 1.0])[:3]
        gripper = self.ws.spec().gripper
        with self._gizmo_lock:
            self._gizmo_t = time.monotonic()
            cmd = {k: float(v) for k, v in (self.twin.status.command or self.twin.joints()).items()}
            seed = dict(self._gizmo_seed) if self._gizmo_seed is not None else cmd
            # restarts=0: `ik` probuje wtedy start z `seed` i z pozycji domowej - ta druga
            # wygrywa tylko, gdy start z `seed` nie trafil, i wtedy odrzuca ja limit skoku.
            sol = self._kin_gizmo.ik(p_base, seed=seed, restarts=0)
            if not sol.ok:
                self._msg(self.tcp_md, "uchwyt poza zasiegiem ramienia - cel bez zmian")
                return
            arm = {k: float(v) for k, v in sol.joints.items() if k != gripper}
            jump = max((abs(v - float(seed.get(k, v))) for k, v in arm.items()), default=0.0)
            if jump > GIZMO_MAX_STEP_DEG:
                self._msg(self.tcp_md, f"odrzucone: IK chcialo skoku {jump:.0f} st. (inna galaz) - "
                                       f"przeciagaj mniejszymi krokami")
                return
            self._gizmo_seed = {**seed, **arm}
        self._msg(self.tcp_md, "")
        try:
            self.twin.set_target(arm, owner=PANEL_OWNER)
        except RuntimeError:                            # ramie odebrane panelowi w miedzyczasie
            self.arm_engage.value = False

    def _tick_arm(self) -> None:
        st = self.twin.status
        joints = st.measured if st.connected and st.measured else self.twin.joints()
        owner = self.twin.owner
        if self.arm_engage.value and owner != PANEL_OWNER and time.monotonic() - self._engage_t > 0.5:
            # Ramie odebrane panelowi (STOP, Dom, polaczenie, petla padla) albo wziete przez zadanie.
            self.arm_engage.value = False
            self._gizmo_seed = None
        if not self.arm_engage.value:
            for name, s in self.sliders.items():
                if name in joints:
                    s.value = round(float(np.clip(joints[name], *self.slider_range[name])), 1)
        driving = (self.arm_engage.value and owner == PANEL_OWNER
                   and time.monotonic() - self._gizmo_t < GIZMO_IDLE_S)
        if self.tcp_gizmo_on.value and not driving:
            # Uchwyt wraca na koncowke, gdy nikt go nie ciagnie: po Domu, przebudowie, polityce
            # czy suwakach nastepne male przesuniecie celowalo w STARE miejsce koncowki.
            with self.twin.lock:
                d, site = self.twin.scene.data, self.twin.scene.kin.site_id
                p = d.site_xpos[site].copy()
            if np.abs(np.asarray(self.tcp_gizmo.position) - p).max() > 1e-4:
                self.tcp_gizmo.position = p
            self._gizmo_seed = None
        self._render_arm()

    # ================================================================= kamery
    def _big_button(self, label: str, **kw):
        """Duzy przycisk we wlasnym pudelku bez ramki: kotwica stylu `.tw-big` i przycisk.
        Pokazuje sie i chowa pudelko - ukryty przycisk nie oddaje stylu sasiadowi."""
        g = self.server.gui
        box = g.add_folder(None)
        with box:
            g.add_html('<span class="tw-big"></span>')
            b = g.add_button(label, **kw)
        return box, b

    def _build_cameras(self) -> None:
        """Zakladka Kamery: lista z jedna plakietka stanu na kamere, wybrana kamera (podglad i tylko
        przyciski, ktore do niej pasuja), na dole dodawanie. Bez kamer - podpowiedz i dodawanie."""
        g = self.server.gui
        self.cams_html = g.add_html("")
        self.cam_box = g.add_folder(None)
        with self.cam_box:
            self.cam_pick = g.add_dropdown("Kamera", ("-",), initial_value="-",
                                           hint="Kamera, ktorej dotyczy podglad i przyciski ponizej.")
            self.cam_preview = g.add_image(np.zeros((240, 320, 3), np.uint8), format="jpeg", jpeg_quality=70)
            self.cam_info = g.add_markdown("")
            en = g.add_button(CAM_ON[0], hint="Wylaczona kamera znika ze sceny, mapy, fali i Sim-Real, ale zostaje "
                                              "w stanowisku razem z kalibracja. Zmiana przebudowuje scene.")
            self.cam_enabled = _Toggle(en, on_look=CAM_ON, off_look=CAM_OFF)
            self.cam_enabled.value = True
            self.cam_sim_box = g.add_folder(None)
            with self.cam_sim_box:
                mv = g.add_button(MOVE_OFF[0], hint="Kamera symulowana: uchwyt w 3D - przeciagnij, zeby ja przestawic "
                                                    "(zmienia sie prawda, nie kalibracja).")
                self.cam_move = _Toggle(mv, on_look=MOVE_ON, off_look=MOVE_OFF)
                here = g.add_button("Postaw tu, skad patrzysz w 3D", icon=viser.Icon.CAMERA_SELFIE, color="gray",
                                    hint="Kamera symulowana staje tam, skad teraz patrzysz na scene 3D.")
            self.cam_remove = g.add_button("Usun kamere", icon=viser.Icon.TRASH, color="red",
                                           hint="Usuwa kamere razem z jej kalibracja. Drugie klikniecie potwierdza.")
            with g.add_folder("Szczegoly", expand_by_default=False):
                self.cam_detail = g.add_markdown("")
                remember = g.add_button("Zapamietaj kadr odniesienia", icon=viser.Icon.PHOTO_CHECK,
                                        hint="Biezacy kadr jako odniesienie: od niego straznik liczy przestawienie "
                                             "kamery (co 2 s, prog 3 px). Po zapisie kalibracji robi sie samo.")
        g.add_divider()
        g.add_markdown("**Dodaj kamere**")
        add_front = g.add_button("Dodaj symulowana przed ramieniem", icon=viser.Icon.CAMERA_PLUS,
                                 hint="Nowa kamera symulowana 640x480 ok. 0,55 m od ramienia, patrzaca na blat; "
                                      "kolejne staja z innych stron. Przestawiasz ja potem uchwytem w 3D.")
        probe = g.add_button("Szukaj kamer USB", icon=viser.Icon.SEARCH,
                             hint="Szuka kamer USB (IDS uEye, gdy jest pyueye; pmd CamBoard nano, gdy jest "
                                  "PMDSDK), ktorych nie ma jeszcze w "
                                  "stanowisku. Na maszynie wirtualnej/zdalnej kamera musi byc "
                                  "przepuszczona w kliencie USB.")
        self.usb_box = g.add_folder(None, visible=False)
        with self.usb_box:
            self.usb_pick = g.add_dropdown("Znaleziona", ("-",), initial_value="-",
                                           hint="Kamera z ostatniego Szukaj kamer USB: zrodlo i rozdzielczosc.")
            add_usb = g.add_button("Dodaj kamere USB", icon=viser.Icon.PLUS, color="green",
                                   hint="Dodaje wybrana kamere USB z nominalnym K - potem zakladka Kalibracja. "
                                        "Kamera pmd dostaje K z wlasnej kalibracji, a poze z tagow na obudowie "
                                        "(folder Kamera PMD ponizej).")
        with g.add_folder("Nowa symulowana: pole widzenia", expand_by_default=False):
            self.sim_fov = g.add_slider("Pole widzenia [st.]", 40.0, 90.0, 1.0, 62.0,
                                        hint="Pionowe pole widzenia NOWEJ kamery symulowanej [st.], 40-90; "
                                             "typowa kamera USB ok. 60.")
        self._build_pmd(g)
        self.cam_gizmo = self.server.scene.add_transform_controls("/uchwyt_kamery", scale=0.1, visible=False)
        self._found_usb: list[dict] = []
        self.cam_ask = _TwoClick()

        @self.cam_pick.on_update
        def _(event):
            self.cam_ask.reset()
            self._select_camera()

        @self.cam_enabled.on_update
        @self._safe
        def _(event):
            if event.client is None:
                return                                  # ustawione z kodu przy wyborze kamery
            rec = self._cam()
            if rec is not None:
                rec.enabled = self.cam_enabled.value
                self._save()
                self.twin.rebuild()
                self.twin.cameras.sync()
                self._refresh_cameras()

        @self.cam_move.on_update
        def _(event):
            self._select_camera()

        @self.cam_gizmo.on_update
        def _(event):
            rec = self._cam()
            if rec is None or not rec.simulated or not rec.enabled or not self.cam_move.value:
                return
            T_w = pose(wxyz_to_mat(self.cam_gizmo.wxyz), np.asarray(self.cam_gizmo.position))
            self._place_sim_camera(rec, T_w, rebuild=False)

        @here.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None or not rec.simulated:
                raise RuntimeError("to dziala dla kamery symulowanej - prawdziwa stoi tam, gdzie ja postawiles")
            self._place_sim_camera(rec, self._viewer_pose(event), rebuild=True)

        @remember.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None:
                return
            self._remember_reference(rec.name)
            self._notify(event, "Zapamietano", f"{rec.name}: kadr odniesienia do wykrywania przestawienia.")

        @self.cam_remove.on_click
        @self._safe
        def _(event):
            rec = self._cam()
            if rec is None:
                return
            if not self.cam_ask.ask(("usun", rec.name)):
                self._render_cameras()                  # pytanie na przycisku, nic nie usuniete
                return
            self.ws.remove_camera(rec.name)
            self.watch.forget(rec.name)
            self._save()
            self.twin.rebuild()
            self.twin.cameras.sync()
            self._refresh_cameras()

        @probe.on_click
        @self._safe
        def _(event):
            from ..cameras import probe_all
            used = {c.source for c in self.ws.cameras}
            self._found_usb = [d for d in probe_all() if d["source"] not in used]
            opts = tuple(d["label"] for d in self._found_usb) or ("-",)
            self.usb_pick.options = opts
            self.usb_pick.value = opts[0]
            self.usb_box.visible = bool(self._found_usb)
            if not self._found_usb:
                self._notify(event, "Brak nowych kamer", "Sprawdz kabel; na maszynie wirtualnej/zdalnej "
                                                         "kamere trzeba przepuscic w kliencie USB.")

        @add_usb.on_click
        @self._safe
        def _(event):
            if not self._found_usb or self.usb_pick.value == "-":
                raise RuntimeError("najpierw wyszukaj kamery")
            d = self._found_usb[[x["label"] for x in self._found_usb].index(self.usb_pick.value)]
            stem = "ueye" if d["source"].startswith("ueye") else ("pmd" if is_pmd(d["source"]) else "kamera")
            rec = CameraRecord(self.ws.free_name(stem), d["source"], d["width"], d["height"])
            if d.get("K") is not None:                  # pmd: K z kalibracji fabrycznej (xyz z SDK)
                rec.K, rec.intrinsics_from = [list(map(float, r)) for r in d["K"]], "pmd"
            rec = self.ws.add_camera(rec)
            self._found_usb = []
            self.usb_box.visible = False
            self._save()
            self.twin.cameras.sync()
            self._refresh_cameras(select=rec.name)

        @add_front.on_click
        @self._safe
        def _(event):
            n = sum(1 for c in self.ws.cameras if c.simulated)
            az = [-0.8, 0.8, -1.5, 1.5, 0.0][n % 5]
            centre = np.array([0.2, 0.0, 0.05])
            eye = centre + np.array([0.55 * np.cos(az), 0.55 * np.sin(az), 0.35])
            z = centre - eye
            z /= np.linalg.norm(z)
            x = np.cross(z, [0, 0, 1.0])
            x /= np.linalg.norm(x)
            self._add_sim_camera(self.T_b2w @ pose(np.column_stack([x, np.cross(z, x), z]), eye))

    def _viewer_pose(self, event) -> np.ndarray:
        cam = event.client.camera
        return pose(wxyz_to_mat(cam.wxyz), np.asarray(cam.position, float))

    def _add_sim_camera(self, T_world: np.ndarray) -> None:
        W, H = 640, 480
        f = (H / 2) / np.tan(np.radians(self.sim_fov.value) / 2)
        # Kamera symulowana ma "prawdziwe" K lekko nieidealne, jak kazda kamera na biurku.
        K = np.array([[f, 0, (W - 1) / 2 + 3.0], [0, f * 0.995, (H - 1) / 2 - 2.0], [0, 0, 1]])
        rec = CameraRecord(self.ws.free_name("sym"), "sim", W, H, K=K.tolist(), intrinsics_from="symulacja")
        rec.sim_pose = (inverse(self.T_b2w) @ T_world).tolist()
        self.ws.add_camera(rec)
        self._save()
        self.twin.rebuild()
        self._refresh_cameras(select=rec.name)

    def _place_sim_camera(self, rec: CameraRecord, T_world: np.ndarray, rebuild: bool) -> None:
        rec.sim_pose = (inverse(self.T_b2w) @ T_world).tolist()
        if rebuild:
            self._save()
            self.twin.rebuild()
        else:
            # Przeciaganie: tylko poza kamery w skompilowanym modelu - przebudowa co ruch myszy bylaby za ciezka.
            def move(scene):
                cid = scene.model.camera(rec.name).id
                scene.model.cam_pos[cid] = T_world[:3, 3]
                scene.model.cam_quat[cid] = mat_to_wxyz(T_world[:3, :3] @ sc.CV_TO_MJ)
                mujoco.mj_kinematics(scene.model, scene.data)
                mujoco.mj_camlight(scene.model, scene.data)
            try:
                self.twin.render_with(move)
            except KeyError:
                # Kamery nie ma w skompilowanym modelu: przebudowa RAZ, gdy przeciaganie ucichnie
                # (petla panelu). Przebudowa w kazdym zdarzeniu myszy (~250 ms, 60 zdarzen/s)
                # zapychala pule watkow visera - STOP czekal za nimi dziesiatki sekund.
                self._dirty_rebuild = time.monotonic()
            self._dirty_save = time.monotonic()
        self._update_frustum(rec)

    def _cam(self) -> CameraRecord | None:
        try:
            return self.ws.camera(self.cam_pick.value)
        except KeyError:
            return None

    def _remember_reference(self, name: str) -> None:
        img = self.twin.cameras.frame(name)
        if img is None:
            raise RuntimeError(f"{name}: brak kadru")
        mask = (self._arm_masks([name]) or {}).get(name)
        self.watch.remember(name, img, mask)

    # ------------------------------------------------------ maska ramienia
    def _joints_at(self, t: float | None) -> dict[str, float] | None:
        """Katy ramienia z chwili `t` (historia z petli panelu); None = brak historii / teraz."""
        if t is None or not self._joint_hist:
            return None
        hist = list(self._joint_hist)
        if t - hist[-1][0] > 0.3:
            return None                                     # historia urwana (rozlaczone) - poza z teraz
        before = [j for (ti, j) in hist if ti <= t]
        return before[-1] if before else hist[0][1]

    def _arm_points(self, joints: dict[str, float]) -> np.ndarray:
        """Punkty ramienia (srodki bryl, poczatki ogniw, TCP) w ukladzie podstawy, (N, 3)."""
        kin = self._kin_mask
        with self._mask_lock:
            kin._apply(kin.to_q(joints))
            d, m = kin.data, kin.model
            body = np.asarray(m.geom_bodyid) > 0                # bez bryl swiata (podloga)
            P = np.vstack([d.geom_xpos[body], d.xpos[1:], d.site_xpos])
            Ti = kin._base_inv()
        return P @ Ti[:3, :3].T + Ti[:3, 3]

    def _arm_shifts(self, rec: CameraRecord, t: float, ref: dict[str, float] | None) -> list[tuple[float, float]]:
        """Przesuniecia [px] sylwetki ramienia w kadrze `rec` w niepewnosci chwili kadru `t`.

        Z historii katow w `MASK_WINDOW` wokol `t`: dla kazdej pozy najdalej przesuniety
        punkt ramienia (rzut przez K i poze kamery) wzgledem pozy renderu `ref`. Maska
        rozciaga sie wiec TYLKO wzdluz drogi ramienia i tylko o tyle, ile ono naprawde
        przejechalo w kadrze. Wczesniej: kwadratowe poszerzenie z najwiekszego ruchu
        STAWU (lacznie z wrist_roll, ktory prawie nie rusza sylwetki) - przy ~16 st./s
        juz na limicie 40 px, maska rosla o ~22 px na kazda strone i chowala kostke
        przy szczekach we wszystkich kamerach naraz, dokladnie w chwili chwytu.
        """
        near = [(ti, j) for (ti, j) in list(self._joint_hist) if t + MASK_WINDOW[0] <= ti <= t + MASK_WINDOW[1]]
        T = rec.true_pose()
        if len(near) < 2 or T is None:
            return []
        K, _ = rec.intrinsics()
        R, p = T[:3, :3], T[:3, 3]

        def uv(P):
            c = (P - p) @ R                                     # uklad kamery (OpenCV, z do przodu)
            z = c[:, 2]
            ok = z > 0.05
            zs = np.where(ok, z, 1.0)
            return np.stack([K[0, 0] * c[:, 0] / zs + K[0, 2], K[1, 1] * c[:, 1] / zs + K[1, 2]], 1), ok

        uv0, ok0 = uv(self._arm_points(ref if ref is not None else near[-1][1]))
        path: list[np.ndarray] = []
        placed = False
        for ti, j in near:
            if not placed and ti > t:
                path.append(np.zeros(2))                        # poza renderu - na swoim miejscu w czasie
                placed = True
            uvj, okj = uv(self._arm_points(j))
            ok = ok0 & okj
            if not ok.any():
                continue
            dv = uvj[ok] - uv0[ok]
            v = dv[int(np.argmax(np.linalg.norm(dv, axis=1)))]
            n = float(np.linalg.norm(v))
            path.append(v * (MASK_MAX_SHIFT_PX / n) if n > MASK_MAX_SHIFT_PX else v)
        if not placed:
            path.append(np.zeros(2))
        shifts = [tuple(path[0])]
        for a, b in zip(path, path[1:]):
            steps = max(1, int(np.ceil(np.linalg.norm(b - a) / MASK_STEP_PX)))
            shifts += [tuple(a + (b - a) * k / steps) for k in range(1, steps + 1)]
        return shifts

    def _arm_masks(self, names: list[str], t: float | dict[str, float] | None = None,
                   dilate: int = 9) -> dict[str, np.ndarray] | None:
        """Maski ramienia (piksele zasloniete) w geometrii SUROWYCH kadrow kamer `names`.

        Render blizniaka to kamera otworkowa w pozie z chwili renderu, a kadr jest
        z dystorsja i sprzed 0-200 ms (plus opoznienie kamery). Dlatego: poza ramienia
        z chwili kadru (historia katow) - KAZDEJ kamery z chwili jej wlasnego kadru
        (`t` jako {kamera: chwila}; jedna chwila najstarszego kadru renderowala maske
        swiezszej kamery w pozie sprzed ~150 ms), maska rozciagnieta wzdluz drogi
        ramienia w niepewnosci chwili kadru (`_arm_shifts`) i przepuszczona przez
        dystorsje kamery.
        """
        names = [n for n in names if n in {c.name for c in self.ws.cameras}]
        if not names:
            return None
        times = t if isinstance(t, dict) else {n: t for n in names}
        poses = {n: self._joints_at(times.get(n)) for n in names}
        shifts = {}
        for n in names:
            tn = times.get(n)
            if tn is not None:
                shifts[n] = self._arm_shifts(self.ws.camera(n), tn, poses[n])

        def render(s):
            q0 = s.data.qpos.copy()
            out = {}
            for n in names:
                if poses[n]:
                    s.set_joints(poses[n])                  # kopia stanu do renderu - scena bez zmian
                elif not np.array_equal(s.data.qpos, q0):
                    s.data.qpos[:] = q0                     # bez historii: poza z teraz, nie poprzedniej kamery
                    mujoco.mj_kinematics(s.model, s.data)
                out[n] = arm_mask(s, n, dilate=dilate)
            return out
        try:
            masks = self.twin.render_with(render)
        except KeyError:                                    # kamery nie ma w modelu (wylaczona, bez pozy)
            return None
        out = {}
        for n, m in masks.items():
            m = smear_mask(m, shifts.get(n, []))
            rec = self.ws.camera(n)
            if not rec.simulated:
                K, dist = rec.intrinsics()
                m = distort_mask(m, K, dist)
            out[n] = m
        return out

    def _has_frame(self, c: CameraRecord) -> bool:
        """Kadr byl w ostatnich GUIDE_FRAME_LOST s (znacznik `_frame_seen` z przewodnika; nowa kamera - tak)."""
        seen = self._frame_seen.get(c.name)
        return seen is None or time.monotonic() - seen < GUIDE_FRAME_LOST

    def _cam_badge(self, c: CameraRecord) -> tuple[str, str]:
        """(slowo, klasa) stanu kamery w zakladce Kamery - kolory jak piramidy w 3D."""
        if not c.enabled:
            return "WYLACZONA", "tw-none"
        if not self._has_frame(c):
            return "BEZ OBRAZU", "tw-bad"
        if self.watch.moved(c.name):
            return "PORUSZONA", "tw-bad"
        if not c.calibrated:
            return "DO KALIBRACJI", "tw-sim" if c.simulated else "tw-none"
        if c.trusted:
            return "SKALIBROWANA", "tw-ok"
        return "KALIBRACJA NIEPEWNA", "tw-warn"

    @staticmethod
    def _k_badge(c: CameraRecord) -> tuple[str, str]:
        if c.simulated or not c.intrinsics_problem():
            return "OBIEKTYW OK", "tw-ok"
        if c.intrinsics_from != "szachownica" or c.K is None:
            return "OBIEKTYW: KROK 1", "tw-none"
        return "OBIEKTYW: POWTORZ", "tw-warn"

    def _pose_badge(self, c: CameraRecord) -> tuple[str, str]:
        if not c.enabled:
            return "WYLACZONA", "tw-none"
        if c.calibrated and self.watch.moved(c.name):
            return "PORUSZONA", "tw-bad"
        if not c.calibrated:
            return "POLOZENIE: KROK 2", "tw-none"
        if c.trusted and not c.intrinsics_problem():
            return "POLOZENIE OK", "tw-ok"
        return "POLOZENIE: POWTORZ", "tw-warn"

    def _cams_html(self, calib: bool) -> str:
        """Lista kamer: nazwa, zrodlo i plakietki (Kamery: stan; Kalibracja: K i poza)."""
        if not self.ws.cameras:
            text = ("Brak kamer - dodaj je w zakladce Kamery." if calib else
                    "Brak kamer. Dodaj symulowana albo USB ponizej. Gotowe stanowisko z dwiema skalibrowanymi "
                    "kamerami w symulacji: zamknij panel, w konsoli lerobot-twin demo, uruchom panel ponownie.")
            return f'<div class="tw-empty">{_html.escape(text)}</div>'
        rows = []
        for c in self.ws.cameras:
            src = "symulowana" if c.simulated else (f"glebia {c.source}" if is_pmd(c.source) else f"USB {c.source}")
            badges = [self._k_badge(c), self._pose_badge(c)] if calib else [self._cam_badge(c)]
            b = "".join(f'<span class="tw-badge {cls}">{_html.escape(w)}</span>' for w, cls in badges)
            rows.append(f'<div class="tw-cam"><b>{_html.escape(c.name)}</b><span class="tw-src">{_html.escape(src)}'
                        f'</span><span class="tw-badges">{b}</span></div>')
        return '<div class="tw-cams">' + "".join(rows) + "</div>"

    def _render_cameras(self) -> None:
        """Lista i przycisk Usun (pytanie wygasa samo) - co takt petli, zapisy tylko przy zmianie."""
        set_if(self.cams_html, "content", self._cams_html(calib=False))
        has = bool(self.ws.cameras)
        set_if(self.cam_box, "visible", has)
        rec = self._cam()
        if rec is not None and self.cam_ask.pending(("usun", rec.name)):
            set_if(self.cam_remove, "label", f"Na pewno usunac {rec.name}? Kliknij ponownie")
            set_if(self.cam_remove, "icon", viser.Icon.ALERT_TRIANGLE)
        else:
            set_if(self.cam_remove, "label", "Usun kamere")
            set_if(self.cam_remove, "icon", viser.Icon.TRASH)

    def _refresh_cameras(self, select: str | None = None) -> None:
        names = tuple(c.name for c in self.ws.cameras) or ("-",)
        self.cam_pick.options = names
        if select is not None:
            self.cam_pick.value = select
        elif self.cam_pick.value not in names:
            self.cam_pick.value = names[0]
        with self.scene_lock:
            for n in list(self.frustums):
                if n not in {c.name for c in self.ws.cameras}:
                    self.frustums.pop(n).remove()
                    self._frustum_key.pop(n, None)
                    if n in self.estimates:
                        self.estimates.pop(n).remove()
        for c in self.ws.cameras:
            self._update_frustum(c)
        self._select_camera()
        self._render_cameras()
        self._mapper_key = None

    def _update_frustum(self, c: CameraRecord) -> None:
        with self.scene_lock:
            self._update_frustum_locked(c)

    def _update_frustum_locked(self, c: CameraRecord) -> None:
        T = c.true_pose()
        if T is None or not c.enabled:
            if c.name in self.frustums:
                self.frustums[c.name].visible = False
            return
        Tw = self.T_b2w @ T
        K, _ = c.intrinsics()
        color = GREEN if c.trusted else (ORANGE if c.calibrated else (BLUE if c.simulated else GREY))
        if self.watch.moved(c.name):
            color = RED
        h = self.frustums.get(c.name)
        key = (color, round(fov_of(K, c.height), 4))
        if h is not None and self._frustum_key.get(c.name) != key:
            h.remove()                                    # kolor i pole widzenia ustawia sie przy tworzeniu
            h = None
        if h is None:
            h = self.server.scene.add_camera_frustum(f"/kamery/{c.name}", fov=fov_of(K, c.height),
                                                     aspect=c.width / c.height, scale=0.06, color=color,
                                                     thickness=0.002, wxyz=mat_to_wxyz(Tw[:3, :3]),
                                                     position=Tw[:3, 3])
            self.frustums[c.name] = h
            self._frustum_key[c.name] = key
        else:
            h.position, h.wxyz, h.visible = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3]), True
        # Symulowana: gdzie ja widzi kalibracja (szara piramida) obok prawdy.
        if c.simulated and c.calibrated:
            Te = self.T_b2w @ np.asarray(c.T_cam2base, float)
            e = self.estimates.get(c.name)
            if e is None:
                self.estimates[c.name] = self.server.scene.add_camera_frustum(
                    f"/kalibracja/{c.name}", fov=fov_of(K, c.height), aspect=c.width / c.height, scale=0.06,
                    color=(200, 200, 200), thickness=0.001, wxyz=mat_to_wxyz(Te[:3, :3]), position=Te[:3, 3])
            else:
                e.position, e.wxyz = Te[:3, 3], mat_to_wxyz(Te[:3, :3])
        elif c.name in self.estimates:
            self.estimates.pop(c.name).remove()

    def _select_camera(self) -> None:
        rec = self._cam()
        if rec is None:
            self.cam_info.content = ""
            self.cam_detail.content = ""
            self.cam_gizmo.visible = False
            return
        self.cam_enabled.value = rec.enabled
        if not rec.simulated and self.cam_move.value:
            self.cam_move.value = False                 # prawdziwej nie przestawia sie z panelu
        set_if(self.cam_sim_box, "visible", rec.simulated)
        # Jedna linia: co z ta kamera i co dalej; liczby w Szczegoly.
        if not rec.enabled:
            info = "wylaczona - nie ma jej w scenie, mapie ani kalibracji"
        elif self.watch.moved(rec.name):
            info = (f"**poruszona** od kalibracji (obraz przesunal sie o "
                    f"{_px(self.watch.shift.get(rec.name, float('inf')))}) - zakladka **Kalibracja**, "
                    f"pomaranczowy przycisk")
        elif not rec.calibrated and is_pmd(rec.source):
            info = ("poza z tagow na obudowie: folder **Kamera PMD** ponizej (os optyczna, Znajdz PMD); "
                    "K juz jest - z kalibracji fabrycznej")
        elif not rec.calibrated:
            info = "jeszcze nieskalibrowana - dalej zakladka **Kalibracja**"
        elif rec.trusted:
            info = "skalibrowana - gotowa do uzycia"
        else:
            why = (rec.calibration.get("reason", "") if not rec.calibration.get("trusted")
                   else "obiektyw zmierzony po ustaleniu polozenia - zrob krok 2 jeszcze raz")
            info = f"kalibracja niepewna: {why}"
        if rec.simulated and rec.calibrated and rec.sim_pose is not None:
            from ..calib.handeye import pose_error
            dt, dr = pose_error(np.asarray(rec.sim_pose), np.asarray(rec.T_cam2base))
            info += f"  \nod prawdy symulacji: **{dt * 1000:.2f} mm, {np.degrees(dr):.3f} st.**"
        err = self.twin.cameras.error(rec.name)
        if err:
            info += f"  \n**blad kamery:** {err}"
        self.cam_info.content = info
        K, dist = rec.intrinsics()
        lines = [f"zrodlo `{rec.source}`, {rec.width}x{rec.height}",
                 f"K ({rec.intrinsics_from}): fx {K[0, 0]:.0f} fy {K[1, 1]:.0f} cx {K[0, 2]:.0f} cy {K[1, 2]:.0f}"]
        if rec.intrinsics_info:
            ii = rec.intrinsics_info
            if str(ii.get("reason", "")).startswith("z pliku "):
                rms = ii.get("rms_px", 0)
                lines.append(f"intrynsyki {ii['reason']}, {ii.get('n_views', 0)} kadrow"
                             + (f", residuum {rms:.2f} px" if rms else ""))
            else:
                lines.append(f"intrynsyki: residuum {ii.get('rms_px', 0):.2f} px, {ii.get('n_views', 0)} kadrow, "
                             f"pokrycie {ii.get('coverage', 0):.0%}"
                             f"{'' if ii.get('trusted') else ' - ' + ii.get('reason', '')}")
        sh = self.watch.shift.get(rec.name)
        if sh is not None:
            lines.append(f"przesuniecie kadru od odniesienia: {_px(sh)}")
        self.cam_detail.content = "  \n".join(lines)
        T = rec.true_pose()
        # Wylaczonej kamery nie ma w scenie - uchwyt przy niej przebudowywal scene co ruch myszy.
        show = rec.simulated and rec.enabled and self.cam_move.value and T is not None
        if show:
            Tw = self.T_b2w @ T
            self.cam_gizmo.position, self.cam_gizmo.wxyz = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3])
        self.cam_gizmo.visible = show
        self._render_cameras()

    # ============================================================= kalibracja
    def _build_calibration(self) -> None:
        """Zakladka Kalibracja dla kogos, kto nie zna slow "intrynsyki" i "fala": po co to jest,
        kamery z plakietkami OBIEKTYW i POLOZENIE, linia "Teraz", dwa kroki z ponumerowana instrukcja.
        Widac tylko to, co w danej chwili dziala; stan przemalowuje `_render_calibration` co takt."""
        g = self.server.gui
        g.add_markdown("Kalibracja uczy blizniaka dwoch rzeczy o kazdej kamerze: **jak jej obiektyw wygina obraz** "
                       "(krok 1) i **gdzie kamera stoi wzgledem ramienia** (krok 2). Bez tego mapa stolu i szukanie "
                       "kostki nie dzialaja. Robisz to raz - powtarzasz tylko po poruszeniu kamery.")
        self.calib_html = g.add_html("")
        self.calib_now_md = g.add_markdown("")
        # --- krok 1: obiektyw (intrynsyki - K i dystorsja)
        self.intr_box = g.add_folder(None)
        with self.intr_box:
            g.add_divider()
            self.intr_title = g.add_markdown("")
            self.intr_help = g.add_markdown(
                "Kazdy obiektyw troche wygina obraz. Program zmierzy to na wydrukowanej szachownicy. "
                "Raz dla kazdej kamery; ramie stoi.\n\n"
                "1. Pobierz szachownice i wydrukuj ja w skali **100%** (bez \"dopasuj do strony\").\n"
                "2. Zmierz linijka bok jednego kwadratu na wydruku i wpisz go nizej.\n"
                "3. Kliknij **Zacznij** (przy kilku kamerach najpierw wybierz kamere).\n"
                "4. Pokazuj szachownice kamerze, az licznik dojdzie do 12, potem **Gotowe**.")
            self.intr_sheet_box = g.add_folder(None)
            with self.intr_sheet_box:
                sheet = g.add_button("Pobierz szachownice do druku (A4)", icon=viser.Icon.DOWNLOAD, color="gray",
                                     hint="Plik PNG z szachownica ChArUco. Drukuj w skali 100%.")
                self.board_mm = g.add_number("Bok kwadratu na wydruku [mm]", 28.0, min=5.0, max=100.0, step=0.1,
                                             hint="Zmierzony linijka bok jednego kwadratu szachownicy (nominalnie 28 "
                                                  "mm - drukarka potrafi zmienic skale).")
            self.intr_cam = g.add_dropdown("Kamera", ("-",), initial_value="-",
                                           hint="Kamera, ktorej obiektyw teraz mierzysz. Na starcie: pierwsza, "
                                                "ktora tego jeszcze nie ma.")
            self.intr_start_box, self.intr_start = self._big_button(
                "Zacznij", icon=viser.Icon.PLAYER_RECORD, color="green",
                hint="Wlacza robienie zdjec szachownicy z wybranej kamery. Zdjecie robi sie samo, gdy szachownica "
                     "jest w nowym miejscu. Ramieniem nie rusza.")
            self.lens_file_box = g.add_folder(None, visible=False)
            with self.lens_file_box:
                self.lens_file = g.add_dropdown("Gotowy obiektyw z pliku", ("-",), initial_value="-",
                                                hint=f"Obiektyw zmierzony gdzie indziej: pliki .txt w {LENS_DIR} "
                                                     "(camera_matrix i dist_coeffs, opcjonalnie image_size).")
                lens_load = g.add_button("Wczytaj obiektyw z pliku", icon=viser.Icon.FILE_IMPORT, color="gray",
                                         hint="Zamiast szachownicy: wpisuje K i dystorsje z pliku do wybranej "
                                              "kamery. Plik musi byc z tej samej rozdzielczosci.")
            self.intr_img = g.add_image(np.zeros((240, 320, 3), np.uint8), format="jpeg", jpeg_quality=70,
                                        visible=False)
            self.intr_md = g.add_markdown("")
            self.intr_run_box = g.add_folder(None, visible=False)
            with self.intr_run_box:
                self.intr_solve = g.add_button("Gotowe - policz obiektyw", icon=viser.Icon.CALCULATOR,
                                               color="green",
                                               hint="Konczy zdjecia i liczy obiektyw kamery (zapisuje sie sam).")
                icancel = g.add_button("Anuluj (nic nie zapisze)", icon=viser.Icon.X, color="gray",
                                       hint="Konczy zdjecia bez liczenia - kamera zostaje, jaka byla.")
        # --- krok 2: polozenie (fala z karta)
        self.pose_box = g.add_folder(None)
        with self.pose_box:
            g.add_divider()
            self.pose_title = g.add_markdown("")
            self.pose_help = g.add_markdown("")
            self.pose_sheet_box = g.add_folder(None)
            with self.pose_sheet_box:
                card = g.add_button("Pobierz karte do druku (A4)", icon=viser.Icon.DOWNLOAD, color="gray",
                                    hint="Plik PNG z karta z dwoma kodami (300 dpi). Drukuj w skali 100%.")
                self.tag_mm = g.add_number("Bok kodu na wydruku [mm]", float(self.ws.card_obj().tag_size * 1000),
                                           min=10.0, max=120.0, step=0.1,
                                           hint="Zmierzony linijka bok kwadratowego kodu (czarna ramka) na karcie. "
                                                "Zmieniony po pomiarze blokuje zapis wyniku.")
            self.wave_box, self.wave_btn = self._big_button(
                "Start", icon=viser.Icon.WAVE_SINE, color="green",
                hint="RUSZA ramieniem: ok. 20 miejsc z karta nad stolem, potem ramie wraca na miejsce. Prawdziwe "
                     "ramie: drugie klikniecie potwierdza, ze karta jest w chwytaku, a stol pusty.")
            self.reloc_box, self.reloc_btn = self._big_button(
                "Kamera poruszona", icon=viser.Icon.CURRENT_LOCATION, color="orange",
                hint="Krotki pomiar tylko dla poruszonej kamery - RUSZA ramieniem; karta w chwytaku. Prawdziwe "
                     "ramie: drugie klikniecie potwierdza.")
            self.wave_md = g.add_markdown("")
            self.truth_btn = g.add_button("Symulacja: pomin - wpisz prawdziwe polozenie kamer", icon=viser.Icon.CHECK,
                                          color="gray",
                                          hint="Skrot tylko w symulacji: program zna prawdziwe polozenie kamer "
                                               "symulowanych. Do proby calej procedury - Start.")
            self.calib_bar = g.add_progress_bar(0.0, animated=True, visible=False)
            self.calib_md = g.add_markdown("")
            self.calib_cancel = g.add_button("Zatrzymaj (ramie wroci na miejsce, nic nie zapisze)",
                                             icon=viser.Icon.PLAYER_STOP, color="gray", visible=False)
            self.calib_apply_box, self.calib_apply = self._big_button(
                "Zapisz wynik", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                hint="Zapisuje polozenie kamer. Kamery z wynikiem 'zle' zapisuja sie jako niepewne, z powodem.")
            with self.calib_apply_box:
                discard = g.add_button("Odrzuc (nic nie zapisze)", icon=viser.Icon.X, color="gray",
                                       hint="Wynik znika, kamery zostaja, jakie byly.")
            self.preset_box = g.add_folder(None, visible=False)
            with self.preset_box:
                self.preset_pick = g.add_dropdown(
                    "Zapisane polozenia kamer", ("-",), initial_value="-",
                    hint="Kazdy zapis kroku 2 to osobny plik w workspace/calib/ - nic nie nadpisuje "
                         "poprzedniego. Domyslnie wgrany jest najnowszy; wybierz starszy, zeby do niego wrocic.")
                self.preset_md = g.add_markdown("")
        self.calib_ask = _TwoClick()
        self._calib_pending = False

        @sheet.on_click
        @self._safe
        def _(event):
            img = Board(square=self.board_mm.value / 1000, marker=0.75 * self.board_mm.value / 1000).image()
            ok, buf = cv2.imencode(".png", img)
            event.client.send_file_download("szachownica.png", buf.tobytes())

        @card.on_click
        @self._safe
        def _(event):
            sheet_img = Card(tag_size=self.tag_mm.value / 1000).sheet(dpi=300)
            ok, buf = cv2.imencode(".png", sheet_img[..., ::-1])
            event.client.send_file_download("karta_kalibracyjna.png", buf.tobytes())

        @self.intr_start.on_click
        @self._safe
        def _(event):
            name = self.intr_cam.value
            try:
                rec = self.ws.camera(name)
            except KeyError:
                raise RuntimeError("wybierz kamere") from None
            if rec.simulated:
                raise RuntimeError("kamera symulowana nie potrzebuje kroku 1")
            if self.calib_job.running:
                raise RuntimeError("ramie pokazuje karte - poczekaj na koniec albo Zatrzymaj")
            board = Board(square=self.board_mm.value / 1000, marker=0.75 * self.board_mm.value / 1000)
            solve_evt = threading.Event()
            self.intr_job.start(lambda job: jobs.run_intrinsics(job, self.twin.cameras, name, board,
                                                                 (rec.width, rec.height), solve=solve_evt))
            self._intr_solve = solve_evt
            self.intr_md.content = ""
            self.calib_ask.reset()
            self._render_calibration()

        @lens_load.on_click
        @self._safe
        def _(event):
            if self.intr_job.running:
                raise RuntimeError("trwa mierzenie szachownica - Gotowe albo Anuluj")
            name = self._import_lens(self.intr_cam.value, self.lens_file.value)
            self._notify(event, "Obiektyw wczytany", f"{name}: {self.ws.camera(name).intrinsics_info['reason']}")

        @self.intr_solve.on_click
        @self._safe
        def _(event):
            if not self.intr_job.running:
                raise RuntimeError("najpierw Zacznij")
            self._intr_solve.set()                      # "Anuluj" to osobny sygnal - bez liczenia

        @icancel.on_click
        def _(event):
            self.intr_job.stop()                        # przerwanie: K zostaje, jakie bylo

        @self.wave_btn.on_click
        @self._safe
        def _(event):
            cams = self._wave_cameras()
            if self._confirm_motion(("fala", tuple(cams))):
                self._start_card_calibration(event, cams, quick=False)

        @self.reloc_btn.on_click
        @self._safe
        def _(event):
            moved = self._moved_cameras()
            if not moved:
                raise RuntimeError("zadna kamera nie jest poruszona")
            if self._confirm_motion(("reloc", moved[0])):
                self._start_card_calibration(event, [moved[0]], quick=True)

        @self.truth_btn.on_click
        @self._safe
        def _(event):
            names = [c.name for c in self._truth_cameras()]
            if not names:
                raise RuntimeError("brak kamer symulowanych do wpisania")
            for n in names:
                rec = self.ws.camera(n)
                rec.T_cam2base = [list(r) for r in rec.sim_pose]
                rec.calibration = {"trusted": True, "reason": "", "rms_px": 0.0, "source": "prawda symulacji",
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._save()
            self._refresh_cameras()
            for n in names:
                try:
                    self._remember_reference(n)
                except RuntimeError:
                    pass
            self.calib_md.content = f"Wpisano prawdziwe polozenie: {', '.join(names)}."
            self._notify(event, "Gotowe", f"Polozenie z symulacji: {', '.join(names)}")

        @self.calib_cancel.on_click
        def _(event):
            self.calib_job.stop()

        @self.calib_apply.on_click
        @self._safe
        def _(event):
            names = self._apply_calibration()
            bad = [f"{n}: {self.ws.camera(n).calibration.get('reason', '')}" for n in names
                   if not self.ws.camera(n).trusted]
            self._notify(event, "Zapisano", f"Polozenie kamer: {', '.join(names)}"
                         + (f". NIEPEWNE: {'; '.join(bad)}" if bad else ""), error=bool(bad))

        @discard.on_click
        def _(event):
            self._calib_pending = False
            self.calib_md.content = "Wynik odrzucony - kamery bez zmian."
            self._render_calibration()

        @self.preset_pick.on_update
        @self._safe
        def _(event):
            name = self.preset_pick.value
            if name == "-" or name == self.ws.calib_preset:
                return
            names = self.ws.load_preset(name)
            self._poses_changed(names)
            self._notify(event, "Polozenie kamer", f"Wgrano {name}: {', '.join(names) or 'zadna kamera nie pasuje'}.")
            self._render_calibration()

    # ---------------------------------------------------- kalibracja: stan
    def _wave_cameras(self) -> list[str]:
        """Kamery, ktore wezma udzial w kroku 2: wlaczone i z kadrem w ostatnim odczycie petli (`_grab`).
        Nie `cameras.frame` - ten renderuje kazda symulowana od nowa, a to jest wolane co takt."""
        with self.frame_lock:
            live = set(self.frames)
        return [c.name for c in self.ws.cameras if c.enabled and c.name in live]

    def _moved_cameras(self) -> list[str]:
        return [c.name for c in self.ws.cameras if c.enabled and c.calibrated and self.watch.moved(c.name)]

    def _truth_cameras(self) -> list[CameraRecord]:
        """Symulowane, ktorym skrot "pomin" cos da: nieskalibrowane, niepewne albo poruszone."""
        return [c for c in self.ws.cameras if c.enabled and c.simulated and c.sim_pose is not None
                and (not c.trusted or self.watch.moved(c.name))]

    def _set_lens(self, rec: CameraRecord, res: intr.Result) -> None:
        """Zapis obiektywu (K, dystorsja) kamery - z szachownicy albo z pliku. Poza ustalona ze starym K
        przestaje byc zaufana: krok 2 trzeba powtorzyc."""
        rec.K, rec.dist = res.K.tolist(), res.dist.tolist()
        rec.intrinsics_from = "szachownica"
        self._simreal_px.pop(rec.name, None)            # nowe K - Sim-Real do sprawdzenia od nowa
        rec.intrinsics_info = {"rms_px": res.rms_px, "n_views": res.n_views, "coverage": res.coverage,
                               "trusted": res.trusted, "reason": res.reason,
                               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if rec.calibrated:
            rec.calibration["reason"] = "obiektyw zmierzony po ustaleniu polozenia - zrob krok 2 jeszcze raz"
            rec.calibration["trusted"] = False
        self._save()
        self.twin.rebuild()
        self._refresh_cameras()

    def _lens_files(self) -> list[str]:
        """Nazwy plikow z gotowym obiektywem w `LENS_DIR` (posortowane)."""
        return sorted(f.name for f in LENS_DIR.glob("*.txt")) if LENS_DIR.is_dir() else []

    def _import_lens(self, name: str, filename: str) -> str:
        """Obiektyw kamery `name` z pliku `LENS_DIR/filename`. Odmawia, gdy plik jest z innej
        rozdzielczosci (K zalezy od niej) - albo, bez `image_size` w pliku, gdy punkt glowny
        lezy daleko od srodka kadru kamery."""
        try:
            rec = self.ws.camera(name)
        except KeyError:
            raise RuntimeError("wybierz kamere") from None
        if rec.simulated:
            raise RuntimeError("kamera symulowana nie potrzebuje kroku 1")
        if filename == "-" or not filename:
            raise RuntimeError(f"brak pliku z obiektywem - wgraj .txt do {LENS_DIR}")
        path = LENS_DIR / filename
        if not path.is_file():
            raise RuntimeError(f"nie ma pliku {path}")
        try:
            res = intr.read_file(path)
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        W, H = rec.width, rec.height
        if res.size != (0, 0) and res.size != (W, H):
            raise RuntimeError(f"{filename} jest z rozdzielczosci {res.size[0]}x{res.size[1]}, "
                               f"a kamera {name} ma {W}x{H}")
        cx, cy = res.K[0, 2], res.K[1, 2]
        if res.size == (0, 0) and (abs(2 * cx - W) > 0.3 * W or abs(2 * cy - H) > 0.3 * H):
            raise RuntimeError(f"{filename}: punkt glowny ({cx:.0f}, {cy:.0f}) nie pasuje do kadru {W}x{H} "
                               f"kamery {name} - plik z innej rozdzielczosci?")
        self._set_lens(rec, res)
        self.intr_md.content = f"Obiektyw **{name}** wczytany {res.reason}."
        todo = self._lens_todo()
        if todo and self.intr_cam.value not in todo:
            self.intr_cam.value = todo[0]
        return name

    def _lens_todo(self) -> list[str]:
        """Prawdziwe wlaczone kamery bez zmierzonego obiektywu (krok 1)."""
        return [c.name for c in self.ws.cameras if c.enabled and not c.simulated and c.intrinsics_problem()]

    def _wave_block(self) -> str:
        """Dlaczego krok 2 nie moze teraz ruszyc ("" = moze) - przycisk wylaczony z tym powodem."""
        if self.intr_job.running:
            return "Najpierw skoncz krok 1 (Gotowe albo Anuluj)."
        todo = self._lens_todo()
        if todo:
            return f"Najpierw krok 1 dla: {', '.join(todo)}."
        if not self.twin.connected:
            return "Najpierw polacz ramie: zakladka Ramie, przycisk Polacz."
        if self._estop_active():
            return "Aktywny STOP - najpierw zolty przycisk Skasuj STOP na gorze."
        busy = self._busy(jobs.CALIB_OWNER)
        if busy:
            return f"Ramie jest zajete ({busy}) - poczekaj albo je zatrzymaj."
        if not self._wave_cameras():
            return "Zadna kamera nie daje obrazu - sprawdz zakladke Kamery."
        return ""

    def _confirm_motion(self, key) -> bool:
        """Sim: od razu. Prawdziwe ramie: pierwszy klik pyta (przycisk zmienia sie w pytanie), drugi rusza."""
        why = self._wave_block()
        if why:
            raise RuntimeError(why)
        if self.twin.status.simulated or self.calib_ask.ask(key):
            return True
        self._render_calibration()
        return False

    def _render_calibration(self) -> None:
        """Przemalowanie zakladki Kalibracja do stanu (co takt; zapisy tylko przy zmianie)."""
        with self._render_lock:
            set_if(self.calib_html, "content", self._cams_html(calib=True))
            set_if(self.calib_now_md, "content", f"**Teraz:** {self._calib_now()}")
            ij, cj = self.intr_job, self.calib_job
            sim = bool(self.twin.status.simulated) if self.twin.connected else self._preset_backend() == "sim"
            # --- krok 1: tylko przy prawdziwej kamerze
            real = [c for c in self.ws.cameras if c.enabled and not c.simulated]
            todo = self._lens_todo()
            set_if(self.intr_box, "visible", bool(real) or ij.running)
            opts = tuple(c.name for c in real) or ("-",)
            if tuple(self.intr_cam.options) != opts:
                cur = self.intr_cam.value
                self.intr_cam.options = opts
                self.intr_cam.value = cur if cur in opts else (todo[0] if todo else opts[0])
            done = bool(real) and not todo
            set_if(self.intr_title, "content", "**Krok 1 z 2: zmierz obiektyw kazdej kamery**" + (
                " - gotowe" if done else f" - zostalo: {', '.join(todo)}"))
            set_if(self.intr_help, "visible", not done and not ij.running)
            set_if(self.intr_sheet_box, "visible", not done and not ij.running)
            set_if(self.intr_cam, "visible", not ij.running and len(real) > 1)
            set_if(self.intr_start_box, "visible", not ij.running)
            files = tuple(self._lens_files())
            set_if(self.lens_file_box, "visible", bool(files) and not ij.running)
            if files and tuple(self.lens_file.options) != files:
                cur = self.lens_file.value
                self.lens_file.options = files
                self.lens_file.value = cur if cur in files else files[0]
            cam = self.intr_cam.value
            label = (f"Zmierz jeszcze raz: {cam}" if cam not in todo else
                     f"Zacznij: pokazuj szachownice kamerze {cam}")
            set_if(self.intr_start, "label", label)
            set_if(self.intr_start, "color", "green" if cam in todo else "gray")
            set_if(self.intr_start, "disabled", cj.running)
            set_if(self.intr_run_box, "visible", ij.running)
            set_if(self.intr_img, "visible", ij.running)
            col = ij.data.get("collector") if ij.running else None
            enough = col is not None and len(col.views) >= 12 and col.coverage >= 0.55
            set_if(self.intr_solve, "label", f"Gotowe - policz obiektyw {ij.data.get('camera', '')}" if enough else
                   "Gotowe - policz obiektyw (jeszcze za malo zdjec)")
            set_if(self.intr_solve, "color", "green" if enough else "gray")
            # --- krok 2
            cams = self._wave_cameras()
            moved = self._moved_cameras()
            enabled = [c for c in self.ws.cameras if c.enabled]
            poses_done = bool(enabled) and not moved and all(c.trusted and not c.intrinsics_problem()
                                                             for c in enabled)
            idle = not cj.running and not self._calib_pending
            set_if(self.pose_title, "content", "**Krok 2 z 2: ustal, gdzie stoja kamery**" + (
                " - gotowe" if poses_done else ""))
            if sim:
                help_ = ("Ramie samo pokaze kamerom karte z kodem w ok. 20 miejscach nad stolem (ok. 2-3 min); "
                         "z tego program liczy, gdzie stoi kazda kamera. W symulacji karta jest w chwytaku sama - "
                         "kliknij **Start**. Szybciej: **Symulacja: pomin** nizej.")
            else:
                help_ = ("Ramie samo pokaze kamerom karte z kodem w ok. 20 miejscach nad stolem (ok. 2-3 min); "
                         "z tego program liczy, gdzie stoi kazda kamera. Wszystkie kamery naraz.\n\n"
                         "1. Pobierz karte i wydrukuj ja w skali **100%**.\n"
                         "2. Zmierz linijka bok kwadratowego kodu (czarna ramka) i wpisz go nizej.\n"
                         "3. Zegnij karte na linii; wolny koniec wloz w szczeki chwytaka - ok. **7 cm** ma wystawac.\n"
                         "4. Zabierz wszystko ze stolu, odsun sie od ramienia i kliknij **Start** (dwa razy - "
                         "drugie klikniecie potwierdza).")
            set_if(self.pose_help, "content", help_)
            set_if(self.pose_help, "visible", idle and not poses_done and not moved)
            set_if(self.pose_sheet_box, "visible", idle and not poses_done and not moved and not sim)
            why = self._wave_block()
            set_if(self.wave_box, "visible", idle)
            names = ", ".join(cams) or "brak kamer z obrazem"
            if why:
                look = ("Start: ramie pokaze karte kamerom", "gray", viser.Icon.WAVE_SINE)
            elif self.calib_ask.pending(("fala", tuple(cams))):
                look = ("Karta w chwytaku, stol pusty, stoisz z daleka? Kliknij jeszcze raz - ramie ruszy",
                        "orange", viser.Icon.ALERT_TRIANGLE)
            elif poses_done or moved:
                look = (f"Zrob krok 2 jeszcze raz ({names})", "gray", viser.Icon.WAVE_SINE)
            else:
                look = (f"Start: ramie pokaze karte kamerom ({names})", "green", viser.Icon.WAVE_SINE)
            _Toggle._show(self.wave_btn, *look, disabled=bool(why))
            set_if(self.reloc_box, "visible", idle and bool(moved))
            if moved:
                if self.calib_ask.pending(("reloc", moved[0])):
                    look = (f"Karta w chwytaku, stol pusty? Kliknij jeszcze raz - ramie pokaze karte {moved[0]}",
                            "orange", viser.Icon.ALERT_TRIANGLE)
                else:
                    look = (f"Kamera {moved[0]} zostala poruszona - pokaz jej karte jeszcze raz (ok. 1 min)",
                            "orange", viser.Icon.CURRENT_LOCATION)
                _Toggle._show(self.reloc_btn, *look, disabled=bool(why))
            notes = []
            if idle and why:
                notes.append(why)
            elif idle and moved:
                notes.append(f"Kamera {moved[0]} przesunela sie od kalibracji. " +
                             ("" if sim else "Wloz karte w chwytak i ") + "kliknij pomaranczowy przycisk - "
                             "ramie pokaze karte tylko tej kamerze.")
            set_if(self.wave_md, "content", "  \n".join(notes))
            set_if(self.wave_md, "visible", bool(notes))
            set_if(self.truth_btn, "visible", idle and sim and bool(self._truth_cameras()))
            set_if(self.calib_bar, "visible", cj.running)
            set_if(self.calib_cancel, "visible", cj.running)
            set_if(self.calib_apply_box, "visible", self._calib_pending)
            used = cj.data.get("tag_size")
            changed = used is not None and abs(self.tag_mm.value / 1000 - used) > 1e-6
            if changed:
                look = (f"Nie da sie zapisac: bok kodu zmieniony po pomiarze ({used * 1000:.1f} -> "
                        f"{self.tag_mm.value:.1f} mm) - zrob krok 2 jeszcze raz", "gray", viser.Icon.DEVICE_FLOPPY)
            else:
                look = ("Zapisz wynik", "green", viser.Icon.DEVICE_FLOPPY)
            _Toggle._show(self.calib_apply, *look, disabled=changed)
            presets = tuple(self.ws.presets())
            set_if(self.preset_pick, "options", presets or ("-",))
            current = self.ws.calib_preset
            if current in presets:
                # Tylko wgrany zestaw: zapis z kodu wola on_update, a ten wgrany pomija - bez petli.
                set_if(self.preset_pick, "value", current)
            set_if(self.preset_md, "content", "" if not presets or current == presets[0]
                   else f"Wgrany starszy zestaw; najnowszy to {presets[0]}.")
            set_if(self.preset_box, "visible", idle and bool(presets))

    def _calib_now(self) -> str:
        """Jedna nastepna akcja w tej zakladce - z krokow 1-3 przewodnika, zeby nie mowic czego innego niz naglowek."""
        cj, ij = self.calib_job, self.intr_job
        if cj.running:
            return f"ramie pokazuje karte kamerom ({int(100 * cj.progress)}%) - nie wchodz w jego zasieg."
        if self._calib_pending:
            return "sprawdz wynik ponizej i kliknij **Zapisz wynik** (albo Odrzuc)."
        if ij.running:
            return (f"pokazuj szachownice kamerze {ij.data.get('camera', '')} (opis pod obrazem), potem "
                    f"**Gotowe - policz obiektyw**.")
        g = self._guide_last
        if g is not None:
            for st in g.steps:
                if st.number in (1, 2, 3) and st.status not in (gd.DONE, gd.SKIP):
                    why = self._wave_block() if st.number == 3 else ""
                    if why:                             # przycisk kroku 2 zablokowany - najpierw powod
                        return why[0].lower() + why[1:].rstrip(".") + "."
                    text = st.action or st.detail
                    for prefix in ("Kalibracja > ", "Kalibracja: ", "Kalibracja, "):
                        text = text.removeprefix(prefix)
                    return f"{text[0].lower() + text[1:]}."
        return "wszystkie kamery skalibrowane. Gdy ktoras poruszysz, pojawi sie tu przycisk dla niej."

    def _apply_calibration(self) -> list[str]:
        """Zapis wyniku kroku 2 - z bokiem kodu i K, z ktorymi go liczono."""
        fit = self.calib_job.result
        if fit is None:
            raise RuntimeError("brak wyniku")
        used = self.calib_job.data.get("tag_size")
        if used is None:
            raise RuntimeError("wynik bez boku taga - zrob krok 2 jeszcze raz")
        if abs(self.tag_mm.value / 1000 - used) > 1e-6:
            # Poza policzona z innym bokiem taga jest przeskalowana (50,0 -> 49,2 mm: 1,6%
            # translacji) - zapis z nowym bokiem ukrylby te niezgodnosc na zawsze.
            raise RuntimeError(f"bok taga zmieniony od pomiaru ({used * 1000:.1f} -> {self.tag_mm.value:.1f} mm) "
                               f"- zrob krok 2 jeszcze raz z nowym bokiem")
        self.ws.card["tag_size"] = used
        names = self.ws.apply_fit(fit, tag_size=used, intrinsics=self.calib_job.data.get("intrinsics"))
        preset = self.ws.save_preset()                  # nowy plik - poprzednie zapisy zostaja
        self._poses_changed(names)
        self._calib_pending = False
        self.calib_md.content = f"Zapisano polozenie: {', '.join(names)} (zestaw {preset.stem})."
        self._render_calibration()
        return names

    def _poses_changed(self, names: list[str]) -> None:
        """Po nowej pozie kamer (zapis kroku 2 albo wgranie zestawu): plik, scena, kadry odniesienia."""
        for n in names:
            self._simreal_px.pop(n, None)               # nowa poza - Sim-Real do sprawdzenia od nowa
        self._save()
        self.twin.rebuild()
        for n in names:
            try:
                self._remember_reference(n)
            except RuntimeError:
                pass
        self._refresh_cameras()

    def _start_card_calibration(self, event, cameras: list[str], quick: bool) -> None:
        mark = self._motion_mark()                      # STOP/Dom/Polacz od teraz - fala nie bierze ramienia
        if not self.twin.connected:
            raise RuntimeError("polacz ramie (sim albo prawdziwe) w zakladce Ramie")
        if not cameras or cameras == ["-"]:
            raise RuntimeError("zadna wlaczona kamera nie daje obrazu")
        if self.calib_job.running:
            raise RuntimeError("ramie juz pokazuje karte")
        # Fala bierze ramie na wlasnosc PRZED startem: jedzie polityka albo identyfikacja -
        # odmowa z powodem (wczesniej trzy watki pisaly cel na zmiane i ramie skakalo).
        # Odebranie ramienia (Dom, STOP, polaczenie) przerywa fale przez `calib_job.stop`.
        self._take_arm(jobs.CALIB_OWNER, preempt=self.calib_job.stop, mark=mark)
        self.ws.card["tag_size"] = self.tag_mm.value / 1000
        self._calib_pending = False
        self.calib_md.content = ""
        try:
            self.calib_job.start(lambda job: jobs.run_card_calibration(job, self.twin, cameras, quick))
        except Exception:
            self.twin.release(jobs.CALIB_OWNER)
            raise
        self.calib_ask.reset()                          # nastepny start (np. poruszona kamera) - nowe potwierdzenie
        self._render_calibration()

    def _tick_calibration(self) -> None:
        j = self.calib_job
        if j.running:
            self.calib_bar.value = 100 * j.progress
            prog = j.data.get("progress", {})
            rows = [f"- {c}: widziala karte {o} razy" for c, (o, p, s) in prog.items()]
            self.calib_md.content = ("**Ramie pokazuje karte kamerom - nie wchodz w jego zasieg.**\n\n"
                                     + "\n".join(rows))
        elif j.state == jobs.DONE and j.result is not None and not self._calib_pending and \
                j.data.get("shown") is not j.result:
            fit = j.result
            rows = ["**Wynik** - kliknij Zapisz wynik:", "", "| kamera | wynik |", "|---|---|"]
            for n, c in fit.cameras.items():
                extra = ""
                rec = next((r for r in self.ws.cameras if r.name == n), None)
                if rec is not None and rec.simulated and rec.sim_pose is not None:
                    from ..calib.handeye import pose_error
                    dt, dr = pose_error(np.asarray(rec.sim_pose), c.T_cam2base)
                    extra = f" (od prawdy symulacji: {dt * 1000:.1f} mm)"
                # Ten sam werdykt, ktory zapisze `Workspace.apply_fit`: K, z ktorym LICZONO poze
                # (z fali), musi byc zaufane i wciaz takie samo jak w kamerze.
                k_problem = rec.fit_problem((j.data.get("intrinsics") or {}).get(n)) if rec is not None else ""
                bad = c.reason if not c.trusted else k_problem
                verdict = (f"**zle** - {bad}; popraw ustawienie kamery i zrob krok 2 jeszcze raz" if bad else
                           f"**dobrze** (dokladnosc {c.rms_px:.2f} px){extra}")
                rows.append(f"| {n} | {verdict} |")
            self.calib_md.content = "\n".join(rows)
            self._calib_pending = True
            j.data["shown"] = fit
        elif j.state in (jobs.FAILED, jobs.CANCELLED) and j.data.get("shown") != j.state:
            # Przerwana przez operatora (STOP, Dom, Zatrzymaj) to nie blad (`jobs.interrupted_by_operator`).
            self.calib_md.content = (f"**Zatrzymane** ({j.message or 'przerwane'}) - nic nie zapisano."
                                     if j.state == jobs.CANCELLED else
                                     f"**Nie udalo sie:** {j.error or j.message}")
            j.data["shown"] = j.state

        ij = self.intr_job
        if ij.running:
            col = ij.data.get("collector")
            img = ij.data.get("last")
            if img is not None:
                vis = img.copy()
                det = ij.data.get("last_det")
                if det is not None:
                    for u, v in det[1]:
                        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0), -1)
                self.intr_img.image = thumb(vis, 480)
            if col is not None:
                self.intr_md.content = (
                    f"**Trzymaj szachownice przed kamera {ij.data.get('camera', '')} i powoli nia ruszaj:** w kazdym "
                    f"rogu obrazu, blisko i daleko, pochylona w lewo, w prawo, w gore i w dol. Zdjecia robia sie "
                    f"same (zielone kropki = szachownica widoczna).\n\n"
                    f"Zdjecia: **{len(col.views)} z 12** - pokrycie obrazu: **{col.coverage:.0%} z 55%**")
        elif ij.state == jobs.DONE and ij.result is not None and ij.data.get("saved") is not ij.result:
            res = ij.result
            ij.data["saved"] = res
            # Kamera, z ktorej ZBIERANO kadry - lista w panelu mogla sie w tym czasie zmienic
            # (K kamery1 ladowalo w kamerze2 i odbieralo jej zaufanie do pozy).
            name = ij.data.get("camera")
            try:
                rec = self.ws.camera(name)
            except KeyError:
                self.intr_md.content = f"**Kamery {name} juz nie ma** - nic nie zapisano"
                return
            self._set_lens(rec, res)
            self.intr_md.content = (f"Obiektyw **{name}** zmierzony - dobrze." if res.trusted else
                                    f"Obiektyw **{name}**: wynik niepewny ({res.reason}). Zmierz jeszcze raz - "
                                    f"wiecej pochylen i rogow obrazu.")
            todo = self._lens_todo()
            if todo and self.intr_cam.value not in todo:
                self.intr_cam.value = todo[0]           # nastepna kamera bez obiektywu
        elif ij.state == jobs.FAILED and ij.data.get("saved") != "err":
            self.intr_md.content = f"**Nie udalo sie:** {ij.error}"
            ij.data["saved"] = "err"
        elif ij.state == jobs.CANCELLED and ij.data.get("saved") != "cancel":
            self.intr_md.content = "Anulowano - kamera bez zmian."
            ij.data["saved"] = "cancel"
        self._render_calibration()

    # ================================================================== mapa
    def _build_map(self) -> None:
        g = self.server.gui
        self._tab_help("Mapa")
        g.add_markdown("Kadry wszystkich skalibrowanych kamer przerysowane na blat i zszyte - przedmiot na stole "
                       "trafia w swoje prawdziwe (x, y), skadkolwiek patrzy kamera.")
        self.map_on = g.add_checkbox("Mapa na zywo", True,
                                     hint="Zszywa kadry zaufanych kamer w mape blatu (ok. 3 razy na s). Wylaczenie "
                                          "zatrzymuje polityke, ktora bierze kostke z kamer.")
        self.map_3d = g.add_checkbox("Pokaz mape na blacie w 3D", True,
                                     hint="Kladzie mape na blacie w widoku 3D - tylko widok.")
        self.map_img = g.add_image(np.zeros((224, 224, 3), np.uint8), label="Mapa blatu", format="jpeg",
                                   jpeg_quality=80)
        self.cube_on = g.add_checkbox("Szukaj kostki", True,
                                      hint="Wykrywa kostke 3 cm w kolorze ponizej (piksele ramienia nie glosuja). "
                                           "Potrzebne dla lift z kamer; wylaczenie zatrzymuje taka polityke.")
        self.cube_color = g.add_dropdown("Kolor kostki", tuple(CUBE_COLORS), initial_value="czerwona",
                                         hint="Kolor kostki dla detektora (progi HSV); dziala od razu.")
        self.map_md = g.add_markdown("")
        self.map_node = None
        self.cube_node = self.server.scene.add_box("/percepcja/kostka", color=(255, 80, 200), dimensions=(0.03,) * 3,
                                                   opacity=0.5, wireframe=True, visible=False)

        @self.cube_color.on_update
        def _(event):
            lo, hi, lo2, hi2 = CUBE_COLORS[self.cube_color.value]
            self.cube_det.hsv_lo, self.cube_det.hsv_hi, self.cube_det.hsv_lo2, self.cube_det.hsv_hi2 = lo, hi, lo2, hi2

        def vision_off(event):
            # Bez mapy nie ma detekcji kostki - polityka z kostka "z kamer" jechalaby do
            # ostatniego polozenia. Zatrzymujemy ja jawnie, zamiast czekac na `hold_s`.
            if event.client is None or (self.map_on.value and self.cube_on.value):
                return
            if self._vision_policy_running():
                self.runner.stop("mapa albo szukanie kostki wylaczone - kostka z kamer niedostepna")
                self._notify(event, "Polityka zatrzymana", "Bez mapy i szukania kostki polityka nie wie, gdzie ona jest.")
        self.map_on.on_update(vision_off)
        self.cube_on.on_update(vision_off)

    def _vision_policy_running(self) -> bool:
        return (self.runner is not None and self.runner.status.running
                and self.runner.cube_provider == self._vision_cube)

    def _mapper_now(self) -> TableMapper | None:
        key = tuple((c.name, c.enabled, c.trusted, str(c.T_cam2base), str(c.K)) for c in self.ws.cameras)
        if key != self._mapper_key:
            self._mapper_key = key
            m = TableMapper.from_workspace(self.ws, only_trusted=True)
            self.mapper = m if m.cameras else None
        return self.mapper

    def _tick_map(self, frames: dict[str, np.ndarray]) -> None:
        """Mapa i detekcja kostki. `last_cube` dostaje NOWA detekcje albo None - nigdy stara.

        Wczesniej `last_cube` zostawal z ostatniej detekcji, gdy mapa byla wylaczona,
        brakowalo kamer albo cos rzucilo - polityka dostawala ja co takt jako swieza,
        `hold_s` trackera nie mijal i zgubiona kostka nigdy nie zatrzymywala ramienia.
        Przypisanie na koncu (zamiast None na poczatku) - watek polityki nie widzi
        "nie ma kostki" przez caly czas dopasowania.
        """
        det = None
        try:
            det = self._map_and_detect(frames)
        finally:
            self.last_cube = det

    def _map_and_detect(self, frames: dict[str, np.ndarray]):
        mapper = self._mapper_now() if self.map_on.value else None
        if mapper is None:
            self.map_md.content = "Brak zaufanej, skalibrowanej kamery - najpierw kalibracja." if self.map_on.value \
                else ""
            if self.map_node is not None:
                self.map_node.visible = False
            self.cube_node.visible = False
            return None
        table, wsum = mapper.fuse(frames)
        self._img(self.map_img, "mapa", table)
        cover = float((wsum > 1e-6).mean())
        txt = f"pokrycie blatu: **{cover:.0%}** z {len(mapper.cameras)} kamer"
        # Wiersz 0 mapy to brzeg +y podstawy, a viser kladzie wiersz 0 obrazu po stronie -y
        # wezla: bez odwrocenia wierszy mapa na blacie byla lustrem w y (kostka z +0,10 m
        # rysowala sie na -0,10 m, obok poprawnej ramki kostki).
        table3d = np.ascontiguousarray(table[::-1])
        Tm = self.T_b2w @ pose(np.eye(3), np.array([mapper.centre[0], mapper.centre[1], 0.0015]))
        if self.map_3d.value:
            if self.map_node is None:
                self.map_node = self.server.scene.add_image("/mapa", table3d, mapper.side, mapper.side,
                                                            format="jpeg", position=Tm[:3, 3],
                                                            wxyz=mat_to_wxyz(Tm[:3, :3]))
                self._map_pose_key = Tm.round(6).tobytes()
            else:
                self._img(self.map_node, "mapa3d", table3d)
                key = Tm.round(6).tobytes()
                if key != self._map_pose_key:          # stol przestawiony ("Zastosuj") - mapa za podstawa
                    self.map_node.position, self.map_node.wxyz = Tm[:3, 3], mat_to_wxyz(Tm[:3, :3])
                    self._map_pose_key = key
                self.map_node.visible = True
        elif self.map_node is not None:
            self.map_node.visible = False
        det = None
        if self.cube_on.value:
            names = [n for n in mapper.cameras if n in frames]
            now = time.monotonic()
            with self.frame_lock:
                times = dict(self.frame_times)
            # Chwila detekcji = chwila NAJSTARSZEGO uzytego kadru (konsument ocenia jej wiek).
            t_cams = {n: times.get(n, now) for n in names}
            t_frames = min(t_cams.values(), default=now)
            # Piksele zasloniete ramieniem (maska z blizniaka, w pozie z chwili kadru KAZDEJ kamery) nie glosuja.
            occ = self._arm_masks(names, t_cams, dilate=5) if names else None
            if names:
                det = self.cube_det.detect_frames({n: frames[n] for n in names}, mapper, occ, t=t_frames)
            if det is not None:
                Tw = self.T_b2w @ pose(det.rot, det.pos)
                self.cube_node.position, self.cube_node.wxyz = Tw[:3, 3], mat_to_wxyz(Tw[:3, :3])
                self.cube_node.visible = True
                yaw = np.degrees(np.arctan2(det.rot[1, 0], det.rot[0, 0]))
                # n_cameras -1 = detekcja z samej mapy (nie wiadomo, ile kamer ja widzialo)
                cams = f"{det.n_cameras} kam." if det.n_cameras >= 0 else "z mapy"
                txt += (f"  \nkostka: x {det.pos[0] * 100:.1f} cm, y {det.pos[1] * 100:.1f} cm, obrot {yaw:.0f} st., "
                        f"pewnosc {det.confidence:.2f}, {cams}, "
                        f"kadr sprzed {1000 * (time.monotonic() - det.t):.0f} ms")
            else:
                self.cube_node.visible = False
                txt += "  \nkostka: nie widac"
        else:
            self.cube_node.visible = False
        self.map_md.content = txt
        return det

    # =============================================================== trening
    def _build_training(self) -> None:
        g = self.server.gui
        self._tab_help("Trening")
        with g.add_folder("Nowy trening (PPO na GPU, MuJoCo Warp)"):
            self.tr_task = g.add_dropdown("Zadanie", ("reach", "lift", "pickplace"), initial_value="reach",
                                          hint="reach - koncowka do punktu; lift - chwyt i podniesienie kostki; "
                                               "pickplace - kostka przeniesiona na cel na blacie i puszczona. "
                                               "Przy Start z polityki zadanie bierze sie z polityki.")
            self.tr_envs = g.add_dropdown("Swiatow naraz", ("1024", "2048", "4096", "8192"), initial_value="4096",
                                          hint="Ile swiatow MuJoCo Warp naraz na GPU; 4096 sprawdzone na A4500, "
                                               "wiecej = wiecej pamieci GPU.")
            self.tr_iters = g.add_number("Iteracje", 200, min=10, max=5000, step=10,
                                         hint="Iteracje PPO. Od zera: reach ok. 160 (~2,5 min), lift ok. 600 "
                                              "(~30 min); douczanie 100-300.")
            self.tr_rand = g.add_checkbox("Randomizacja dziedziny", True,
                                          hint="Losuje dynamike serw, kostke i opoznienia wokol zmierzonej "
                                               "dynamiki. Odznaczone = dokladnie zmierzona (model percepcji "
                                               "kostki zostaje).")
            self.tr_spread = g.add_slider("Szerokosc randomizacji", 0.0, 2.0, 0.1, 1.0,
                                          hint="Mnoznik zakresow randomizacji (0-2, 1 = domyslne) wokol dynamiki "
                                               "zmierzonej na ramieniu (Dynamika serw ponizej).")
            self.tr_name = g.add_text("Nazwa (puste = zadanie + data)", "",
                                      hint="Katalog nowej polityki; ta sama nazwa co istniejaca nadpisuje jej "
                                           "przebieg.")
            self.tr_init = g.add_dropdown("Start z polityki", ("(od zera)",), initial_value="(od zera)",
                                          hint="Douczanie z wag wybranej polityki (np. reach-v3 / lift-v3 po "
                                               "identyfikacji dynamiki), 100-300 iteracji. Zadanie bierze sie "
                                               "wtedy z polityki.")
            start = g.add_button("Ucz", icon=viser.Icon.PLAYER_PLAY, color="green",
                                 hint="Trening w osobnym procesie na GPU (pierwsza iteracja po ~20 s kompilacji). "
                                      "Ramieniem nie rusza; drugi trening naraz - odmowa.")
            stop = g.add_button("Zatrzymaj (zapisze polityke)", icon=viser.Icon.PLAYER_STOP,
                                hint="Konczy trening i zapisuje polityke z miejsca, w ktorym stanal.")
            self.tr_bar = g.add_progress_bar(0.0, animated=True, visible=False)
            self.tr_md = g.add_markdown("")
            self.tr_plot = g.add_uplot(
                (np.array([0.0]), np.array([0.0])),
                (uplot.Series(label="mln krokow"), uplot.Series(label="sukces", stroke="#ffc400", width=2)),
                title="Sukces w trakcie treningu", aspect=1.6, visible=False,
                scales={"y": uplot.Scale(range=(0, 1))})
        with g.add_folder("Dynamika serw (sim-to-real)"):
            self.dyn_md = g.add_markdown("")
            ident = g.add_button("Identyfikuj na polaczonym ramieniu (~20 s ruchu)", icon=viser.Icon.ACTIVITY,
                                 hint="RUSZA ramieniem ~20 s (ruchy po 12 st.; do pozy startowej 30 st./s), potem "
                                      "7-11 s dopasowania bez ramienia. Odmowa, gdy ramie ma inne zadanie.")
            self.dyn_confirm = g.add_checkbox("Prawdziwe ramie: przestrzen wokol wolna", False,
                                              hint="Prawdziwe ramie: potwierdzenie przed kazda identyfikacja (gasnie "
                                                   "po starcie). W sim niepotrzebne.")
            self.dyn_bar = g.add_progress_bar(0.0, visible=False)
            self.dyn_res = g.add_markdown("")
            self.dyn_force = g.add_checkbox("Zapisz mimo to (wynik nic nie wyjasnia)", False, visible=False,
                                            hint="Tylko gdy identyfikacja uznala wynik za nieprzydatny (niepewnosc "
                                                 "ponad 30% albo blad prawie jak model). Zaznaczone odblokowuje "
                                                 "Zapisz; przewodnik i tak nie zalicza kroku 5.")
            keep = g.add_button("Zapisz jako dynamike stanowiska", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                visible=False,
                                hint="Wynik identyfikacji staje sie dynamika stanowiska - srodkiem randomizacji "
                                     "w kolejnych treningach. Wynik, ktory nic nie wyjasnia, zapisuje dopiero po "
                                     "zaznaczeniu 'Zapisz mimo to'.")
            reset = g.add_button("Wroc do modelu Menagerie", icon=viser.Icon.RESTORE,
                                 hint="Kasuje zapisana dynamike (bez pytania): trening wraca do modelu Menagerie.")
            self.dyn_keep = keep
        self._show_dynamics()

        @start.on_click
        @self._safe
        def _(event):
            name = self.tr_name.value.strip() or f"{self.tr_task.value}-{time.strftime('%Y%m%d-%H%M%S')}"
            init = None
            if self.tr_init.value != "(od zera)":
                from ..rl.policy import list_policies
                init = next(p["path"] for p in list_policies(self.policies_dir) if p["name"] == self.tr_init.value)
            self._save()
            self.train.start(self.tr_task.value, int(self.tr_envs.value), int(self.tr_iters.value),
                             float(self.tr_spread.value), self.tr_rand.value, name, init=init)
            self._notify(event, "Trening ruszyl", f"{name}: pierwsza iteracja po kompilacji kerneli (~20 s)")

        @stop.on_click
        def _(event):
            self.train.stop()

        @ident.on_click
        @self._safe
        def _(event):
            mark = self._motion_mark()                  # STOP/Dom/Polacz od teraz - nagranie nie bierze ramienia
            if not self.twin.connected:
                raise RuntimeError("polacz ramie w zakladce Ramie")
            if not self.twin.status.simulated and not self.dyn_confirm.value:
                raise RuntimeError("potwierdz, ze wokol ramienia jest wolne miejsce")
            if self.sysid_job.running:
                if "recording" in self.sysid_job.data:
                    raise RuntimeError("trwa dopasowanie poprzedniego nagrania - poczekaj na wynik")
                raise RuntimeError("identyfikacja juz trwa")
            # Ramie dla identyfikacji albo odmowa z powodem - fala czy polityka w toku mieszaly
            # swoje cele z pobudzeniem, a nagranie z obu szlo do dopasowania dynamiki.
            self._take_arm(jobs.SYSID_OWNER, preempt=self.sysid_job.stop, mark=mark)
            self.dyn_keep.visible = False
            self.dyn_force.visible = False
            try:
                self.sysid_job.start(lambda job: jobs.run_sysid(job, self.twin))
            except Exception:
                self.twin.release(jobs.SYSID_OWNER)
                raise
            self.dyn_confirm.value = False

        @keep.on_click
        @self._safe
        def _(event):
            self._keep_dynamics()

        @self.dyn_force.on_update
        def _(event):
            self.dyn_keep.disabled = not (self.sysid_job.data.get("useful", True) or self.dyn_force.value)

        @reset.on_click
        @self._safe
        def _(event):
            self.ws.dynamics = {}
            self._save()
            self._show_dynamics()

    def _keep_dynamics(self) -> None:
        """Zapisz jako dynamike stanowiska - wynik, ktory nic nie wyjasnia, tylko z 'Zapisz mimo to'.

        Zmierzone (verify2, s6): dopasowanie 0,55 -> 0,51 st. z armatura +-inf dalo sie zapisac
        jednym kliknieciem, a trening centrowal randomizacje na armaturze x0,45 (prawda 1,0 poza
        kazdym swiatem treningu). Taki zapis dostaje znacznik w zrodle - przewodnik go nie zalicza.
        """
        import dataclasses

        j = self.sysid_job
        dyn = j.result
        if dyn is None:
            raise RuntimeError("brak wyniku identyfikacji")
        useful = bool(j.data.get("useful", True))
        if not useful:
            if not self.dyn_force.value:
                raise RuntimeError("wynik nic nie wyjasnia - powtorz identyfikacje albo zaznacz 'Zapisz mimo to'")
            why = j.data.get("useless_why") or "identyfikacja uznala wynik za nieprzydatny"
            dyn = dataclasses.replace(dyn, source=f"{dyn.source}; {gd.USELESS_MARK}: {why}")
        self.ws.dynamics = dyn.to_dict()
        self._save()
        self.dyn_keep.visible = self.dyn_force.visible = False
        self.dyn_force.value = False
        self._show_dynamics()

    def _show_dynamics(self) -> None:
        from ..rl.randomize import Dynamics
        d = Dynamics.from_dict(self.ws.dynamics)
        names = fitted_names(d)
        if names:
            fitted, fixed = dynamics_parts(d, names, getattr(d, "band", None))
            vals = f"dopasowane: {', '.join(fitted)}" + (f"; z modelu: {', '.join(fixed)}" if fixed else "")
        else:                                           # model albo zapis bez listy dopasowanych
            vals = ", ".join(sum(dynamics_parts(d, ()), []))
        self.dyn_md.content = (f"Srodek randomizacji: **{d.source}** - {vals}"
                               + (f", blad dopasowania {d.fit_deg:.2f} st." if d.fit_deg else "")
                               + ("" if gd.dynamics_useful(self.ws.dynamics) else
                                  "  \n**Ta dynamika nic nie wyjasnia** - powtorz identyfikacje albo "
                                  "Wroc do modelu Menagerie."))

    def _tick_training(self) -> None:
        p = self.train.progress()
        running = self.train.running
        self.tr_bar.visible = running
        if p:
            self.tr_bar.value = 100 * p["iteration"] / max(1, p["iterations"])
            hist = p.get("history", [])
            if hist:
                self.tr_plot.visible = True
                self.tr_plot.data = (np.array([h["steps"] / 1e6 for h in hist]), np.array([h["success"] for h in hist]))
            state = "trwa" if running else ("koniec" if self.train.exit_code == 0 else
                                            f"proces zakonczony kodem {self.train.exit_code}")
            self.tr_md.content = (f"**{self.train.run_dir.name}** ({state}): iteracja {p['iteration']}/{p['iterations']}, "
                                  f"{p['steps'] / 1e6:.1f} mln krokow, {p['fps'] / 1e3:.0f} tys./s, "
                                  f"sukces **{p['success']:.0%}**, {p['elapsed']:.0f} s")
            if not running and self.train.exit_code is not None:
                tail = self.train.tail(4)
                if "Ewaluacja" in tail or "Zapisano" in tail:
                    self.tr_md.content += f"\n\n```\n{tail}\n```"
        elif running:
            self.tr_md.content = "kompilacja kerneli MuJoCo Warp i pierwsza iteracja..."
        j = self.sysid_job
        self.dyn_bar.visible = j.running
        if j.running:
            self.dyn_bar.value = 100 * j.progress
            self.dyn_res.content = j.message
        elif j.state == jobs.DONE and j.result is not None and j.data.get("shown") is not j.result:
            d = j.result
            band = j.data.get("band") or getattr(d, "band", None) or {}
            fitted, fixed = dynamics_parts(d, fitted_names(d, sysid_default=True), band)
            head = (f"Dopasowano: {', '.join(fitted)}" if fitted else
                    "Wynik (nie wiadomo, ktore parametry dopasowano)") + \
                   (f"; z modelu (nie dopasowane): {', '.join(fixed)}" if fixed else "")
            # Na czym nagrano: wynik z sim, ktory skonczyl sie dopasowywac po Polacz do prawdziwego
            # ramienia, wygladal jak dynamika stanowiska.
            rec_backend = str(j.data.get("backend") or "?")
            st = self.twin.status
            now_backend = st.backend if st.connected else ""
            where = f"Nagrane na: **{rec_backend}**" + (
                f" (polaczone jest teraz: {now_backend} - to NIE jest jego dynamika)"
                if now_backend and rec_backend not in ("?", now_backend) else "")
            useful = bool(j.data.get("useful", True))
            why = j.data.get("useless_why") or "identyfikacja uznala go za nieprzydatny"
            verdict = "" if useful else (
                f"  \n**Wynik nic nie wyjasnia** ({why}) - powtorz identyfikacje z wolnym ramieniem; "
                f"zapis tylko po zaznaczeniu 'Zapisz mimo to'.")
            self.dyn_res.content = (f"{head}. Blad symulacji wzgledem nagrania: "
                                    f"**{j.data.get('base', float('nan')):.2f} st. -> {d.fit_deg:.2f} st.**  \n"
                                    f"{where}{verdict}")
            self.dyn_force.value = False
            self.dyn_force.visible = not useful
            self.dyn_keep.disabled = not useful
            self.dyn_keep.visible = True
            j.data["shown"] = d
        elif j.state == jobs.FAILED and j.data.get("shown") != "err":
            self.dyn_res.content = f"**Blad**: {j.error}"
            self.dyn_keep.visible = self.dyn_force.visible = False   # "Zapisz" zapisywalby wynik, ktorego nie ma
            j.data["shown"] = "err"
        elif j.state == jobs.CANCELLED and j.data.get("shown") != "cancel":
            # Wczesniej przerwane zadanie znikalo bez slowa - pasek gasl, a wyniku nie bylo.
            self.dyn_res.content = f"**Przerwane**: {j.message or 'identyfikacja przerwana'}"
            self.dyn_keep.visible = self.dyn_force.visible = False
            j.data["shown"] = "cancel"

    # =============================================================== polityki
    def _build_policies(self) -> None:
        g = self.server.gui
        self._tab_help("Polityki")
        self.pol_pick = g.add_dropdown("Polityka", ("-",), initial_value="-",
                                       hint="Polityka do ewaluacji i uruchomienia: bazowe (reach-v3, lift-v3) i "
                                            "nauczone na stanowisku.")
        refresh = g.add_button("Odswiez liste", icon=viser.Icon.REFRESH,
                               hint="Czyta liste polityk z dysku od nowa (np. po treningu z wiersza polecen).")
        self.pol_md = g.add_markdown("")
        ev = g.add_button("Ewaluuj na CPU (zwykle MuJoCo, 50 epizodow)", icon=viser.Icon.CHART_BAR,
                          hint="Epizody w zwyklym MuJoCo na CPU, bez randomizacji i z nia; wynik zapisuje w "
                               "polityce (bazowych z repozytorium nie nadpisuje). Ramieniem nie rusza.")
        self.pol_eval_md = g.add_markdown("")
        with g.add_folder("Uruchom na blizniaku"):
            self.pol_cube_src = g.add_dropdown("lift/pickplace: skad polozenie kostki", ("symulacja", "kamery"),
                                               initial_value="symulacja",
                                               hint="Tylko lift i pickplace. symulacja - kostka z blizniaka (tylko ramie sim); "
                                                    "kamery - ta sama percepcja co na biurku (zakladka Mapa, zaufana "
                                                    "kamera). Prawdziwe ramie: tylko kamery.")
            self.pol_confirm = g.add_checkbox("Prawdziwe ramie: rozumiem, ze sie ruszy", False,
                                              hint="Prawdziwe ramie: potwierdzenie przed kazdym Uruchom (gasnie po "
                                                   "starcie). W sim niepotrzebne.")
            run = g.add_button("Uruchom", icon=viser.Icon.PLAYER_PLAY, color="green",
                               hint="Polityka jedzie na polaczonym ramieniu (przez nadzor) - RUSZA ramieniem. "
                                    "Odmowa przy aktywnym STOP-ie albo gdy ramie ma fala / identyfikacja; "
                                    "STOP, Dom albo Zatrzymaj w trakcie startu - nie rusza. Panel sie rozbraja.")
            halt = g.add_button("Zatrzymaj", icon=viser.Icon.PLAYER_STOP,
                                hint="Zatrzymuje polityke - takze jeszcze ladowana po Uruchom; ramie trzyma "
                                     "ostatni cel.")
            new_goal = g.add_button("reach: losowy cel", icon=viser.Icon.TARGET,
                                    hint="Nowy losowy cel z obszaru treningu (zolta kulka); gdy jedzie reach, "
                                         "ramie za nim pojedzie. Cel mozna tez przeciagac w 3D.")
            new_cube = g.add_button("lift (sim): poloz kostke losowo", icon=viser.Icon.CUBE,
                                    hint="Tylko w blizniaku: kladzie kostke w losowym miejscu blatu symulacji "
                                         "(dodaje ja do sceny, gdy jej nie ma). Ramieniem nie rusza.")
            self.pol_run_md = g.add_markdown("")
        with g.add_folder("Tag na biurku"):
            self.tag_id = g.add_number("id taga (AprilTag 36h11)", 0, min=0, max=586, step=1,
                                       hint="Tag lezacy na blacie; bok czarnego kwadratu bierze sie z pola "
                                            "'Zmierzony bok taga' w zakladce Kamery.")
            self.tag_above_cm = g.add_number("wysokosc nad tagiem [cm]", 8.0, min=2.0, max=30.0, step=1.0,
                                             hint="Ile nad srodkiem taga (wzdluz osi z podstawy) ma stanac "
                                                  "koncowka po 'Jedz nad tag'.")
            find_tag = g.add_button("Znajdz tag w kamerach", icon=viser.Icon.TARGET,
                                    hint="Kadry wlaczonych kamer z zaufana poza -> poza taga w ukladzie "
                                         "podstawy (mediana z kamer), zielona plytka w 3D. Ramieniem nie rusza.")
            goto_tag = g.add_button("Jedz nad tag", icon=viser.Icon.PLAYER_PLAY, color="orange",
                                    hint="RUSZA ramieniem: IK do punktu nad znalezionym tagiem, rampa przez "
                                         "nadzor. Wymaga wolnego ramienia (panel rozbrojony) i, na prawdziwym "
                                         "ramieniu, zaznaczonego potwierdzenia z zakladki Polityki.")
            self.tag_md = g.add_markdown("")
        self.specs = add_specs_folder(self, g)
        self._tag_fix = None
        self.goal_node = self.server.scene.add_icosphere("/cel", radius=0.012, color=(255, 196, 0), visible=False)
        self.goal_gizmo = self.server.scene.add_transform_controls("/cel_uchwyt", scale=0.08, disable_rotations=True,
                                                                   visible=False)
        self._refresh_policies()

        @refresh.on_click
        def _(event):
            self._refresh_policies()

        @self.pol_pick.on_update
        def _(event):
            self._show_policy()

        @ev.on_click
        @self._safe
        def _(event):
            path = self._policy_path()
            self.eval_job.start(lambda job: jobs.run_eval(job, path, self.ws))

        @self.goal_gizmo.on_update
        def _(event):
            self._set_goal((inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3])

        @new_goal.on_click
        @self._safe
        def _(event):
            from ..rl import task as tk
            g_ = tk.sample_goals(self.twin.scene.kin, tk.make_task("reach", self.ws.spec()), np.random.default_rng(), 1)[0]
            self._set_goal(g_)

        @new_cube.on_click
        @self._safe
        def _(event):
            self._place_cube_randomly()

        @find_tag.on_click
        @self._safe
        def _(event):
            self._find_tag()

        @goto_tag.on_click
        @self._safe
        def _(event):
            self._goto_tag()

        @run.on_click
        @self._safe
        def _(event):
            self._start_policy(event)

        @halt.on_click
        def _(event):
            self._halt_policy()

    def _policy_path(self) -> str:
        from ..rl.policy import list_policies
        for p in list_policies(self.policies_dir):
            if p["name"] == self.pol_pick.value:
                return p["path"]
        raise RuntimeError("wybierz polityke")

    def _refresh_policies(self) -> None:
        from ..rl.policy import list_policies
        items = list_policies(self.policies_dir)
        self._policy_list = items
        opts = tuple(p["name"] for p in items) or ("-",)
        cur = self.pol_pick.value
        self.pol_pick.options = opts
        self.pol_pick.value = cur if cur in opts else opts[0]
        if hasattr(self, "tr_init"):
            init_opts = ("(od zera)",) + tuple(p["name"] for p in items)
            if self.tr_init.options != init_opts:
                cur = self.tr_init.value
                self.tr_init.options = init_opts
                self.tr_init.value = cur if cur in init_opts else "(od zera)"
        self._show_policy()

    def _show_policy(self) -> None:
        from ..rl.policy import list_policies
        p = next((x for x in list_policies(self.policies_dir) if x["name"] == self.pol_pick.value), None)
        if p is None:
            self.pol_md.content = "Brak polityk - naucz pierwsza w zakladce Trening."
            return
        ev = p["evals"]
        lines = [f"**{p['name']}** - zadanie `{p['task']}`, {p['steps'] / 1e6:.1f} mln krokow, {p['created']}",
                 f"- GPU (trening, z eksploracja): {p['success'] or 0:.0%}"]
        for k, label in (("cpu_nominal", "CPU bez randomizacji"), ("cpu_rand", "CPU z randomizacja")):
            if k in ev:
                e = ev[k]
                extra = ", ".join(f"{kk} {vv:.1f}" for kk, vv in e.items() if kk.endswith("_mm"))
                lines.append(f"- {label}: **{e['success']:.0%}** ({extra})")
        self.pol_md.content = "\n".join(lines)

    def _set_goal(self, p_base: np.ndarray) -> None:
        """Cel reach: rzutowany na obszar, z ktorego losowano cele w treningu, i tam pokazany.

        Uchwyt celu mozna bylo zaciagnac pod blat - polityka bez limitu epizodu wciskala
        wtedy szczeki w stol, az serwo zablokowalo sie na 25 st. rozjazdu. Uchwyt
        wraca na rzutowany punkt, zeby bylo widac, dokad ramie naprawde jedzie.
        """
        from ..rl import task as tk
        from ..rl.runner import project_goal
        runner = self.runner if self.runner is not None and self.runner.task.name in ("reach", "pickplace") \
            else None
        if runner is not None and runner.task.target:
            # pickplace: cel odlozenia na blacie (z = 0) i w pierscieniu celow z treningu;
            # podkladka celu w blizniaku idzie za nim.
            runner.goal = np.asarray(p_base, float)
            g = np.asarray(runner.target, float)
            self._show_target_pad(g)
        elif runner is not None:
            runner.goal = np.asarray(p_base, float)       # setter rzutuje na obszar zadania tej polityki
            g = np.asarray(runner.goal, float)
        else:
            if self._reach_task is None:
                self._reach_task = tk.make_task("reach", self.ws.spec())
            g = project_goal(self._reach_task, np.asarray(p_base, float))
        self._goal_base = g
        pw = (self.T_b2w @ np.r_[g, 1.0])[:3]
        self.goal_node.position = pw
        if np.abs(np.asarray(self.goal_gizmo.position) - pw).max() > 1e-4:
            self.goal_gizmo.position = pw

    def _cube_in_scene(self) -> bool:
        return self._body_in_scene("cube")

    def _body_in_scene(self, name: str) -> bool:
        with self.twin.lock:
            return name in [self.twin.scene.model.body(b).name for b in range(self.twin.scene.model.nbody)]

    def _configure_objects(self, cube: bool, pad: bool) -> None:
        """Kostka i/lub podkladka celu pickplace w scenie blizniaka (przebudowa, stan zostaje)."""
        from ..rl import task as tk
        from ..rl.env import TARGET_RGBA, TARGET_SIZE
        task = tk.make_task("lift", self.ws.spec())
        h = task.cube_half
        objs, sensors = [], []
        if cube:
            objs.append(sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0), mass=task.cube_mass))
            sensors.append("cube")
        if pad:
            # Tylko do ogladania (mocap, bez kolizji) - kostka przez nia przenika.
            objs.append(sc.Box("target", TARGET_SIZE, (0.3, 0.1), rgba=TARGET_RGBA, free=False, mocap=True))
        self.twin.configure(objects=objs, grasp_sensors=sensors)

    def _show_target_pad(self, target_base: np.ndarray) -> None:
        """Podkladka celu pickplace w blizniaku na `target_base` (uklad podstawy, blat)."""
        from ..rl.env import TARGET_SIZE
        if not self._body_in_scene("target"):
            return
        with self.twin.lock:
            s = self.twin.scene
            k = int(s.model.body_mocapid[s.model.body("target").id])
            T = s.T_base2world
            s.data.mocap_pos[k] = T[:3, :3] @ np.array([target_base[0], target_base[1], TARGET_SIZE[2]]) + T[:3, 3]

    # ------------------------------------------------------------ kamera PMD z tagami
    def _build_pmd(self, g) -> None:
        """Folder "Kamera PMD" w zakladce Kamery: tagi na obudowie -> uklad tagow z kadrow -> poza w 3D.

        Stan: `self.pmd` (`TagBody`: numery, bok, uklad, os optyczna), `_pmd_obs` (kadry do nauki
        ukladu), `_pmd_T` (ostatnia poza w ukladzie podstawy), `_pmd_nodes` (wezly 3D pod /pmd).
        Z pliku stanowiska (sekcja `pmd`) wraca uklad, os i poza - PMD jest w 3D od startu.
        """
        self.pmd = TagBody.from_dict(self.ws.pmd) if self.ws.pmd else TagBody()
        self._pmd_obs: list = []
        self._pmd_fix = None
        self._pmd_T: np.ndarray | None = None
        if self.ws.pmd and self.ws.pmd.get("T_body2base") is not None:
            self._pmd_T = np.asarray(self.ws.pmd["T_body2base"], float)
        self._pmd_nodes: dict[str, Any] = {}
        self._pmd_key = None
        with g.add_folder("Kamera PMD (tagi na obudowie)", expand_by_default=False):
            g.add_markdown("Kamera glebi (pmd CamBoard nano) z tagami 36h11 naklejonymi na obudowie: kamery "
                           "stanowiska, ktore maja poze, widza tagi - stad miejsce PMD w 3D.")
            self.pmd_ids = g.add_text("id tagow", " ".join(str(i) for i in self.pmd.ids),
                                      hint="Numery AprilTag 36h11 na obudowie, po spacji. PIERWSZY to uklad kamery PMD "
                                           "(srodek w srodku tego taga, z = jego normalna). Karta kalibracyjna ma 0 i 1: "
                                           "te same numery na obudowie myla fale, gdy PMD jest w kadrze.")
            self.pmd_mm = g.add_number("Bok tagu na obudowie [mm]", round(self.pmd.tag_size * 1000, 1), min=5.0,
                                       max=100.0, step=0.5,
                                       hint="Bok CZARNEGO kwadratu taga na obudowie, zmierzony linijka. Zmiana "
                                            "przeskalowuje nauczony uklad.")
            self.pmd_optical = g.add_dropdown("Kamera patrzy wzdluz", self._pmd_axis_options(self.pmd.ids),
                                              initial_value=self._pmd_axis_label(self.pmd.optical),
                                              hint="Os optyczna PMD wzgledem jednego z tagow: '-z taga 0' = tag 0 na "
                                                   "tylnej sciance (kamera patrzy od niego); '+x/-x' = tag na bocznej; "
                                                   "'+y/-y' = tag na gornej. Rysuje piramide pola widzenia 90 x 69 st. "
                                                   "'nieznana' = same tagi.")
            learn = g.add_button("Ucz ukladu tagow: dodaj kadr", icon=viser.Icon.CAMERA_PLUS,
                                 hint="Biezacy kadr kazdej wlaczonej kamery z K (poza kamery niepotrzebna) doklada "
                                      "do nauki, gdzie na obudowie sa tagi wzgledem siebie. Kilka kadrow z roznych "
                                      "stron, az kazdy tag ma poze. Ramieniem nie rusza.")
            forget = g.add_button("Nauka od nowa", icon=viser.Icon.TRASH, color="gray",
                                  hint="Wyrzuca zebrane kadry i uklad tagow (np. po przeklejeniu taga).")
            find = g.add_button("Znajdz PMD w kamerach", icon=viser.Icon.TARGET,
                                hint="Kadry kamer z poza -> poza kamery PMD w ukladzie podstawy (mediana z kamer); "
                                     "tagi, osie i piramida w 3D. Ramieniem nie rusza.")
            self.pmd_follow = g.add_checkbox("Sledz na zywo", False,
                                             hint="Szuka PMD w kazdym takcie panelu i przesuwa ja w 3D - do "
                                                  "ustawiania kamery na stanowisku.")
            save = g.add_button("Zapisz do stanowiska", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                hint="Numery, bok, uklad tagow, os optyczna i ostatnia znaleziona poza do "
                                     "twin.json (sekcja pmd) - po restarcie PMD od razu jest w 3D.")
            self.pmd_md = g.add_markdown("")
        if self._pmd_T is not None:
            self._pmd_draw(self._pmd_T)
            self._pmd_note("poza z pliku stanowiska - Znajdz PMD w kamerach odswieza")

        @learn.on_click
        @self._safe
        def _(event):
            self._pmd_learn()

        @forget.on_click
        @self._safe
        def _(event):
            self._pmd_forget()

        @find.on_click
        @self._safe
        def _(event):
            self._pmd_find(self._grab())

        @save.on_click
        @self._safe
        def _(event):
            self._pmd_save(event)

        @self.pmd_optical.on_update
        @self._safe
        def _(event):
            if event.client is None:
                return                                  # ustawione z kodu (zmiana numerow)
            self._pmd_sync()
            if self._pmd_T is not None:
                self._pmd_draw(self._pmd_T)

    @staticmethod
    def _pmd_axis_options(ids) -> tuple[str, ...]:
        return ("nieznana",) + tuple(f"{axis} taga {i}" for i in ids for axis in OPTICAL_AXES)

    @staticmethod
    def _pmd_axis_label(optical) -> str:
        return "nieznana" if optical is None else f"{optical[1]} taga {int(optical[0])}"

    def _pmd_sync(self) -> TagBody:
        """Bryla PMD z pol panelu. Nowe numery zeruja uklad i nauke; nowy bok skaluje uklad."""
        try:
            ids = tuple(int(t) for t in self.pmd_ids.value.replace(",", " ").split())
        except ValueError:
            raise ValueError(f"id tagow: liczby po spacji, jest {self.pmd_ids.value!r}") from None
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("id tagow: co najmniej jeden numer, bez powtorzen")
        size = float(self.pmd_mm.value) / 1000.0
        body = self.pmd
        if ids != body.ids:
            body = TagBody(body.name, ids, size, {}, None, body.fov_deg, body.resolution)
            self._pmd_obs, self._pmd_fix = [], None
            self.pmd_optical.options = self._pmd_axis_options(ids)
            self.pmd_optical.value = "nieznana"
        elif abs(size - body.tag_size) > 1e-9:
            k = size / body.tag_size
            layout = {i: T.copy() for i, T in body.layout.items()}
            for T in layout.values():
                T[:3, 3] *= k
            body = TagBody(body.name, ids, size, layout, body.optical, body.fov_deg, body.resolution)
            self._pmd_obs = []                          # pozy tagow w kadrach liczone starym bokiem
        label = self.pmd_optical.value
        body.optical = None if label == "nieznana" else (int(label.split()[2]), label.split()[0])
        self.pmd = body
        return body

    def _pmd_note(self, text: str) -> None:
        """Tekst pod przyciskami, z ostrzezeniem o numerach wspolnych z karta kalibracyjna."""
        shared = sorted(set(self.pmd.ids) & set(self.ws.card_obj().ids))
        if shared:
            text += (f"\n\n**Uwaga:** tagi {', '.join(map(str, shared))} sa tez na karcie kalibracyjnej - PMD "
                     f"w kadrze podczas fali myli kalibracje; bezpieczniej inne numery na obudowie.")
        self.pmd_md.content = text

    def _pmd_learn(self) -> None:
        body = self._pmd_sync()
        obs = observe(self._grab(), self.ws.cameras, body)
        if not obs:
            with_k = [c.name for c in self.ws.cameras if c.enabled and c.K is not None]
            self._pmd_note(f"zadna kamera nie widzi tagow {', '.join(map(str, body.ids))} "
                           f"(wlaczone z K: {', '.join(with_k) or 'brak - najpierw Kalibracja, krok 1'})")
            return
        self._pmd_obs.extend(obs)
        fit = learn_layout(self._pmd_obs, body)
        body.layout = fit.layout
        seen = "; ".join(f"{o.camera}: tagi {', '.join(map(str, o.ids))}" for o in obs)
        pairs = ", ".join(f"{a}-{b} x{n}" for (a, b), n in sorted(fit.pairs.items()))
        missing = fit.missing(body.ids)
        text = f"kadr {len(self._pmd_obs)}: {seen}. Pary tagow w jednym kadrze: {pairs or 'brak'}. "
        if missing:
            text += (f"**Bez pozy: tag {', '.join(map(str, missing))}** - pokaz go kamerze razem z tagiem, "
                     f"ktory juz ma poze.")
        else:
            text += "Uklad kompletny - Znajdz PMD w kamerach."
        self._pmd_note(text)
        if self._pmd_T is not None:
            self._pmd_draw(self._pmd_T)

    def _pmd_forget(self) -> None:
        body = self._pmd_sync()
        body.layout = {body.ref: np.eye(4)}
        self._pmd_obs, self._pmd_fix = [], None
        self._pmd_note("nauka ukladu od nowa - dodaj kadry, w ktorych widac po dwa tagi naraz")
        if self._pmd_T is not None:
            self._pmd_draw(self._pmd_T)

    def _pmd_find(self, frames: dict[str, np.ndarray]) -> None:
        """Poza PMD z kadrow kamer z poza -> wezly 3D i tekst z rozrzutem miedzy kamerami."""
        body = self._pmd_sync()
        fix = locate(frames, self.ws.cameras, body)
        self._pmd_fix = fix
        if fix is None:
            posed = [c.name for c in self.ws.cameras if c.enabled and c.T_cam2base is not None and c.K is not None]
            self._pmd_note(f"tagi {', '.join(map(str, body.known()))} nie widziane (kamery z poza: "
                           f"{', '.join(posed) or 'brak - najpierw Kalibracja'})")
            return
        self._pmd_T = fix.T_body2base
        self._pmd_draw(fix.T_body2base)
        self._pmd_pose_to_record(fix)
        p = fix.position
        cams = "; ".join(f"{n}: tagi {', '.join(map(str, fix.tags_seen[n]))}, {r:.2f} px"
                         for n, r in fix.reproj_px.items())
        text = (f"PMD (tag {body.ref}): x {p[0] * 1000:.0f} mm, y {p[1] * 1000:.0f} mm, z {p[2] * 1000:.0f} mm; "
                f"{cams}; rozrzut miedzy kamerami {fix.spread_mm:.1f} mm")
        untrusted = [n for n in fix.per_camera if not self.ws.camera(n).trusted]
        if untrusted:
            text += (f". Kamery bez zaufanej pozy: {', '.join(untrusted)} - poza PMD jest tylko tak dobra, "
                     f"jak ich kalibracja.")
        self._pmd_note(text)

    def _pmd_records(self) -> list[CameraRecord]:
        return [c for c in self.ws.cameras if is_pmd(c.source)]

    def _pmd_pose_to_record(self, fix) -> None:
        """Poza z tagow -> `T_cam2base` kamery stanowiska o zrodle `pmd` (gdy wybrano os optyczna).

        Poza jest NIEZAUFANA z zalozenia: uklad optyczny zaczyna sie w srodku wybranego taga,
        nie w obiektywie (kilka-kilkanascie mm), a wchodzi przez kalibracje kamer stanowiska.
        Do sceny i piramidy w 3D wystarcza; mapa i percepcja biora tylko kamery zaufane.
        """
        To = self.pmd.T_optical2body()
        recs = self._pmd_records()
        if To is None or not recs:
            return
        T = np.asarray(fix.T_body2base, float) @ To
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for rec in recs:
            rec.T_cam2base = T.tolist()
            rec.calibration = dict(rms_px=float(max(fix.reproj_px.values())), max_px=float(max(fix.reproj_px.values())),
                                   n_obs=len(fix.per_camera), spread_deg=0.0, trusted=False,
                                   reason="poza z tagow na obudowie (srodek w tagu, nie w obiektywie)", time=now,
                                   spread_mm=float(fix.spread_mm), tags=[int(self.pmd.optical[0]), self.pmd.optical[1]])
            self._update_frustum(rec)
        self._dirty_rebuild = time.monotonic()          # kamera w scenie MuJoCo - przebudowa po chwili spokoju

    def _tick_pmd(self, frames: dict[str, np.ndarray]) -> None:
        if self.pmd_follow.value:
            self._pmd_find(frames)

    def _pmd_draw(self, T_body2base: np.ndarray) -> None:
        """Tagi (obrazy), osie ukladu PMD i piramida pola widzenia pod /pmd w scenie 3D.

        Wezly powstaja raz na zestaw (tagi, os, bok) i potem tylko sie przesuwaja - w trybie
        "sledz na zywo" nie ma co slac obrazow tagow co takt.
        """
        body = self.pmd
        Tw = self.T_b2w @ np.asarray(T_body2base, float)
        has_record = any(c.enabled and c.T_cam2base is not None for c in self._pmd_records())
        key = (tuple(body.known()), body.optical, round(body.tag_size, 6), has_record)
        with self.scene_lock:
            sc_ = self.server.scene
            if self._pmd_key != key:
                for h in self._pmd_nodes.values():
                    h.remove()
                self._pmd_nodes = {}
                self._pmd_key = key
            plate = body.tag_size / aptags.MARKER_FRACTION
            poses: dict[str, np.ndarray] = {"os": Tw, "etykieta": Tw}
            for i in body.known():
                # Plytka 0,5 mm przed srodkiem taga, zeby nie migotala z obudowa / plytka sasiada.
                poses[f"tag{i}"] = Tw @ body.layout[i] @ pose(np.eye(3), np.array([0.0, 0.0, 0.0005]))
            To = body.T_optical2body()
            own_frustum = not any(c.enabled and c.T_cam2base is not None for c in self._pmd_records())
            if To is not None and own_frustum:          # kamera pmd w stanowisku ma wlasna piramide z obrazem
                poses["kamera"] = Tw @ To
            for name, T in poses.items():
                h = self._pmd_nodes.get(name)
                if h is not None:
                    h.position, h.wxyz = T[:3, 3], mat_to_wxyz(T[:3, :3])
                    continue
                q = mat_to_wxyz(T[:3, :3])
                if name == "os":
                    h = sc_.add_frame("/pmd/os", axes_length=0.03, axes_radius=0.0015, wxyz=q, position=T[:3, 3])
                elif name == "etykieta":
                    h = sc_.add_label("/pmd/etykieta", "PMD", wxyz=q, position=T[:3, 3] + np.array([0.0, 0.0, 0.04]))
                elif name == "kamera":
                    h = sc_.add_camera_frustum("/pmd/kamera", fov=float(np.radians(body.fov_deg[1])),
                                               aspect=body.resolution[0] / body.resolution[1], scale=0.05,
                                               color=(120, 200, 255), thickness=0.002, wxyz=q, position=T[:3, 3])
                else:
                    # Wiersz 0 obrazu viser kladzie na -y (jak mapa blatu, `table[::-1]`); tag ma gore na +y.
                    img = aptags.render(int(name[3:]), 100)[::-1]
                    img = np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
                    h = sc_.add_image(f"/pmd/{name}", img, plate, plate, format="png", wxyz=q, position=T[:3, 3])
                self._pmd_nodes[name] = h

    def _pmd_save(self, event) -> None:
        body = self._pmd_sync()
        data = body.to_dict()
        if self._pmd_T is not None:
            data["T_body2base"] = np.asarray(self._pmd_T, float).tolist()
        self.ws.pmd = data
        self._save()
        self._notify(event, "Zapisano", f"kamera PMD w {self.ws_path.name}: tagi {', '.join(map(str, body.ids))}, "
                                        f"uklad {len(body.known())}/{len(body.ids)}, "
                                        f"poza {'zapisana' if self._pmd_T is not None else 'jeszcze nieznana'}")

    # ------------------------------------------------------------ tag na biurku
    def _find_tag(self) -> None:
        """Poza taga z kadrow zaufanych kamer -> zielona plytka w 3D i tekst z rozrzutem miedzy kamerami."""
        from ..tagreach import tag_from_frames
        frames = self._grab()
        fix = tag_from_frames(frames, self.ws.cameras, int(self.tag_id.value), self.tag_mm.value / 1000.0)
        self._tag_fix = fix
        if fix is None:
            trusted = [c.name for c in self.ws.cameras if c.enabled and c.T_cam2base is not None]
            self.tag_md.content = (f"tag {int(self.tag_id.value)} nie widziany (kamery z zaufana poza: "
                                   f"{', '.join(trusted) or 'brak - najpierw kalibracja'})")
            return
        self._show_tag(fix.T_tag2base)
        p = fix.position
        cams = ", ".join(f"{n} {r:.2f} px" for n, r in fix.reproj_px.items())
        self.tag_md.content = (f"tag {int(self.tag_id.value)}: x {p[0] * 1000:.0f} mm, y {p[1] * 1000:.0f} mm, "
                               f"z {p[2] * 1000:.0f} mm; kamery: {cams}; rozrzut miedzy kamerami "
                               f"{fix.spread_mm:.1f} mm")

    def _show_tag(self, T_tag2base: np.ndarray) -> None:
        """Plytka taga w scenie blizniaka (mocap, bez kolizji) w zmierzonej pozie."""
        half = self.tag_mm.value / 2000.0
        if not self._body_in_scene("tag"):
            objs = [o for o in self.twin.extras.get("objects", []) if o.name != "tag"]
            objs.append(sc.Box("tag", (half, half, 0.001), (0.0, 0.0), rgba=(0.15, 0.85, 0.35, 1.0),
                               free=False, mocap=True))
            self.twin.configure(**{**self.twin.extras, "objects": objs})
        with self.twin.lock:
            s = self.twin.scene
            k = int(s.model.body_mocapid[s.model.body("tag").id])
            T = s.T_base2world @ np.asarray(T_tag2base, float)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.ascontiguousarray(T[:3, :3]).ravel())
            s.data.mocap_pos[k] = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.001])
            s.data.mocap_quat[k] = q

    def _goto_tag(self) -> None:
        """IK do punktu nad tagiem i rampa przez nadzor - w watku, ramie jako wlasciciel 'tag'."""
        from ..tagreach import hover_point
        fix = self._tag_fix
        if fix is None:
            raise RuntimeError("najpierw Znajdz tag w kamerach")
        if not self.twin.connected:
            raise RuntimeError("ramie nie jest polaczone")
        if self.twin.owner is not None:
            raise RuntimeError(f"ramie zajete: {self.twin.owner} - rozbroj panel / zatrzymaj")
        if not self.twin.status.simulated and not self.pol_confirm.value:
            raise RuntimeError("prawdziwe ramie: zaznacz potwierdzenie w zakladce Polityki")
        p = hover_point(fix.T_tag2base, self.tag_above_cm.value / 100.0)
        cmd = {k: float(v) for k, v in (self.twin.status.command or self.twin.joints()).items()}
        sol = self._kin_gizmo.ik(p, seed=cmd)
        if not sol.ok:
            raise RuntimeError(f"punkt nad tagiem poza zasiegiem (blad {sol.pos_err * 1000:.0f} mm)")
        gripper = self.ws.spec().gripper
        target = {k: float(v) for k, v in sol.joints.items() if k != gripper}
        jump = max(abs(target[k] - cmd.get(k, target[k])) for k in target)
        duration = max(2.0, jump / 20.0)                     # najwyzej 20 st./s, nadzor i tak ogranicza
        self.pol_confirm.value = False
        self.tag_md.content = f"jade nad tag: {jump:.0f} st. najwiekszy skok, {duration:.1f} s"

        def run():
            try:
                self.twin.move(target, duration, owner="tag")
                self.tag_md.content = f"nad tagiem: {p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f} mm"
            except Exception as exc:
                self.tag_md.content = f"przerwane: {exc}"
            finally:
                try:
                    self.twin.release("tag")
                except Exception:
                    pass

        threading.Thread(target=run, name="tag-move", daemon=True).start()

    def _place_cube_randomly(self) -> None:
        from ..rl import task as tk
        task = tk.make_task("lift", self.ws.spec())
        if not self._cube_in_scene():
            self._configure_objects(cube=True, pad=self._body_in_scene("target"))
        pos, quat = tk.sample_cubes(task, np.random.default_rng(), 1)
        with self.twin.lock:
            s = self.twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a, dv = m.jnt_qposadr[m.body_jntadr[b]], m.jnt_dofadr[m.body_jntadr[b]]
            T = s.T_base2world
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            d.qpos[a:a + 3] = T[:3, :3] @ pos[0] + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            d.qvel[dv:dv + 6] = 0.0
            mujoco.mj_forward(m, d)

    def _sim_cube(self):
        with self.twin.lock:
            s = self.twin.scene
            b = s.model.body("cube").id
            Ti = inverse(s.T_base2world)
            return (Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]).copy(), (Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)).copy()

    def _vision_cube(self):
        """Kostka z kamer dla polityki - z dlonia, gdy szczeki ja zaslaniaja albo niosa.

        Do trackera idzie tylko detekcja NOWA (z kadrow pozniejszych niz poprzednio
        podana) i mlodsza niz `CUBE_MAX_AGE`. Stara detekcja podawana co takt jako
        swieza odnawiala `hold_s` w nieskonczonosc - zamrozona kamera albo wylaczona
        mapa i polityka jechala po kostke, ktorej juz tam nie bylo.
        """
        kin = self._kin_vision
        st = self.twin.status
        joints = dict(st.measured) or self.twin.joints()
        q = kin.to_q(joints)
        # Rozkaz chwytaka, ktory NAPRAWDE idzie do serwa (`status.command`), a nie `runner.q_cmd`:
        # runner startuje od zmierzonego kata szczeki, wiec po STOP/Dom z kostka w szczekach
        # tracker nie widzial "sciska" i nie bral kostki "w dloni" - polityka stawala z
        # "kamery jej nie widza" (verify2, s1).
        grip = kin.spec.gripper
        k = kin.spec.joints.index(grip)
        sent = (getattr(st, "command", None) or {}).get(grip)
        if sent is not None:
            cmd = kin.to_q({**joints, grip: float(sent)})[k]
        else:
            cmd = self.runner.q_cmd[k] if self.runner is not None else q[k]
        now = time.monotonic()
        det = self.last_cube
        if det is not None and (det.confidence <= 0.4 or not det.t or det.t <= self._cube_used_t
                                or now - det.t > CUBE_MAX_AGE):
            det = None
        if det is not None:
            self._cube_used_t = det.t
        return self.cube_tracker.update(det, kin.tcp(joints), q[k], cmd, self._grip_closed_q(), now)

    def _grip_closed_q(self) -> float:
        """Kat szczeki [rad] pustego, zamknietego chwytaka (chwytak 0) - `grip_closed` trackera.

        Ten sam, co w polityce (`Limits.lo` jej biegacza), a nie dolny kraniec MJCF:
        po przeliczeniu chwytaka przez tiki serwa zamknieta szczeka stoi na -5,4 st.,
        a MJCF ma -10 st. - warunek "szczeka stoi szerzej niz zamknieta" byl wtedy
        zawsze prawdziwy i pusta dlon mogla wyjsc na "w dloni".
        """
        runner = self.runner
        if runner is not None and getattr(runner, "limits", None) is not None:
            return float(runner.limits.lo[runner.limits.grip])
        kin = self._kin_vision
        return float(kin.to_q({kin.spec.gripper: 0.0})[kin.spec.joints.index(kin.spec.gripper)])

    def _start_policy(self, event) -> None:
        from ..rl.policy import Policy
        from ..rl.runner import PolicyRunner
        # Przed wszystkimi sprawdzeniami: STOP, Dom, Polacz albo Zatrzymaj od tej chwili (ladowanie
        # polityki trwa 100-150 ms) = polityka nie bierze ramienia (`_claim_kw`).
        mark = self._motion_mark()
        if not self.twin.connected:
            raise RuntimeError("polacz ramie w zakladce Ramie (sim, zeby bezpiecznie sprawdzic)")
        if not self.twin.status.simulated and not self.pol_confirm.value:
            raise RuntimeError("prawdziwe ramie: zaznacz potwierdzenie, ze sie ruszy")
        if self._estop_active():
            raise RuntimeError("aktywny STOP - Skasuj STOP (zolty, pod STOP-em)")
        busy = self._busy(POLICY_OWNER)
        if busy:
            raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj")
        if self.runner is not None:
            self.runner.stop("nowa polityka")
        pol = Policy.load(self._policy_path())
        provider = None
        if pol.task.target and not self._body_in_scene("target"):
            # Podkladka celu (tylko do ogladania); kostka zostaje, jesli juz byla w scenie.
            self._configure_objects(cube=self._cube_in_scene(), pad=True)
        if pol.task.cube:
            if self.pol_cube_src.value == "symulacja":
                if not self.twin.status.simulated:
                    raise RuntimeError("na prawdziwym ramieniu kostka musi pochodzic z kamer")
                if not self._cube_in_scene():
                    self._place_cube_randomly()
                provider = self._sim_cube
            else:
                if self.mapper is None:
                    raise RuntimeError("kostka z kamer wymaga co najmniej jednej skalibrowanej kamery (zakladka Mapa)")
                if not (self.map_on.value and self.cube_on.value):
                    raise RuntimeError("kostka z kamer wymaga wlaczonej mapy i szukania kostki (zakladka Mapa)")
                self.cube_tracker = CubeTracker.for_spec(self.ws.spec())
                self._cube_used_t = 0.0
                provider = self._vision_cube
        self._release_panel()
        self.policy = pol
        # Sukces lift na ramieniu liczy sie tylko ze swiezej pozy kostki (kamery / w dloni),
        # nigdy z "ostatnio widzianej" - runner pyta tracker o zrodlo co takt.
        source = (lambda: self.cube_tracker.source) if provider == self._vision_cube else None
        self.runner = PolicyRunner(self.twin, pol, cube_provider=provider, cube_source=source)
        self._runner_backend = str(self.twin.status.backend)  # przewodnik: sukces liczy sie TEMU backendowi
        if pol.task.name == "reach":
            p = (inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3]
            if not self.goal_node.visible:
                from ..rl import task as tk
                p = tk.sample_goals(self.twin.scene.kin, pol.task, np.random.default_rng(), 1)[0]
            self._set_goal(p)
            self.runner.episode_limit = False                # cel przeciagany na zywo - bez limitu epizodu
        elif pol.task.target:
            # pickplace: cel odlozenia z uchwytu celu (rzutowany na blat), a gdy go nie widac -
            # losowany jak w treningu, co najmniej `target_min_sep` od kostki.
            p = (inverse(self.T_b2w) @ np.r_[self.goal_gizmo.position, 1.0])[:3]
            if not self.goal_node.visible:
                from ..rl import task as tk
                cube = self._sim_cube()[0] if self._cube_in_scene() else np.array([0.2, 0.0, pol.task.cube_half])
                p = tk.sample_targets(pol.task, np.random.default_rng(), cube[None])[0]
            self._set_goal(p)
        self.goal_node.visible = self.goal_gizmo.visible = pol.task.name in ("reach", "pickplace")
        try:
            # Bierze ramie ("polityka") albo RuntimeError: zajete, aktywny STOP, odebrane od `mark`.
            self.runner.start(**self._claim_kw(mark))
        except RuntimeError as exc:
            self.runner.status.stopped_because = f"nie ruszyla: {exc}"
            raise
        self.pol_confirm.value = False                       # nastepne uruchomienie - nowe potwierdzenie

    def _tick_policies(self) -> None:
        if self.runner is not None:
            rs = self.runner.status
            txt = (f"{'**jedzie**' if rs.running else 'stoi'} - krok {rs.step}, {rs.hz:.0f} Hz")
            if self.runner.task.name == "reach" and np.isfinite(rs.distance):
                txt += f", do celu **{rs.distance * 1000:.0f} mm**" + (" (w celu)" if rs.success else "")
                if rs.goal_clamped:
                    txt += " - cel przyciety do obszaru z treningu"
            if self.runner.task.target and np.isfinite(rs.distance):
                # pickplace: odleglosc kostki od celu odlozenia (xy)
                txt += f", kostka do celu **{rs.distance * 1000:.0f} mm**" + (" (niesiona)" if rs.carried else "")
                if rs.goal_clamped:
                    txt += " - cel przesuniety na obszar z treningu"
            if self.runner.cube_provider == self._vision_cube:
                txt += f", kostka: **{self.cube_tracker.source}**"
            if self.runner.task.cube and self.twin.status.simulated and self._cube_in_scene():
                pos, _ = self._sim_cube()
                txt += f"  \nprawda symulacji: kostka **{100 * (pos[2] - self.runner.task.cube_half):.1f} cm** nad blatem"
            if not rs.running and rs.stopped_because:
                txt += f"  \nzatrzymana: {rs.stopped_because}"
            self.pol_run_md.content = txt
        j = self.eval_job
        if j.running:
            self.pol_eval_md.content = f"ewaluacja... {j.message}"
        elif j.state == jobs.DONE and j.data.get("shown") is not j.result:
            self._show_policy()
            lines = []
            for k, label in (("cpu_nominal", "bez randomizacji"), ("cpu_rand", "z randomizacja")):
                e = (j.result or {}).get(k)
                if e:
                    lines.append(f"- CPU {label}: **{e['success']:.0%}** z {e['episodes']} epizodow")
            self.pol_eval_md.content = "\n".join(lines)
            j.data["shown"] = j.result
        elif j.state == jobs.FAILED and j.data.get("shown") != "err":
            self.pol_eval_md.content = f"**Blad**: {j.error}"
            j.data["shown"] = "err"

    # =============================================================== sim-real
    def _build_simreal(self) -> None:
        g = self.server.gui
        self._tab_help("Sim-Real")
        g.add_markdown("Prawdziwy kadr i render blizniaka z TEJ SAMEJ kamery (skalibrowana poza, K, katy "
                       "zmierzone na serwach). Rozjazd krawedzi to blad kalibracji albo modelu - widac go od razu.")
        self.sr_cam = g.add_dropdown("Kamera", ("-",), initial_value="-",
                                     hint="Prawdziwa, skalibrowana kamera; render blizniaka z jej pozy i K.")
        self.sr_alpha = g.add_slider("Przezroczystosc renderu", 0.0, 1.0, 0.05, 0.5,
                                     hint="Udzial renderu w nakladce: 0 = sam kadr, 1 = sam render. Dziala przy "
                                          "odznaczonych krawedziach.")
        self.sr_edges = g.add_checkbox("Krawedzie symulacji zamiast renderu", True,
                                       hint="Zolte krawedzie renderu na kadrze i mediana ich odleglosci od krawedzi "
                                            "kadru [px]; do 3 px - kalibracja i model zgodne.")
        self.sr_img = g.add_image(np.zeros((240, 320, 3), np.uint8), format="jpeg", jpeg_quality=75)
        self.sr_md = g.add_markdown("")

    def _tick_simreal(self, frames: dict[str, np.ndarray]) -> None:
        name = self.sr_cam.value
        try:
            rec = self.ws.camera(name)
        except KeyError:
            self.sr_md.content = "Wybierz kamere."
            return
        if rec.simulated:
            self.sr_md.content = ("Kamera symulowana - jej porownanie z prawda (mm, st.) jest w zakladce Kamery.")
            return
        real = frames.get(name)
        if real is None or not rec.calibrated:
            self.sr_md.content = "Potrzebny kadr i skalibrowana poza kamery."
            return
        K, dist = rec.intrinsics()
        if dist is not None and np.any(dist):
            # Render blizniaka to kamera otworkowa (samo K) - kadr musi byc bez dystorsji,
            # inaczej przy k1 = -0,25 krawedzie przy brzegu rozjezdzaly sie o dziesiatki px
            # przy dobrej kalibracji, a wynik ponizej kazal ja powtarzac.
            real = cv2.undistort(real, K, dist)
        sim = self.twin.render(name)
        if sim.shape != real.shape:
            sim = cv2.resize(sim, (real.shape[1], real.shape[0]))
        a = float(self.sr_alpha.value)
        if self.sr_edges.value:
            e_sim = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 60, 160) > 0
            out = real.copy()
            out[e_sim] = (0.3 * out[e_sim] + 0.7 * np.array([255, 196, 0])).astype(np.uint8)
            e_real = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 60, 160)
            dt = cv2.distanceTransform(255 - e_real, cv2.DIST_L2, 3)
            score = float(np.median(dt[e_sim])) if e_sim.any() else float("nan")
            # Przewodnik, krok 4: najlepsza mediana od ostatniej zmiany pozy/stolu/K (zatrzask -
            # jedna dobra klatka zalicza, szum wokol 3 px nie przelacza kroku). NaN (render bez
            # krawedzi) tylko, gdy nie ma zadnej liczby.
            prev = self._simreal_px.get(name)
            if np.isfinite(score):
                if prev is None or not np.isfinite(prev) or score < prev:
                    self._simreal_px[name] = score
            elif prev is None:
                self._simreal_px[name] = score
            self.sr_md.content = f"mediana odleglosci krawedzi symulacji od krawedzi kadru: **{score:.1f} px**"
        else:
            out = cv2.addWeighted(real, 1 - a, sim, a, 0)
            self.sr_md.content = ""
        self.sr_img.image = thumb(out, 640)

    # ================================================================== petla
    def _grab(self) -> dict[str, np.ndarray]:
        """Kadry wlaczonych kamer z chwilami ich wykonania (`frame_times`).

        Kadr starszy niz `FRAME_MAX_AGE` jest pomijany: strumien, ktory stanal
        (zerwane przekazanie USB), oddawal te sama klatke w nieskonczonosc,
        a mapa co takt "widziala" w niej kostke od nowa.
        """
        frames, times = {}, {}
        now = time.monotonic()
        for c in self.ws.cameras:
            if not c.enabled:
                continue
            try:
                img, t = self.twin.cameras.frame_t(c.name)
            except Exception:
                img, t = None, 0.0
            if img is None or (t and now - t > FRAME_MAX_AGE):
                continue
            frames[c.name] = img
            times[c.name] = t or now
        with self.frame_lock:
            self.frames, self.frame_times = frames, times
            self._t_grab = now
        return frames

    def stream_frames(self) -> dict[str, tuple[np.ndarray, float]]:
        """Kadry z ostatniego `_grab` z chwilami wykonania - dla `FrameServer` (inny watek, tylko odczyt).
        Nie `cameras.frame_t`: symulowane renderuja sie w MuJoCo, a to nie jest bezpieczne z innego watku."""
        with self.frame_lock:
            return {n: (img, self.frame_times.get(n, 0.0)) for n, img in self.frames.items()}

    def _fresh_frames(self, max_age: float) -> dict[str, np.ndarray]:
        """Kadry z ostatniego `_grab`, jesli sa mlodsze niz `max_age` [s], inaczej nowe."""
        with self.frame_lock:
            if time.monotonic() - self._t_grab < max_age:
                return dict(self.frames)
        return self._grab()

    def _heal_cameras(self, now: float | None = None) -> None:
        """Kamera, ktora sie nie otworzyla albo ktorej strumien padl, otwiera sie sama: `sync` co
        `HEAL_S`, poki ktoras wlaczona prawdziwa kamera zglasza blad. Dotad trzeba bylo ja wylaczyc
        i wlaczyc w panelu - 2026-09-26: Brio i uEye przegrywaly wyscig z resetem USB przy starcie
        panelu (zamkniecie uchwytu w starym procesie resetuje kamere), uEye po restarcie tez."""
        now = time.monotonic() if now is None else now
        if now - self._heal_t < HEAL_S:
            return
        hub = self.twin.cameras
        if any(hub.error(c.name) for c in self.ws.cameras if c.enabled and not c.simulated):
            self._heal_t = now
            hub.sync()

    def _tick_slow(self, frames: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        status = self._status_text()
        self.status_md.content = status
        set_if(self.status_md, "visible", bool(status))
        with self._render_lock:
            self._render_chip(*self._arm_mode())        # odswieza tez Hz na plakietce
        opts = tuple(c.name for c in self.ws.cameras) or ("-",)
        for dd in (self.sr_cam,):
            if dd.options != opts:
                cur = dd.value
                dd.options = opts
                dd.value = cur if cur in opts else opts[0]
        self._heal_cameras()
        if frames is None:
            frames = self._grab()
        self._render_cameras()
        rec = self._cam()
        if rec is not None:
            # Wybrana kamera bez kadru (nie otworzyla sie, wylaczona, strumien stanal) - napis,
            # a nie ostatni kadr POPRZEDNIEJ kamery pod nazwa nowej.
            img = frames.get(rec.name)
            self._img(self.cam_preview, "podglad", thumb(img, 480) if img is not None else no_frame_image())
        else:
            # Usunieta ostatnia kamera: podglad zostawal z jej ostatnim kadrem, jakby wciaz byla.
            self._img(self.cam_preview, "podglad", no_frame_image("brak kamer"))
        with self.scene_lock:
            for name, img in frames.items():
                h = self.frustums.get(name)
                if h is not None and h.visible:
                    self._img(h, f"piramida/{name}/{id(h)}", thumb(img, 160))
        self._tick_calibration()
        self._tick_training()
        self._tick_policies()
        self._guard("pmd", self._tick_pmd, frames)
        # Po zakladkach: wynik fali / identyfikacji do zapisu jest juz widoczny w tym takcie.
        self._guard("przewodnik", self._tick_guide)
        self._guard("spectacles", self.specs.tick)
        return frames

    def _tick_watch(self, frames: dict[str, np.ndarray]) -> None:
        changed = False
        for c in self.ws.cameras:
            if c.name not in frames or c.name not in self.watch.refs:
                continue
            was = self.watch.moved(c.name)
            with self.frame_lock:
                t = self.frame_times.get(c.name)
            mask = (self._arm_masks([c.name], t) or {}).get(c.name)
            self.watch.check(c.name, frames[c.name], mask)
            if self.watch.moved(c.name) != was:
                changed = True
                if not was:
                    self._notify(None, "Kamera przestawiona",
                                 f"{c.name}: kadr przesunal sie o {_px(self.watch.shift[c.name])} od kalibracji. "
                                 f"Kalibracja > pomaranczowy przycisk tej kamery.", error=True)
        if changed:
            self._refresh_cameras()

    def _guard(self, what: str, fn, *args):
        """Blad jednego odswiezenia nie moze zabic petli panelu - pod nim jezdzi ramie."""
        try:
            return fn(*args)
        except Exception:
            if what not in self._tick_errors:                # jeden wpis na rodzaj, nie co takt
                self._tick_errors.add(what)
                logger.exception("Blad w odswiezaniu panelu (%s) - petla dziala dalej", what)
            return None

    def _sync_mirror(self) -> None:
        """Widok 3D za scena: przebudowa po `twin.rebuild`, potem tylko pozy cial.

        Wersja i model czytane RAZEM pod blokada: przebudowa w trakcie budowy widoku
        (~0,6 s) zapisywala nowa wersje przy siatkach ze starego modelu - widok zostawal
        na starym modelu na zawsze, a `update` czytal ciala nowego modelu po starych numerach.
        """
        with self.twin.lock:
            v, model = self.twin.version, self.twin.scene.model
        if v != self._mirror_version:
            if self.mirror is not None:
                self.mirror.remove()
            self.mirror = SceneMirror(self.server, model)
            self._mirror_version = v
        with self.twin.lock:
            if self.twin.version == self._mirror_version:
                self.mirror.update(self.twin.scene.data)

    def _record_joints(self, now: float) -> None:
        """Historia katow do masek ramienia w chwili kadru (`_arm_masks`)."""
        st = self.twin.status
        if st.connected and st.measured:
            self._joint_hist.append((now, dict(st.measured)))

    def run(self) -> None:
        port = self.server.get_port()
        remote = "  (zdalnie: http://<adres-maszyny>:{port})" if self.host not in ("127.0.0.1", "localhost") else ""
        print(f"Panel blizniaka: http://localhost:{port}{remote.format(port=port)}", flush=True)
        if self.stream is not None:
            print(f"Podglad kamer:  {self.stream.url}  (z sieci: http://<adres-maszyny>:{self.stream.port}/)",
                  flush=True)
        t_slow = t_map = t_watch = t_list = 0.0
        frames: dict[str, np.ndarray] = {}
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                self._guard("scena", self._sync_mirror)
                self._record_joints(t0)
                self._guard("ramie", self._tick_arm)
                vision_policy = self._vision_policy_running()
                if t0 - t_slow > 0.2:
                    t_slow = t0
                    # Kadry osobno od reszty odswiezania: blad w zakladce Trening nie moze
                    # zostawic mapy z kadrami sprzed minuty (wczesniej `... or frames`).
                    frames = self._guard("kadry", self._fresh_frames, 0.1) or {}
                    self._guard("odswiezanie", self._tick_slow, frames)
                if t0 - t_map > (0.12 if vision_policy else 0.33):
                    t_map = t0
                    if vision_policy:                          # polityka z kamer: kadry swieze na te mape
                        frames = self._guard("kadry", self._fresh_frames, 0.05) or {}
                    self._guard("mapa", self._tick_map, frames)
                    self._guard("sim-real", self._tick_simreal, frames)
                if t0 - t_watch > 2.0:
                    t_watch = t0
                    self._guard("straznik kamer", self._tick_watch, frames)
                if t0 - t_list > 5.0:
                    t_list = t0
                    if self.train.exit_code is not None and self.train.run_dir is not None:
                        self._guard("lista polityk", self._refresh_policies)
                if self._dirty_save and t0 - self._dirty_save > 1.0:
                    self._dirty_save = 0.0
                    self._guard("zapis", self._save)
                if self._dirty_rebuild and t0 - self._dirty_rebuild > 1.0:
                    self._dirty_rebuild = 0.0
                    self._guard("przebudowa", self.twin.rebuild)
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Najpierw ramie (polityka, zadania, rozlaczenie), dopiero potem trening i serwer.

        Wczesniej `train.shutdown` (do 8 s na zapis polityki + 5 s na zabicie) szlo
        PRZED rozlaczeniem ramienia, a drugie Ctrl+C w tym czasie pomijalo
        `twin.close()` - proces konczyl sie bez czystego rozlaczenia serw.
        """
        self._stop.set()
        try:
            if self.runner is not None:
                self.runner.stop("zamkniecie panelu")
            specs = getattr(self, "specs", None)           # release the arm before disconnecting
            if specs is not None:
                self._guard("spectacles", specs.stop)
            for job in (self.calib_job, self.sysid_job, self.intr_job, self.eval_job):
                job.stop()
        finally:
            try:
                self.twin.close()
            finally:
                try:
                    # Trening w osobnym procesie bez konsoli - bez tego zostawal po zamknieciu panelu.
                    self._guard("trening", self.train.shutdown)
                finally:
                    if self.stream is not None:
                        self._guard("podglad kamer", self.stream.stop)
                    self.server.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="lerobot-twin ui", description="Panel cyfrowego blizniaka w przegladarce.")
    # Panel steruje ramieniem i nie ma hasla: z sieci tylko na zyczenie (--host 0.0.0.0).
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--stream-port", type=int, default=None)
    ap.add_argument("--stream-host", default="0.0.0.0")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Sondy HTTP (HEAD, niedokonczone polaczenia) nie sa bledami panelu - websockets
    # drukuje po kilkadziesiat linii stosu na kazda, zasypujac to, co wazne.
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    TwinApp(a.workspace, a.host, a.port, stream_port=a.stream_port, stream_host=a.stream_host).run()
    return 0
