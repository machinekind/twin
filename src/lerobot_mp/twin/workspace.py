"""Stanowisko blizniaka: wszystko, co wiadomo o biurku, w jednym pliku JSON.

Ramie i jego port, stol i to, gdzie stoi na nim podstawa, karta kalibracyjna
(ze ZMIERZONYM bokiem taga) i kazda kamera z intrynsykami, poza wzgledem
podstawy i wynikiem kalibracji. Z tego jednego opisu powstaje scena MuJoCo,
srodowisko RL i widok w UI - wiec kamera skalibrowana na biurku trafia do
symulacji dokladnie tam, gdzie stoi, bez przepisywania liczb recznie.

Domyslnie `workspace/twin.json` wzgledem korzenia klonu repozytorium (poza
klonem: wzgledem katalogu uruchomienia, patrz `lerobot_mp.paths`); katalog jest
w `.gitignore`, bo stanowisko jest per biurko, nie per repozytorium. Przyklad
w symulacji: `examples/twin.sim.json` (`lerobot-twin demo`).

Polozenie kamer (krok 2) jest za drogie, zeby je nadpisywac: kazdy zapis to
osobny plik `workspace/calib/<data-godzina>.json` ze wszystkimi kamerami, ktore
maja poze (`save_preset`), nigdy nie nadpisywany. `twin.json` trzyma kopie
robocza wgranego zestawu (`calib_preset`); bez wskazania wgrywany jest
najnowszy (`load`). Starszy zestaw wraca przez `load_preset`.

A package that runs on this station without importing the twin (training) gets
it as data: `export` (`lerobot-twin export <folder>`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import data_path
from .calib.card import Card
from .robots import RobotSpec, default_robot, get_spec
from .scene import CameraView, SceneConfig, Table

#: Domyslny plik stanowiska. W klonie repozytorium liczony od jego korzenia,
#: zeby panel odpalony z innego katalogu nie zaczynal od pustego stanowiska.
DEFAULT_REL = Path("workspace") / "twin.json"
DEFAULT_PATH = data_path(DEFAULT_REL)


def default_path() -> Path:
    """Domyslny plik stanowiska liczony w chwili uzycia (katalog biezacy moze sie zmienic)."""
    return data_path(DEFAULT_REL)

#: Poziome pole widzenia typowej kamerki internetowej - tylko do nominalnych
#: intrynsyk, zanim kamera zobaczy szachownice.
NOMINAL_HFOV_DEG = 65.0


def nominal_K(width: int, height: int, hfov_deg: float = NOMINAL_HFOV_DEG) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def same_intrinsics(a: tuple[Any, Any], b: tuple[Any, Any]) -> bool:
    """Te same intrynsyki (K, dist)? Brak dystorsji (None) = same zera; JSON gubi tylko ostatnie bity."""
    Ka, Kb = np.asarray(a[0], float), np.asarray(b[0], float)
    if Ka.shape != Kb.shape or not np.allclose(Ka, Kb, rtol=1e-9, atol=1e-6):
        return False
    da = np.zeros(0) if a[1] is None else np.asarray(a[1], float).ravel()
    db = np.zeros(0) if b[1] is None else np.asarray(b[1], float).ravel()
    n = max(len(da), len(db))
    da, db = np.pad(da, (0, n - len(da))), np.pad(db, (0, n - len(db)))
    return bool(np.allclose(da, db, rtol=1e-9, atol=1e-9))


@dataclass
class CameraRecord:
    name: str
    #: Indeks kamery ("0", "1"), plik wideo, adres strumienia albo "sim".
    source: str = "0"
    width: int = 640
    height: int = 480
    fps: int = 30
    K: list[list[float]] | None = None
    dist: list[float] | None = None
    T_cam2base: list[list[float]] | None = None
    #: Skad sa intrynsyki: "szachownica", "nominalne", "symulacja".
    intrinsics_from: str = "nominalne"
    #: Wynik ostatniej kalibracji polozenia: rms_px, spread_deg, n_obs, trusted, reason, time.
    calibration: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    #: Tylko kamery symulowane: gdzie kamera NAPRAWDE stoi w blizniaku (ustawiana
    #: w panelu). Kalibracja jej nie zna - wyznacza `T_cam2base` z kadrow, tak
    #: jak dla prawdziwej kamery - wiec wynik da sie ocenic wzgledem prawdy.
    sim_pose: list[list[float]] | None = None
    #: Wynik kalibracji intrynsyk: rms_px, n_views, coverage, trusted, reason, time.
    intrinsics_info: dict[str, Any] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.T_cam2base is not None

    @property
    def simulated(self) -> bool:
        return self.source == "sim"

    def true_pose(self) -> np.ndarray | None:
        """Poza, z ktorej kamera widzi scene: prawda dla symulowanej, kalibracja dla prawdziwej."""
        T = self.sim_pose if self.simulated and self.sim_pose is not None else self.T_cam2base
        return None if T is None else np.asarray(T, float)

    @property
    def trusted(self) -> bool:
        """Poza zaufana - i wciaz liczona z TYM K, ktore kamera ma teraz.

        Zapisany werdykt nie wystarcza: K zmienione po zapisie pozy (reczna edycja
        pliku, inna sciezka niz krok 1 w panelu) zostawialo poze z nominalnego K
        jako zaufana. Poza bez zapisanego K (sprzed zapisu K przy pozie) - werdykt jak byl.
        """
        if not self.calibration.get("trusted", False):
            return False
        used = self.calibration.get("K")
        return used is None or same_intrinsics((used, self.calibration.get("dist")), self.intrinsics())

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray | None]:
        K = np.asarray(self.K, float) if self.K is not None else nominal_K(self.width, self.height)
        dist = None if self.dist is None else np.asarray(self.dist, float)
        return K, dist

    def intrinsics_problem(self) -> str:
        """Dlaczego K tej kamery nie wystarcza do ZAUFANEJ pozy ("" = wystarcza).

        Symulowana zna swoje prawdziwe K. Prawdziwa - tylko z zaufanej sesji ChArUco.
        """
        if self.simulated:
            return ""
        if self.intrinsics_from != "szachownica" or self.K is None:
            return "intrynsyki nominalne - najpierw krok 1 (tablica ChArUco)"
        if not self.intrinsics_info.get("trusted", False):
            why = self.intrinsics_info.get("reason", "")
            return "intrynsyki z niezaufanej sesji ChArUco" + (f" ({why})" if why else "") + " - powtorz krok 1"
        return ""

    def fit_problem(self, used: tuple[Any, Any] | None) -> str:
        """Dlaczego poze liczona z intrynsykami `used` = (K, dist) nie mozna uznac za zaufana.

        "" = mozna. Ocena dotyczy K, z ktorym LICZONO poze, a nie K w rekordzie w chwili
        zapisu: fala z nominalnym K (fx 502), potem krok 1 zapisal zaufane K z ChArUco
        (fx 552, k1 -0,2) i "Zapisz" oznaczalo poze z nominalnego K jako zaufana.
        Gdy `used` rozni sie od obecnego K, werdykt "intrynsyki zmienione od fali";
        gdy jest takie samo - opis obecnego K (`intrinsics_problem`) opisuje wlasnie je.
        `used` None = nie wiadomo, z jakim K liczono - ocena obecnego K (stare wywolania).
        """
        if used is not None and not same_intrinsics(used, self.intrinsics()):
            K_used = np.asarray(used[0], float)
            K_now, _ = self.intrinsics()
            what = (f"fx {K_used[0, 0]:.0f} -> {K_now[0, 0]:.0f}" if abs(K_used[0, 0] - K_now[0, 0]) > 0.5
                    else "K albo dystorsja")
            return f"intrynsyki zmienione od fali ({what}) - uruchom fale ponownie"
        return self.intrinsics_problem()

    def view(self) -> CameraView | None:
        """Kamera jako czesc sceny - tylko, gdy wiadomo, gdzie stoi.

        Kamera symulowana renderuje z prawdziwej pozy (`sim_pose`) i z prawdziwym
        K - jej "prawdziwe" intrynsyki to te zapisane w `K`, a nominalne sa tylko
        dla kamer, ktore jeszcze nie widzialy szachownicy.
        """
        T = self.true_pose()
        if T is None:
            return None
        K, _ = self.intrinsics()
        return CameraView(self.name, K, self.width, self.height, T)


@dataclass
class Workspace:
    #: Ramie z zainstalowanego pakietu (`robots.robot_folders`); nowe stanowisko bierze
    #: pierwsze zarejestrowane (`robots.default_robot`).
    robot: str = field(default_factory=default_robot)
    #: "sim" - blizniak bez sprzetu, inaczej wtyczka backendu z pakietu ramienia
    #: (np. "a1x" - Galaxea A1X po CAN).
    backend: str = "sim"
    #: Poza domowa tego stanowiska [jednostki aplikacji] ("Ustaw dom = tu" w panelu);
    #: None = poza domowa z opisu ramienia (`RobotSpec.home`).
    home: dict[str, float] | None = None
    port: str | None = None
    table: dict[str, Any] = field(default_factory=lambda: asdict(Table()))
    card: dict[str, Any] = field(default_factory=lambda: asdict(Card()))
    cameras: list[CameraRecord] = field(default_factory=list)
    #: Dynamika serw zidentyfikowana na tym ramieniu (`dynamics.Dynamics`);
    #: pusta = model Menagerie. `export` daje ja treningowi w `station.json`.
    dynamics: dict[str, Any] = field(default_factory=dict)
    #: Wgrany zestaw polozen kamer - nazwa pliku z `workspace/calib/` bez `.json`;
    #: None = nie wybrano, `load` wgrywa najnowszy.
    calib_preset: str | None = None
    path: Path | None = field(default=None, compare=False, repr=False)

    # --------------------------------------------------------------- obiekty
    def spec(self) -> RobotSpec:
        return get_spec(self.robot)

    def card_obj(self) -> Card:
        data = dict(self.card)
        data["ids"] = tuple(data.get("ids", (0, 1)))
        return Card(**data)

    def table_obj(self) -> Table:
        data = dict(self.table)
        for key in ("size", "base_xy", "rgba"):
            if key in data:
                data[key] = tuple(data[key])
        return Table(**data)

    def camera(self, name: str) -> CameraRecord:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"brak kamery {name!r} w stanowisku")

    def add_camera(self, record: CameraRecord) -> CameraRecord:
        if any(c.name == record.name for c in self.cameras):
            raise ValueError(f"kamera {record.name!r} juz jest w stanowisku")
        self.cameras.append(record)
        return record

    def remove_camera(self, name: str) -> None:
        self.cameras = [c for c in self.cameras if c.name != name]

    def free_name(self, stem: str = "kamera") -> str:
        taken = {c.name for c in self.cameras}
        k = 1
        while f"{stem}{k}" in taken:
            k += 1
        return f"{stem}{k}"

    def scene_config(self, *, with_card: bool = False, **extra: Any) -> SceneConfig:
        """Scena z tego stanowiska: stol, ramie i kazda WLACZONA, umiejscowiona kamera."""
        views = [v for c in self.cameras if c.enabled for v in [c.view()] if v is not None]
        cfg = SceneConfig(self.spec(), table=self.table_obj(), cameras=views,
                          card=self.card_obj() if with_card else None)
        for key, value in extra.items():
            setattr(cfg, key, value)
        return cfg

    def export(self, folder: str | Path) -> Path:
        """Writes the station into `folder` as data for packages that do not import the twin (training).

        `scene.xml` and `assets/` (`scene.export`): the floor, table, lights, the arm and every
        enabled camera that has a pose, with its full K (MJCF holds no distortion); a simulated
        camera stands at its true pose. `station.json`: this workspace as `save` writes it to
        `twin.json` (the table with the base pose, every camera with its K, distortion and
        calibration, the identified dynamics, home). Returns the folder.
        """
        from .scene import export

        folder = Path(folder)
        export(self.scene_config(), folder)
        (folder / "station.json").write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                                             encoding="utf-8")
        return folder

    def apply_fit(self, fit, tag_size: float | None = None,
                  intrinsics: dict[str, tuple[Any, Any]] | None = None) -> list[str]:
        """Wpisuje wynik kalibracji do kamer. Zwraca nazwy kamer, ktore dostaly poze.

        Poza jest zapisywana takze dla kamery niezaufanej - z powodem obok -
        bo lepiej widziec w UI, jak bardzo sie rozjechala, niz nie widziec nic.

        `tag_size` - bok taga, z ktorym LICZONO dopasowanie (fala czyta go na
        starcie); bez niego wpisywany byl bok z pola w panelu w chwili zapisu,
        a ten mogl juz byc inny niz ten, z ktorym policzono poze.
        `intrinsics` - {kamera: (K, dist)}, z ktorymi liczono; trafiaja do
        `calibration`, zeby pozniejsza zmiana K byla widoczna jako niezgodnosc
        (`CameraRecord.trusted`). K inne niz obecne w kamerze (krok 1 zrobiony
        w trakcie fali albo po niej) = poza niezaufana (`CameraRecord.fit_problem`).

        Prawdziwa kamera jest zaufana TYLKO z K z szachownicy, ktora sama byla
        zaufana. Z nominalnym K (65 st. pola widzenia) albo z niezaufanej
        sesji ChArUco poza wychodzi pewna siebie i przesunieta: zmierzone
        w `calib.simulate` z K o 10% za duzym - residuum 0,90 px (przechodzi
        prog), a kamera 72,7 mm od prawdziwego miejsca.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        size = float(tag_size) if tag_size is not None else float(self.card_obj().tag_size)
        updated = []
        for name, cam_fit in fit.cameras.items():
            try:
                rec = self.camera(name)
            except KeyError:
                continue
            trusted, reason = bool(cam_fit.trusted), cam_fit.reason
            if intrinsics is None:
                k_problem = rec.intrinsics_problem()
            elif name not in intrinsics:
                # Nie wiadomo, z jakim K liczono - niezaufana, a nie ocena obecnego K.
                k_problem = "brak intrynsyk z fali - uruchom fale ponownie"
            else:
                k_problem = rec.fit_problem(intrinsics[name])
            if trusted and k_problem:
                trusted, reason = False, k_problem
            rec.T_cam2base = np.asarray(cam_fit.T_cam2base, float).tolist()
            rec.calibration = dict(rms_px=float(cam_fit.rms_px), max_px=float(cam_fit.max_px),
                                   n_obs=int(cam_fit.n_obs), spread_deg=float(cam_fit.spread_deg),
                                   trusted=trusted, reason=reason, time=now, tag_size=size)
            if intrinsics and name in intrinsics:
                K, dist = intrinsics[name]
                rec.calibration["K"] = np.asarray(K, float).tolist()
                rec.calibration["dist"] = None if dist is None else np.asarray(dist, float).ravel().tolist()
            updated.append(name)
        return updated

    # ------------------------------------------------------- zestawy polozen
    def presets_dir(self) -> Path:
        return Path(self.path or default_path()).parent / "calib"

    def presets(self) -> list[str]:
        """Nazwy zapisanych zestawow polozen kamer, od najnowszego."""
        d = self.presets_dir()
        return sorted((p.stem for p in d.glob("*.json")), reverse=True) if d.is_dir() else []

    def save_preset(self) -> Path:
        """Polozenie wszystkich kamer z poza (T_cam2base + werdykt) do NOWEGO pliku; staje sie wgranym zestawem.

        Nazwa z chwili ostatniej kalibracji (`20260926-145216`); ten sam wynik zapisany drugi raz
        trafia do tego samego pliku, inny wynik z tej samej sekundy dostaje przyrostek `-2`.
        Istniejacy plik nigdy nie jest nadpisywany inna trescia.
        """
        cams = {c.name: {"T_cam2base": c.T_cam2base, "calibration": c.calibration}
                for c in self.cameras if c.calibrated}
        if not cams:
            raise ValueError("zadna kamera nie ma polozenia - nie ma czego zapisac")
        times = [c["calibration"].get("time") for c in cams.values()]
        stamp = max(t for t in times if t) if any(times) else time.strftime("%Y-%m-%dT%H:%M:%S")
        name = stamp.replace("-", "").replace(":", "").replace("T", "-")
        body = json.dumps({"time": stamp, "tag_size": self.card.get("tag_size"), "cameras": cams},
                          indent=2, ensure_ascii=False)
        d = self.presets_dir()
        d.mkdir(parents=True, exist_ok=True)
        path, k = d / f"{name}.json", 2
        while path.exists() and path.read_text(encoding="utf-8") != body:
            path, k = d / f"{name}-{k}.json", k + 1
        path.write_text(body, encoding="utf-8")
        self.calib_preset = path.stem
        return path

    def load_preset(self, name: str) -> list[str]:
        """Wgrywa zapisany zestaw do kamer o tych samych nazwach. Zwraca nazwy kamer, ktore dostaly poze.

        Kamery spoza zestawu zostaja, jak byly. Werdykt "zaufana" liczy sie na nowo wzgledem
        OBECNEGO K kamery (`CameraRecord.trusted`) - zestaw sprzed zmiany obiektywu wraca jako niezaufany.
        """
        data = json.loads((self.presets_dir() / f"{name}.json").read_text(encoding="utf-8"))
        done = []
        for cam_name, entry in data["cameras"].items():
            try:
                rec = self.camera(cam_name)
            except KeyError:
                continue
            rec.T_cam2base, rec.calibration = entry["T_cam2base"], dict(entry["calibration"])
            done.append(cam_name)
        self.calib_preset = name
        return done

    # ------------------------------------------------------------------ dysk
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("path", None)
        return data

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else (self.path or default_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = target
        if self.calib_preset is None and any(c.calibrated for c in self.cameras):
            self.save_preset()                          # polozenie sprzed zestawow - pierwszy zestaw
        tmp = target.with_suffix(".tmp")
        # Najpierw do pliku obok, potem podmiana - przerwany zapis nie zniszczy
        # kalibracji, na ktora ktos poswiecil pol godziny.
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> Workspace:
        """Wczytuje stanowisko; brak pliku to nowe, puste stanowisko (nie blad)."""
        target = Path(path) if path else default_path()
        if not target.is_file():
            ws = cls()
            ws.path = target
            return ws
        data = json.loads(target.read_text(encoding="utf-8"))
        cams = [CameraRecord(**c) for c in data.pop("cameras", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in ("cameras", "path")}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"nieznane pola w {target}: {sorted(unknown)}")
        ws = cls(cameras=cams, **data)
        ws.path = target
        if ws.calib_preset is None:
            newest = ws.presets()
            if newest:
                ws.load_preset(newest[0])
        return ws
