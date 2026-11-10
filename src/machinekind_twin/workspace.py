"""Stanowisko blizniaka: wszystko, co wiadomo o biurku, w jednym pliku JSON.

Ramie i jego port, stol i to, gdzie stoi na nim podstawa, karta kalibracyjna
(ze ZMIERZONYM bokiem taga) i kazda kamera z intrynsykami, poza wzgledem
podstawy i wynikiem kalibracji. Z tego jednego opisu powstaje scena MuJoCo,
srodowisko RL i widok w UI - wiec kamera skalibrowana na biurku trafia do
symulacji dokladnie tam, gdzie stoi, bez przepisywania liczb recznie.

Domyslnie `workspace/twin.json` wzgledem korzenia klonu repozytorium (poza
klonem: wzgledem katalogu uruchomienia, patrz `machinekind_twin.paths`); katalog jest
w `.gitignore`, bo stanowisko jest per biurko, nie per repozytorium. Przyklad
w symulacji: `examples/twin.sim.json` (`twin demo`).

Polozenie kamer (krok 2) jest za drogie, zeby je nadpisywac: kazdy zapis to
osobny plik `workspace/calib/<data-godzina>.json` ze wszystkimi kamerami, ktore
maja poze (`save_preset`), nigdy nie nadpisywany. `twin.json` trzyma kopie
robocza wgranego zestawu (`calib_preset`); bez wskazania wgrywany jest
najnowszy (`load`). Starszy zestaw wraca przez `load_preset`.

A package that runs on this station without importing the twin (training) gets
it as data: `export` (`twin export <folder>`).

A workspace loaded from a stand folder (`stand`, `twin stand load`) stays bound to it: its pose
presets are the stand's and every `save` writes back to the stand (`stand.sync`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .calib.card import Card
from .calib.session import WaveLimits
from .camera_controls import CameraControls
from .config import from_mapping
from .paths import data_path
from .robots import RobotSpec, default_robot, get_spec
from .scene import PREFIX, CameraView, Look, PassiveRobot, SceneConfig, Table

#: Domyslny plik stanowiska. W klonie repozytorium liczony od jego korzenia,
#: zeby panel odpalony z innego katalogu nie zaczynal od pustego stanowiska.
DEFAULT_REL = Path("workspace") / "twin.json"
DEFAULT_PATH = data_path(DEFAULT_REL)


def default_path() -> Path:
    """Domyslny plik stanowiska liczony w chwili uzycia (katalog biezacy moze sie zmienic)."""
    return data_path(DEFAULT_REL)


#: (robot, offsets) -> (the description they were added to, the description with them).
_OFFSET_SPECS: dict[tuple[str, tuple[tuple[str, float], ...]], tuple[RobotSpec, RobotSpec]] = {}


def _spec_with_offsets(robot: str, offsets: tuple[tuple[str, float], ...]) -> RobotSpec:
    """`get_spec(robot)` with `offsets`: one object per description and offsets, a new one when either changes."""
    base = get_spec(robot)
    if not offsets:
        return base
    hit = _OFFSET_SPECS.get((robot, offsets))
    if hit is None or hit[0] is not base:
        hit = _OFFSET_SPECS[(robot, offsets)] = (base, replace(base, joint_offsets=dict(offsets)))
    return hit[1]

#: Poziome pole widzenia typowej kamerki internetowej - tylko do nominalnych
#: intrynsyk, zanim kamera zobaczy szachownice.
NOMINAL_HFOV_DEG = 65.0
#: Default tolerance [px] of a camera on a wobbly mount (`CameraRecord.wobble_px`).
WOBBLE_PX = 4.0


def nominal_K(width: int, height: int, hfov_deg: float = NOMINAL_HFOV_DEG) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def stamp_name(stamp: str) -> str:
    """File name of a dated entry: 2026-09-26T14:52:16 -> 20260926-145216."""
    return stamp.replace("-", "").replace(":", "").replace("T", "-")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def when_text(stamp: str) -> str:
    """A stored time for the panel: 2026-11-10T17:27:03 -> "10.11 17:27", with the year when it is not this one."""
    try:
        t = time.strptime(str(stamp)[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return str(stamp)
    year = "" if t.tm_year == time.localtime().tm_year else f".{t.tm_year}"
    return f"{t.tm_mday:02d}.{t.tm_mon:02d}{year} {t.tm_hour:02d}:{t.tm_min:02d}"


def px_text(value: Any) -> str:
    """Pixels the Polish way: 1.79 -> "1,8 px", 0.42 -> "0,42 px"."""
    v = float(value)
    return (f"{v:.2f}" if v < 1 else f"{v:.1f}").replace(".", ",") + " px"


def by_hand(info: dict[str, Any]) -> bool:
    """`calibration` or `intrinsics_info` is trusted because a person said so (`set_by_hand`)."""
    manual = info.get("manual")
    return isinstance(manual, dict) and bool(manual.get("trusted")) and bool(info.get("trusted"))


def set_by_hand(info: dict[str, Any], trusted: bool, text: str, now: str | None = None) -> None:
    """Trusts or distrusts `calibration` / `intrinsics_info` by hand; the measured numbers stay.

    `manual` keeps the verdict from before the first decision by hand: the reason says what the person
    did, when, and what the verdict was. Going back to that verdict drops `manual` and restores it.
    """
    manual = info.get("manual")
    if isinstance(manual, dict):
        was, was_trusted = str(manual.get("was", "")), bool(manual.get("was_trusted", False))
    else:
        was, was_trusted = str(info.get("reason", "") or ""), bool(info.get("trusted", False))
    if trusted == was_trusted:
        info.pop("manual", None)
        info["trusted"], info["reason"] = was_trusted, was
        return
    now = now or _now()
    info["trusted"] = trusted
    info["reason"] = f"{text} {when_text(now)}" + (f" (wczesniej: {was})" if was else "")
    info["manual"] = {"trusted": trusted, "time": now, "was": was, "was_trusted": was_trusted}


def clear_trust(info: dict[str, Any], reason: str) -> None:
    """Trust taken away by a rule (new lens, servo offsets, moved camera); a trust by hand says it is gone."""
    if by_hand(info):
        reason = f"reczne zaufanie cofniete: {reason}"
    info.pop("manual", None)
    info["trusted"], info["reason"] = False, reason


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
    #: The mount sways when the arm moves (Kamery > Chwiejny statyw): the move watch judges the camera
    #: only with the arm still, and a lasting shift is a notice that leaves the pose's trust alone.
    wobbly: bool = False
    #: How far [px] a wobbly camera's picture may wander with the arm still.
    wobble_px: float = WOBBLE_PX
    #: Tylko kamery symulowane: gdzie kamera NAPRAWDE stoi w blizniaku (ustawiana
    #: w panelu). Kalibracja jej nie zna - wyznacza `T_cam2base` z kadrow, tak
    #: jak dla prawdziwej kamery - wiec wynik da sie ocenic wzgledem prawdy.
    sim_pose: list[list[float]] | None = None
    #: Wynik kalibracji intrynsyk: rms_px, n_views, coverage, trusted, reason, time; `focus` when the
    #: lens was measured at a locked focus.
    intrinsics_info: dict[str, Any] = field(default_factory=dict)
    #: Focus and exposure the twin sets whenever it opens this camera (`camera_controls`).
    controls: CameraControls = field(default_factory=CameraControls)
    #: Why the lens does not hold in this session: its focus is not locked on the camera ("" = it is,
    #: or the lens carries no focus). Set by the camera hub when it opens the camera; never saved.
    focus_problem: str = field(default="", compare=False, repr=False)

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
        if not self.calibration.get("trusted", False) or self.focus_problem:
            return False
        used = self.calibration.get("K")
        return used is None or same_intrinsics((used, self.calibration.get("dist")), self.intrinsics())

    @property
    def lens_focus(self) -> int | None:
        """The focus the lens was measured at; None for a lens measured without a locked focus."""
        focus = self.intrinsics_info.get("focus")
        return None if focus is None else int(focus)

    def applied_controls(self) -> CameraControls:
        """The controls the twin sets on open: `controls`, with the lens's focus where they set no focus."""
        focus = self.lens_focus
        if focus is None or self.controls.focus_absolute is not None or self.controls.autofocus:
            return self.controls
        return replace(self.controls, autofocus=False, focus_absolute=focus)

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray | None]:
        K = np.asarray(self.K, float) if self.K is not None else nominal_K(self.width, self.height)
        dist = None if self.dist is None else np.asarray(self.dist, float)
        return K, dist

    @property
    def lens_measured(self) -> bool:
        """K comes from a ChArUco session or a lens file, not the nominal field of view."""
        return self.intrinsics_from == "szachownica" and self.K is not None

    @property
    def lens_by_hand(self) -> bool:
        return by_hand(self.intrinsics_info)

    @property
    def pose_by_hand(self) -> bool:
        return by_hand(self.calibration)

    @property
    def lens_trusted(self) -> bool:
        """The stored lens verdict is "trusted" (focus in this session aside)."""
        return bool(self.intrinsics_info.get("trusted")) and (self.lens_measured or self.lens_by_hand)

    def trust_lens(self, now: str | None = None) -> None:
        """Trusts the lens by hand. A nominal lens gets its K written down, so it travels with the camera."""
        if self.simulated:
            raise ValueError("kamera symulowana zna swoj obiektyw")
        if self.lens_trusted:
            return
        if self.lens_measured:
            set_by_hand(self.intrinsics_info, True, "zaufane recznie", now)
            return
        K, _ = self.intrinsics()
        self.K, self.dist, self.intrinsics_from = K.tolist(), None, "nominalne"
        self.intrinsics_info = {}
        set_by_hand(self.intrinsics_info, True, "nominalny, zaufany recznie", now)

    def distrust_lens(self, now: str | None = None) -> None:
        """Withdraws the lens's trust by hand; a nominal lens trusted by hand goes back to no K at all.
        A trusted pose stands on the lens, so it is withdrawn too."""
        if not self.lens_trusted:
            return
        if not self.lens_measured and self.lens_by_hand:
            self.K, self.dist, self.intrinsics_info = None, None, {}
        else:
            set_by_hand(self.intrinsics_info, False, "zaufanie cofniete recznie", now)
        if self.calibrated and self.calibration.get("trusted"):
            self.withdraw_pose("zaufanie do obiektywu cofniete")

    def pose_trust_problem(self) -> str:
        """Why the pose cannot be trusted by hand now ("" = it can)."""
        if not self.calibrated:
            return "brak polozenia - najpierw krok 2"
        used = self.calibration.get("K")
        if used is not None and not same_intrinsics((used, self.calibration.get("dist")), self.intrinsics()):
            return "polozenie liczone z innym obiektywem - zrob krok 2 jeszcze raz"
        problem = self.intrinsics_problem()
        return f"najpierw obiektyw: {problem}" if problem else ""

    def trust_pose(self, now: str | None = None) -> None:
        """Trusts the pose by hand; refused while the lens it stands on is not trusted (`pose_trust_problem`)."""
        if self.calibration.get("trusted"):
            return
        problem = self.pose_trust_problem()
        if problem:
            raise ValueError(problem)
        set_by_hand(self.calibration, True, "zaufane recznie", now)

    def distrust_pose(self, now: str | None = None) -> None:
        """Withdraws the pose's trust by hand; after a trust by hand the measured verdict comes back."""
        if self.calibration.get("trusted"):
            set_by_hand(self.calibration, False, "zaufanie cofniete recznie", now)

    def withdraw_pose(self, reason: str) -> None:
        """The pose stops being trusted for `reason`; a trust by hand is withdrawn too."""
        clear_trust(self.calibration, reason)

    def pose_problem(self) -> str:
        """Why the pose is not trusted right now ("" = it is, or there is none)."""
        if not self.calibrated or self.trusted:
            return ""
        if not self.calibration.get("trusted"):
            return str(self.calibration.get("reason", "") or "niepewne")
        if self.focus_problem:
            return f"obiektyw niewazny w tej sesji: {self.focus_problem}"
        return "obiektyw zmierzony po ustaleniu polozenia - zrob krok 2 jeszcze raz"

    def lens_quality(self) -> str:
        """The lens in plain words: residual and when it was measured, or that it is nominal."""
        if self.simulated:
            return "symulowany"
        if not self.lens_measured:
            return "nominalny (bez pomiaru)"
        ii = self.intrinsics_info
        manual = ii.get("manual")
        base = str(manual.get("was", "")) if isinstance(manual, dict) else str(ii.get("reason", "") or "")
        from_file = base.startswith("z pliku ")
        parts = [base] if from_file else []
        if ii.get("rms_px"):
            parts.append(f"residuum {px_text(ii['rms_px'])}")
        if ii.get("time"):
            parts.append(f"{'wczytane' if from_file else 'zmierzone'} {when_text(ii['time'])}")
        return ", ".join(parts) or "zmierzony"

    def pose_quality(self) -> str:
        """The pose in plain words: residual and when it was measured."""
        c = self.calibration
        if not self.calibrated:
            return "brak polozenia"
        source = str(c.get("source", "") or "")
        parts = [source] if source else []
        if c.get("rms_px") is not None and not source:
            parts.append(f"residuum {px_text(c['rms_px'])}")
        if c.get("time"):
            parts.append(f"{'wpisane' if source else 'zmierzone'} {when_text(c['time'])}")
        return ", ".join(parts) or "bez opisu pomiaru"

    def intrinsics_problem(self) -> str:
        """Why this camera's K is not enough for a TRUSTED pose ("" = it is).

        A simulated camera knows its true K. A real one needs a trusted ChArUco session or lens file, or
        a lens trusted by hand (`trust_lens`). A focus that does not hold in this session wins over both.
        """
        if self.simulated:
            return ""
        if not self.lens_measured and not self.lens_by_hand:
            return "intrynsyki nominalne - najpierw krok 1 (tablica ChArUco)"
        if not self.intrinsics_info.get("trusted", False):
            why = self.intrinsics_info.get("reason", "")
            return "intrynsyki z niezaufanej sesji ChArUco" + (f" ({why})" if why else "") + " - powtorz krok 1"
        if self.focus_problem:
            return f"obiektyw niewazny w tej sesji: {self.focus_problem}"
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
class RobotRecord:
    """One robot of the station besides the driven arm, which the twin only watches.

    `robot`, `backend` and `port` mean what they mean for the driven arm (`Workspace`). The backend
    is opened read-only (`BackendContext.read_only`); without one the robot keeps `joints`.
    """

    #: Its name in the scene (`<name>/<body>`) and in the panel.
    name: str
    #: Robot description from an installed package (`robots.robot_folders`).
    robot: str
    #: Base position relative to the driven arm's base [m]: x forward, y left, z up.
    xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Base rotation about the vertical axis relative to the driven arm's base [deg].
    yaw_deg: float = 0.0
    #: Backend of the robot's package, opened read-only; None = the fixed pose `joints`.
    backend: str | None = None
    port: str | None = None
    #: Joint values [app units]: the fixed pose, or the last pose read from `backend`.
    #: Joints not given take the robot's home pose.
    joints: dict[str, float] = field(default_factory=dict)

    def T_base(self) -> np.ndarray:
        """Base pose in the driven arm's base frame (4x4)."""
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:3, 3] = [float(v) for v in self.xyz]
        return T

    def placed(self) -> PassiveRobot:
        """This robot as part of the scene."""
        return PassiveRobot(self.name, get_spec(self.robot), self.T_base(), dict(self.joints))


@dataclass
class Workspace:
    #: Ramie z zainstalowanego pakietu (`robots.robot_folders`); nowe stanowisko bierze
    #: pierwsze zarejestrowane (`robots.default_robot`).
    robot: str = field(default_factory=default_robot)
    #: "sim" - blizniak bez sprzetu, inaczej wtyczka backendu z pakietu ramienia.
    backend: str = "sim"
    #: Poza domowa tego stanowiska [jednostki aplikacji] ("Ustaw dom = tu" w panelu);
    #: None = poza domowa z opisu ramienia (`RobotSpec.home`).
    home: dict[str, float] | None = None
    port: str | None = None
    table: dict[str, Any] = field(default_factory=lambda: asdict(Table()))
    #: Lights and headlight of the scene; the default is the twin's own look.
    look: Look = field(default_factory=Look)
    card: dict[str, Any] = field(default_factory=lambda: asdict(Card()))
    cameras: list[CameraRecord] = field(default_factory=list)
    #: Dynamika serw zidentyfikowana na tym ramieniu (`dynamics.Dynamics`);
    #: pusta = model Menagerie. `export` daje ja treningowi w `station.json`.
    dynamics: dict[str, Any] = field(default_factory=dict)
    #: Wgrany zestaw polozen kamer - nazwa pliku z `workspace/calib/` bez `.json`;
    #: None = nie wybrano, `load` wgrywa najnowszy.
    calib_preset: str | None = None
    #: Zero offsets of this arm's servos [app units] from calibration step 2 (`RobotSpec.joint_offsets`).
    #: The camera poses were fitted with them, so they come and go together (`apply_fit`, presets).
    joint_offsets: dict[str, float] = field(default_factory=dict)
    #: What the calibration wave must keep out of on this stand (`calib.session.WaveLimits`): keep-out
    #: boxes in the base frame [m] and the base's turn range [degrees]. Empty = only the arm's own limits.
    wave: WaveLimits = field(default_factory=WaveLimits)
    #: Other robots of the station, which the twin only watches (`RobotRecord`). Empty = the driven
    #: arm alone; such a workspace saves without this key, as before.
    other_robots: list[RobotRecord] = field(default_factory=list)
    #: The stand folder this workspace was loaded from (absolute path); None = no stand, as before.
    stand: str | None = None
    path: Path | None = field(default=None, compare=False, repr=False)

    # --------------------------------------------------------------- obiekty
    def spec(self) -> RobotSpec:
        """The arm's description with this workstation's servo zero offsets. The same object while the
        offsets stay the same: the arm mask rebuilds its kinematics when the description changes."""
        return _spec_with_offsets(self.robot, tuple(sorted((k, float(v)) for k, v in self.joint_offsets.items())))

    def reset_joint_offsets(self, reason: str) -> list[str]:
        """Back to the servos' own zeros. Camera poses fitted with the offsets stop being trusted (`reason`);
        returns their names."""
        self.joint_offsets = {}
        return self._distrust_poses(reason)

    def _distrust_poses(self, reason: str, keep: tuple[str, ...] = ()) -> list[str]:
        names = []
        for rec in self.cameras:
            if rec.calibrated and rec.name not in keep and rec.calibration.get("trusted"):
                rec.withdraw_pose(reason)
                names.append(rec.name)
        return names

    def robots(self) -> list[RobotRecord]:
        """Every robot of the station: the driven arm first (at the origin), then `other_robots`."""
        return [RobotRecord(PREFIX.rstrip("/"), self.robot, backend=self.backend, port=self.port),
                *self.other_robots]

    def other_robot(self, name: str) -> RobotRecord:
        for rec in self.other_robots:
            if rec.name == name:
                return rec
        raise KeyError(f"brak robota {name!r} w stanowisku")

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
        self._refuse_robot_camera_names([record])
        self.cameras.append(record)
        return record

    def _refuse_robot_camera_names(self, records: list[CameraRecord]) -> None:
        """A real camera may carry the name of a camera on the arm: with a real arm the hub serves it
        in that camera's place. A simulated one may not: the scene renders both by name."""
        on_arm = self.spec().cameras
        clash = sorted(r.name for r in records if r.simulated and r.name in on_arm)
        if clash:
            raise ValueError(f"kamery symulowane {clash} nazywaja sie jak kamery na ramieniu {self.robot} - "
                             "zmien ich nazwy")

    def remove_camera(self, name: str) -> None:
        self.cameras = [c for c in self.cameras if c.name != name]

    def free_name(self, stem: str = "kamera") -> str:
        taken = {c.name for c in self.cameras}
        k = 1
        while f"{stem}{k}" in taken:
            k += 1
        return f"{stem}{k}"

    def scene_config(self, *, with_card: bool = False, **extra: Any) -> SceneConfig:
        """The scene of this workspace: table, arm and every ENABLED camera with a known pose.

        A real camera named like a camera on the arm rides on the arm, never fixed to the base, so
        it gets no view whatever `T_cam2base` it has (from a wave or a preset matched by name).
        """
        self._refuse_robot_camera_names(self.cameras)
        on_arm = self.spec().cameras
        fixed = [c for c in self.cameras if c.enabled and c.name not in on_arm]
        views = [v for c in fixed for v in [c.view()] if v is not None]
        cfg = SceneConfig(self.spec(), table=self.table_obj(), cameras=views,
                          card=self.card_obj() if with_card else None, look=self.look,
                          passive=[r.placed() for r in self.other_robots])
        for key, value in extra.items():
            setattr(cfg, key, value)
        return cfg

    def export(self, folder: str | Path) -> Path:
        """Writes the station into `folder` as data for packages that do not import the twin (training).

        `scene.xml` and `assets/` (`scene.export`): the floor, table, lights, the arm and every
        enabled camera that has a pose, with its full K (MJCF holds no distortion); a simulated
        camera stands at its true pose. `station.json`: this workspace as `save` writes it to
        `twin.json` (the table with the base pose, the scene's look, every camera with its K,
        distortion and calibration, the identified dynamics, home). Returns the folder.
        """
        from .scene import export

        folder = Path(folder)
        export(self.scene_config(), folder)
        data = self.to_dict()
        data.pop("stand", None)                         # a path on this machine, not station data
        (folder / "station.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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
        # The poses above were fitted with these offsets on top of the old ones; any other camera's pose
        # was fitted with the old ones and no longer matches the arm.
        delta = {n: float(v) for n, v in (getattr(fit, "joint_offsets", None) or {}).items() if v}
        if delta:
            for n, v in delta.items():
                self.joint_offsets[n] = round(self.joint_offsets.get(n, 0.0) + v, 3)
            self._distrust_poses("poprawki zer serw zmienione po tym pomiarze - zrob krok 2 jeszcze raz",
                                 keep=tuple(updated))
        return updated

    # ------------------------------------------------------- zestawy polozen
    def presets_dir(self) -> Path:
        """`calib/` next to the workspace file; for a workspace bound to a stand, the stand's pose history."""
        if self.stand is not None:
            return Path(self.stand) / "calib" / "poses"
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
        entry = self.preset_entry()
        if not entry["cameras"]:
            raise ValueError("zadna kamera nie ma polozenia - nie ma czego zapisac")
        name = stamp_name(entry["time"])
        body = json.dumps(entry, indent=2, ensure_ascii=False)
        d = self.presets_dir()
        d.mkdir(parents=True, exist_ok=True)
        path, k = d / f"{name}.json", 2
        while path.exists() and path.read_text(encoding="utf-8") != body:
            path, k = d / f"{name}-{k}.json", k + 1
        path.write_text(body, encoding="utf-8")
        self.calib_preset = path.stem
        return path

    def preset_entry(self) -> dict[str, Any]:
        """What a pose preset holds: every camera with a pose, the tag size and the servo zero offsets.

        `time` is the newest calibration time (now, when no camera has one).
        """
        cams = {c.name: {"T_cam2base": c.T_cam2base, "calibration": c.calibration}
                for c in self.cameras if c.calibrated}
        times = [c["calibration"].get("time") for c in cams.values()]
        stamp = max(t for t in times if t) if any(times) else time.strftime("%Y-%m-%dT%H:%M:%S")
        # The servo zero offsets the poses were fitted with go along; a set without any keeps its old form.
        offsets = {"joint_offsets": dict(self.joint_offsets)} if self.joint_offsets else {}
        return {"time": stamp, "tag_size": self.card.get("tag_size"), "cameras": cams, **offsets}

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
        # The set's poses hold only with the servo zero offsets they were fitted with (none in an older set).
        offsets = {k: float(v) for k, v in data.get("joint_offsets", {}).items()}
        if offsets != self.joint_offsets:
            self.joint_offsets = offsets
            self._distrust_poses(f"poprawki zer serw z zestawu {name} - zrob krok 2 dla tej kamery", keep=tuple(done))
        self.calib_preset = name
        return done

    # ------------------------------------------------------------------ dysk
    def to_dict(self) -> dict[str, Any]:
        """The workspace as `save` writes it: JSON types only (the look's tuples come out as lists)."""
        data = asdict(self)
        data.pop("path", None)
        if not data["other_robots"]:
            data.pop("other_robots")
        if data["stand"] is None:
            data.pop("stand")
        for cam, rec in zip(data["cameras"], self.cameras):
            cam.pop("focus_problem", None)
            if rec.controls.empty:
                cam.pop("controls")
            else:
                cam["controls"] = rec.controls.to_dict()
            # Written only when set: a steady camera's record stays as it was.
            if not rec.wobbly:
                cam.pop("wobbly")
            if rec.wobble_px == WOBBLE_PX:
                cam.pop("wobble_px")
        return json.loads(json.dumps(data))

    def save(self, path: str | Path | None = None) -> Path:
        """Writes the workspace file. A workspace bound to a stand writes back to it first; when that
        fails, the workspace file is still written and the error is raised after it."""
        target = Path(path) if path else (self.path or default_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = target
        problem = None
        if self.stand is not None:
            # Imported here: the stand module imports this one.
            from .stand import sync

            try:
                sync(self)
            except Exception as exc:                    # noqa: BLE001 - reported after the file is safe
                problem = exc
        elif self.calib_preset is None and any(c.calibrated for c in self.cameras):
            self.save_preset()                          # polozenie sprzed zestawow - pierwszy zestaw
        tmp = target.with_suffix(".tmp")
        # Najpierw do pliku obok, potem podmiana - przerwany zapis nie zniszczy
        # kalibracji, na ktora ktos poswiecil pol godziny.
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        if problem is not None:
            raise RuntimeError(f"stanowisko {self.stand} nie zapisane: {problem}") from problem
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
        cams = [camera_record(c, f"{target}: cameras") for c in data.pop("cameras", [])]
        others = [from_mapping(RobotRecord, r, f"{target}: other_robots")
                  for r in data.pop("other_robots", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in ("cameras", "other_robots", "path")}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"nieznane pola w {target}: {sorted(unknown)}")
        if "look" in data:
            data["look"] = from_mapping(Look, data["look"], f"{target}: look")
        if "wave" in data:
            data["wave"] = from_mapping(WaveLimits, data["wave"], f"{target}: wave")
        ws = cls(cameras=cams, other_robots=others, **data)
        ws.path = target
        if ws.calib_preset is None and ws.stand is None:    # a stand names its poses itself
            newest = ws.presets()
            if newest:
                ws.load_preset(newest[0])
        return ws


def camera_record(data: dict[str, Any], where: str) -> CameraRecord:
    """A camera from its JSON form; `controls` as a strict mapping (`CameraControls`)."""
    data = dict(data)
    controls = data.pop("controls", None) or {}
    try:
        return CameraRecord(**data, controls=from_mapping(CameraControls, controls, f"{where}.controls"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} {data.get('name', '?')!r}: {exc}") from exc
