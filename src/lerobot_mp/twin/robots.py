"""Rejestr ramion: jeden opis na model, wspolny dla symulacji, UI i kalibracji.

Opis mowi tylko to, czego nie da sie wyczytac z samego MJCF: ktore stawy sa
ramieniem w kolejnosci aplikacji, ktory z nich jest chwytakiem, gdzie jest
punkt narzedzia (TCP) i na jakim ciele jedzie karta kalibracyjna. Reszta -
zakresy, masy, siatki - siedzi w MJCF i jest czytana z niego, a nie powielana.

Jednostki aplikacji sa jednostkami LeRobota: stawy w stopniach, chwytak w skali
0..100. MJCF liczy w radianach (staw obrotowy) albo metrach (suwak), a chwytak
to w nim kat albo wysuw szczeki. Przeliczenie jest w `kinematics.RobotKinematics`
i tylko tam - dla chwytaka liniowo przez `RobotSpec.gripper_range_q`, gdy opis
go podaje, a inaczej przez te same tiki serwa
(`RobotConfig.gripper_closed_ticks..gripper_open_ticks`), co backend `feetech`,
zeby 0..100 znaczylo ten sam kat szczeki w blizniaku i na ramieniu.

Tu sa tez dane ramienia, ktorych MJCF nie zna: limity i predkosci stawow
aplikacji (wezsze niz mechaniczne), progi szczeki w jednostkach MJCF i skala
stolu, na ktorym ramie pracuje (`Desk`). `robot_config(spec)` sklada z nich
konfiguracje aplikacji dla danego ramienia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import JointConfig, TwinConfig, config_from_data, read_config_data

#: Korzen repozytorium (wspolny dla calego pakietu, patrz `lerobot_mp.paths`).
from ..paths import REPO_ROOT

logger = logging.getLogger(__name__)


def resolve_asset(path: str | Path) -> Path:
    """Sciezka zasobu wzgledem biezacego katalogu albo korzenia repozytorium.

    Ta sama kolejnosc co w podgladzie 3D: najpierw katalog, z ktorego
    uruchomiono aplikacje, potem repozytorium - zeby dzialalo i z klonu,
    i z instalacji `pip install -e .`.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in (Path.cwd(), REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return REPO_ROOT / candidate


@dataclass(frozen=True)
class JawThresholds:
    """Progi szczeki w jednostkach MJCF stawu chwytaka (rad dla obrotu, m dla suwaka).

    Wartosci domyslne to SO-101 (szczeka obrotowa, radiany).
    """

    #: Szczeka "stoi" - zmiana miedzy taktami mniejsza niz tyle.
    still: float = 0.02
    #: Szczeka szersza niz rozkaz / niz zamknieta o tyle = cos ja rozpiera.
    margin: float = 0.05
    #: Najwieksze dopuszczalne domykanie sie szczeki w serii sukcesu.
    creep: float = 0.03
    #: Szczeka szersza niz tyle = na pewno pusta.
    open: float = 0.40


@dataclass(frozen=True)
class Desk:
    """Gdzie ramie pracuje na stole [m, rad], uklad podstawy.

    Wartosci domyslne to SO-101: pierscienie celow i kostki zadan RL, fala
    kalibracyjna, losowanie kamer w symulacji i mapa stolu z gory.
    """

    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    wave_radius: tuple[float, float] = (0.12, 0.36)
    search_centre: tuple[float, float, float] = (0.2, 0.0, 0.05)
    search_dist: tuple[float, float] = (0.4, 0.9)
    camera_dist: tuple[float, float] = (0.45, 0.75)
    camera_centre: tuple[float, float, float] = (0.20, 0.0, 0.06)
    map_centre: tuple[float, float] = (0.22, 0.0)
    map_side: float = 0.56


@dataclass(frozen=True)
class RobotSpec:
    """Wszystko, czego blizniak potrzebuje o ramieniu, poza samym MJCF."""

    name: str
    title: str
    #: Plik MJCF samego ramienia (bez stolu i swiatla) - scena dokleja reszte.
    mjcf: str
    #: Stawy w kolejnosci aplikacji. Nazwy jak w MJCF.
    joints: tuple[str, ...]
    #: Ktory ze stawow jest chwytakiem (jednostki aplikacji 0..100); None = brak.
    gripper: str | None
    #: Site punktu narzedzia - to jego poze liczy IK i do niego celuje UI.
    tcp_site: str
    #: Cialo, na ktorym jedzie karta kalibracyjna trzymana w szczekach.
    hand_body: str
    #: Cialo podstawy - uklad, w ktorym podajemy pozy kamer i celow.
    base_body: str
    #: Poza spoczynkowa w jednostkach aplikacji.
    home: dict[str, float] = field(default_factory=dict)
    #: Kierunek podejscia (od nadgarstka przez koncowki szczek) i os zamykania
    #: szczek, oba w ukladzie `tcp_site`. Z nich karta kalibracyjna wylicza swoja
    #: nominalna poze: sterczy wzdluz podejscia, a jej normalna to os zamykania.
    #: Sprawdza je test na geometrii modelu, a nie wiara w konwencje autora MJCF.
    tcp_approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tcp_closing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Geomy czubkow obu szczek. Srodek miedzy nimi przy zamknietym chwytaku
    #: to miejsce, w ktorym szczeki trzymaja karte kalibracyjna.
    fingertips: tuple[str, str] = ("", "")
    #: Staw obracajacy narzedzie wokol kierunku podejscia. Fala kalibracyjna
    #: dobiera go analitycznie, zeby karta patrzyla na kamere.
    roll_joint: str | None = None
    #: Zakresy [jednostki aplikacji], z ktorych fala losuje pozostale stawy -
    #: wezsze niz mechaniczne, zeby karta krazyla nad stolem, a nie za plecami.
    wave_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Kamery zamontowane na ramieniu (nazwy kamer w MJCF).
    onboard_cameras: tuple[str, ...] = ()
    #: Ciala obu szczek (stala, ruchoma) - czujniki kontaktu chwytu w zadaniach RL.
    jaw_bodies: tuple[str, str] = ("", "")
    #: Limity stawow [jednostki aplikacji] - wezsze niz mechaniczne z MJCF.
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Najwieksza predkosc zadana [jednostki aplikacji / s].
    max_vel: dict[str, float] = field(default_factory=dict)
    #: Chwytak 0..100 liniowo na (zamkniety, otwarty) w jednostkach MJCF;
    #: None = przez tiki serwa STS3215 (backend `feetech`).
    gripper_range_q: tuple[float, float] | None = None
    #: Stawy MJCF sprzezone z innym stawem: staw -> (staw aplikacji, przelozenie).
    #: Kinematyka wpisuje je przy kazdym zapisie qpos.
    mirror_joints: dict[str, tuple[str, float]] = field(default_factory=dict)
    #: Progi szczeki w jednostkach MJCF chwytaka.
    jaw: JawThresholds = JawThresholds()
    #: Skala stolu, na ktorym ramie pracuje.
    desk: Desk = Desk()
    #: Dom etapami: grupy stawow jadace do pozy domowej jedna po drugiej (reszta stoi),
    #: kazdy etap sprawdzony w blizniaku na kolizje przed ruchem. Puste = jedna rampa
    #: wszystkich stawow naraz. Dla ramienia zlozonego na podstawie (A1X) prosta rampa
    #: ze wszystkimi stawami naraz obracala nadgarstek w kolumne barku.
    home_stages: tuple[tuple[str, ...], ...] = ()

    @property
    def mjcf_path(self) -> Path:
        return resolve_asset(self.mjcf)

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j != self.gripper)


SO101 = RobotSpec(
    name="so101",
    title="SO-101 (LeRobot)",
    mjcf="assets/robots/so101/so101.xml",
    joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    gripper="gripper",
    # `gripperframe` lezy miedzy szczekami, przy koncowkach - tam, gdzie chwyta.
    tcp_site="gripperframe",
    hand_body="gripper",
    base_body="base",
    home={
        "shoulder_pan": 0.0,
        "shoulder_lift": -23.3,
        "elbow_flex": 46.4,
        "wrist_flex": 3.3,
        "wrist_roll": 0.0,
        "gripper": 35.0,
    },
    fingertips=("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"),
    roll_joint="wrist_roll",
    wave_ranges={
        "shoulder_pan": (-75.0, 75.0),
        "shoulder_lift": (-80.0, 30.0),
        "elbow_flex": (-30.0, 90.0),
        "wrist_flex": (-70.0, 80.0),
    },
    onboard_cameras=("wrist_cam",),
    jaw_bodies=("gripper", "moving_jaw_so101_v1"),
    # Te same liczby co `config._default_joints` aplikacji dloni (test pilnuje).
    limits={
        "shoulder_pan": (-105.0, 105.0),
        "shoulder_lift": (-95.0, 95.0),
        "elbow_flex": (-92.0, 92.0),
        "wrist_flex": (-95.0, 95.0),
        "wrist_roll": (-150.0, 150.0),
        "gripper": (0.0, 100.0),
    },
    max_vel={
        "shoulder_pan": 140.0,
        "shoulder_lift": 120.0,
        "elbow_flex": 120.0,
        "wrist_flex": 160.0,
        "wrist_roll": 220.0,
        "gripper": 300.0,
    },
)

#: Galaxea A1X: szesc stawow obrotowych i chwytak na suwaku. MJCF generuje
#: `scripts/import_a1x_model.py` z URDF producenta (`assets/robots/a1x/README.md`).
A1X = RobotSpec(
    name="a1x",
    title="Galaxea A1X",
    mjcf="assets/robots/a1x/a1x.xml",
    joints=("arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6", "gripper"),
    gripper="gripper",
    tcp_site="tcp",
    hand_body="gripper_link",
    base_body="base_link",
    # Keyframe `home` modelu (1,0 / -1,6 / 0,6 rad); chwytak zamkniety (0), jak dom stanowiska
    # ("Ustaw dom = tu" zapisuje chwytak 0) i jak zglasza A1X bez pomiaru (`a1x.gripper_start`).
    home={
        "arm_joint1": 0.0,
        "arm_joint2": 57.3,
        "arm_joint3": -91.7,
        "arm_joint4": 34.4,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
        "gripper": 0.0,
    },
    # Site `tcp` jak z generatora galaxeo: x - podejscie, y - zamykanie szczek.
    tcp_approach=(1.0, 0.0, 0.0),
    tcp_closing=(0.0, 1.0, 0.0),
    # Kolejnosc tak, zeby od pierwszego do drugiego czubka szlo wzdluz `tcp_closing` (+y):
    # szczeka 1 jedzie po +y, szczeka 2 po -y. Srodek miedzy nimi nie zalezy od kolejnosci.
    fingertips=("finger2_tip", "finger1_tip"),
    roll_joint="arm_joint6",
    # Ramie i lokiec wysuniete do przodu, nadgarstek w dol: 85 % poz ma TCP 0,20-0,55 m
    # od osi i nad blatem (mediana 0,35 m wysokosci). Szersze zakresy z planu (J2 20..120,
    # J3 -150..-30, J4 +-60) dawaly 57 %, z karta zwykle ponad kadrem kamer przy stole.
    wave_ranges={
        "arm_joint1": (-75.0, 75.0),
        "arm_joint2": (60.0, 130.0),
        "arm_joint3": (-130.0, -60.0),
        "arm_joint4": (0.0, 80.0),
        "arm_joint5": (-45.0, 45.0),
    },
    jaw_bodies=("gripper_finger_link1", "gripper_finger_link2"),
    # Zakresy URDF minus 3-5 st. zapasu: rozkaz nie dociska stawu do ogranicznika.
    # Poza spoczynkowa ramienia (J2 ~ -0,2 st., J3 ~ +1,5 st.) lezy poza nimi celowo -
    # nadzor wyprowadza z niej staw miekkimi limitami (`keep_outside`), jak ramie 2 SO-101.
    limits={
        "arm_joint1": (-160.0, 160.0),
        "arm_joint2": (3.0, 177.0),
        "arm_joint3": (-187.0, -3.0),
        "arm_joint4": (-85.0, 85.0),
        "arm_joint5": (-85.0, 85.0),
        "arm_joint6": (-160.0, 160.0),
        "gripper": (0.0, 100.0),
    },
    # 90 st/s = ogranicznik predkosci mostka; chwytak jak SO-101 (ten sam `grip_step` zadan RL).
    max_vel={**{f"arm_joint{k}": 90.0 for k in range(1, 7)}, "gripper": 300.0},
    # Wysuw szczeki [m]: 0 = plytki zsuniete (czubki 3 mm od siebie), 0,05 = pelne otwarcie.
    gripper_range_q=(0.0, 0.05),
    mirror_joints={"gripper_finger_joint2": ("gripper", -1.0)},
    # Metry: 20 mm wysuwu = 43 mm miedzy czubkami, wiecej niz przekatna 42 mm kostki.
    jaw=JawThresholds(still=0.001, margin=0.003, creep=0.002, open=0.020),
    desk=Desk(
        goal_radius=(0.25, 0.50),
        goal_height=(0.03, 0.25),
        cube_radius=(0.28, 0.45),
        cube_bearing=(-0.7, 0.7),
        wave_radius=(0.20, 0.55),
        search_centre=(0.35, 0.0, 0.08),
        search_dist=(0.6, 1.2),
        camera_dist=(0.7, 1.1),
        camera_centre=(0.35, 0.0, 0.08),
        map_centre=(0.36, 0.0),
        map_side=0.9,
    ),
    # Ze zlozonego spoczynku (J2 ~0, J3 ~0): najpierw lokiec rozklada przedramie w gore,
    # potem nadgarstek, potem bark i podstawa - kolejnosc sprawdzona w blizniaku (path_clear);
    # nadgarstek przed lokciem uderza chwytakiem w kolumne barku (arm_link1) po 7 % drogi.
    home_stages=(("arm_joint3",), ("arm_joint4", "arm_joint5", "arm_joint6"),
                 ("arm_joint2", "arm_joint1"), ("gripper",)),
)

REGISTRY: dict[str, RobotSpec] = {SO101.name: SO101, A1X.name: A1X}


def get_spec(name: str) -> RobotSpec:
    try:
        return REGISTRY[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Nieznane ramie {name!r}. Dostepne: {known}") from exc


def joint_configs(spec: RobotSpec) -> dict[str, JointConfig]:
    """JointConfig kazdego stawu opisu, w kolejnosci opisu.

    `gain/invert/offset` zostaja domyslne - czyta je tylko mapowanie dloni,
    blizniak nie.
    """
    return {
        n: JointConfig(min=spec.limits[n][0], max=spec.limits[n][1], max_vel=spec.max_vel[n])
        for n in spec.joints
    }


#: Pliki, z ktorych juz ostrzezono o pominietych tabelach SO-101 (raz na proces).
_SKIP_WARNED: set[tuple[str, str]] = set()


def robot_config(
    spec: RobotSpec, path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> TwinConfig:
    """`load_config()` dla tego ramienia: stawy z opisu, poza domowa z opisu.

    Plik YAML (`path` albo `LEROBOT_MP_CONFIG`) i `overrides` leza na wierzchu,
    jak w `load_config`. Dwie zasady:

    * Sekcje `joints` i `safety.home` pliku to tabele aplikacji dloni, kluczowane
      stawami SO-101 (`configs/default.yaml`, kopiowany do `configs/local.yaml`).
      Stosuja sie tylko do ramienia o stawach SO-101 (te same co w aplikacji dloni);
      innemu ramieniu plik ich nie daje - limity i poze domowa bierze z opisu
      (nazwa `gripper` jest wspolna, wiec nawet pojedynczy wpis SO-101 dalby
      mu zly zakres albo dom). `overrides` dzialaja zawsze i scisle.
    * Poza domowa z opisu wchodzi tylko wtedy, gdy ani plik, ani `overrides` nie
      maja `safety.home`. Czesciowa `home` z pliku zastepuje ja w calosci, tak
      jak w `load_config` (brakujace stawy uzupelnia potem `SafetySupervisor`).

    Dla SO-101 wynik rozni sie wiec od `load_config(path, overrides)` - w kazdym
    przypadku - tylko `gain` stawow, ktorego blizniak nie czyta.
    """
    data = read_config_data(path)
    if tuple(spec.joints) != SO101.joints:
        data = _without_hand_app_tables(data, spec, path)
    if not (_has_home(data) or _has_home(overrides or {})):
        safety = data.get("safety")
        data = dict(data, safety=dict(safety if isinstance(safety, dict) else {}, home=dict(spec.home)))
    return config_from_data(data, overrides, joints=joint_configs(spec))


def _has_home(data: dict[str, Any]) -> bool:
    safety = data.get("safety")
    return isinstance(safety, dict) and "home" in safety


def _without_hand_app_tables(data: dict[str, Any], spec: RobotSpec, path: str | Path | None) -> dict[str, Any]:
    """Zawartosc pliku bez sekcji `joints` i `safety.home` (tabel SO-101)."""
    data = dict(data)
    skipped = []
    if "joints" in data:
        del data["joints"]
        skipped.append("joints")
    if _has_home(data):
        data["safety"] = {k: v for k, v in data["safety"].items() if k != "home"}
        skipped.append("safety.home")
    key = (str(path), spec.name)
    if skipped and key not in _SKIP_WARNED:
        _SKIP_WARNED.add(key)
        logger.warning("Konfiguracja: sekcje %s pliku sa dla stawow SO-101 - ramie %s bierze je z opisu",
                       ", ".join(skipped), spec.name)
    return data
