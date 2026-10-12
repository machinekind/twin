"""Mrowka RealAnt w blizniaku: model MuJoCo, tagi na grzbiecie i na maszcie, poza z kamer stanowiska.

    ant = Ant.from_dict(ws.ant) if ws.ant else Ant()     # tag 2 plasko na grzbiecie, tag 1 na maszcie
    obs = observe(frames, ws.cameras, ant.tags)           # jak kamera PMD (`tagbody`): uklad tagow z kadrow
    ant.tags.layout = learn_layout(obs_a + obs_b + ..., ant.tags).layout
    fix = locate(frames, ws.cameras, ant)                 # T_ant2base z kamer, ktore maja poze
    ant.mount = level_mount(fix.tags.T_body2base, ant)    # wysokosc i pochylenie taga - z blatu

Trzy uklady, ktorych trzeba pilnowac:

* **uklad mrowki** - srodek w polowie stosu plyt korpusu, x do przodu (kierunek chodu z realant-rl),
  y w lewo, z do gory. Nogi 1..4 (serwa s1-s2, ..., s7-s8) wychodza na -x-y, +x-y, +x+y, -x+y: 1 z tylu
  po prawej, 2 z przodu po prawej, 3 z przodu po lewej, 4 z tylu po lewej. Korpus modelu jest do tego
  ukladu obrocony o 180 st. (`scene.ANT_TORSO_IN_ANT`).
* **uklad tagow** (`tagbody.TagBody`) - uklad taga na grzbiecie (`ids[0]`, domyslnie 2). Gdzie jest tag
  na maszcie (1) wzgledem niego, uczy sie z kadrow: kazdy kadr z oboma tagami daje ich wzajemna poze,
  wszystkie rogi naraz - uklad (`tagbody.learn_layout`). Linijka niepotrzebna.
* **montaz** (`mount`) - gdzie tag na grzbiecie siedzi w ukladzie mrowki: x, y, z [m] i odchylenie
  (wokol pionu), pochylenie, przechylenie [st.] (`mount_matrix`). Na tej mrowce krawedzie taga sa rownolegle
  do plyty korpusu, wiec jego osie ida wzdluz nog: odchylenie 45 + k * 90 st. (`snap_yaw`), a tag na maszcie
  jest do niego prostopadly (`square_layout`).

Montazu kamery nie widza w calosci - widza tylko tagi, nie korpus. Dzieli sie wiec na dwie czesci:

* wysokosc taga nad srodkiem korpusu i jego pochylenie - **z blatu** (`level_mount`): mrowka lezy albo
  stoi na blacie, wiec korpus jest poziomo na wysokosci wynikajacej z katow nog (`rest_height`), a tag
  widziany wyzej / pochylony jest o tyle wyzej / pochylony wzgledem korpusu;
* polozenie taga na grzbiecie (x, y) i jego obrot wokol pionu - **linijka i oko**: od tego zalezy, gdzie
  w bliznaku sa nogi; w Sim-Real krawedzie modelu na kadrze kamery pokazuja, czy trafiaja w prawdziwe.

Model to RealAnt v2 z realant-rl (`assets/robots/realant`, drukowane biodra, serwa AX-12A). Mrowka z tego
stanowiska to v1 (aluminiowe biodra FP04): korpus, nogi i serwa te same, inne uchwyty bioder - rozmiar
i polozenie stop sie zgadzaja, ksztalt bioder w renderze nie.

Numery tagow: karta kalibracyjna ramienia ma tagi 0 i 1 (`Card.ids`). Tag 1 na maszcie mrowki jest dla
fali nie do odroznienia od karty - mrowka nie moze byc w kadrze podczas kalibracji kamer (albo inny numer
na maszcie; `ids` sa dowolne, panel ostrzega).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from .kinematics import inverse, pose
from .robots import resolve_asset
from .scene import ANT_JOINTS, STANDIN_PREFIX, AntPlacement
from .tagbody import BodyFix, TagBody
from .tagbody import locate as locate_body

MJCF = "assets/robots/realant/realant.xml"
#: Stawy modelu w kolejnosci realant-rl (akcja polityki): biodro i kostka nogi 1, potem 2, 3, 4.
JOINTS = ANT_JOINTS
#: Spod korpusu (dolna plyta) pod srodkiem stosu plyt [m] - na tym mrowka lezy, gdy nogi sa wyzej.
BELLY = 0.0244
#: Gorna plyta korpusu nad srodkiem stosu plyt [m].
TOP = 0.0244
#: Promien zaokraglenia czubka stopy w modelu [m].
TIP_RADIUS = 0.0059

#: Pozy nog [st.]: "lezy" - serwa wylaczone, brzuch na blacie, nogi poziomo; "stoi" - start z realant-rl
#: (biodra 0, kostki 0,9 rad), na czubkach stop.
LEG_POSES: dict[str, dict[str, float]] = {
    "lezy": {j: 0.0 for j in JOINTS},
    "stoi": {j: (0.0 if j.startswith("hip") else round(float(np.degrees(0.9)), 2)) for j in JOINTS},
}

#: Tag na grzbiecie: srodek korpusu, 5 mm nad gorna plyta (tektura), krawedzie rownolegle do plyty korpusu - osie
#: taga wzdluz nog, czyli 45 st. od przodu mrowki (przod jest MIEDZY nogami 2 i 3). Ktora z czterech nog pokazuje
#: x taga, mowi kalibracja kierunku (`snap_yaw`: 45 + k * 90 st.).
DEFAULT_MOUNT: dict[str, float] = {"x": 0.0, "y": 0.0, "z": TOP + 0.005, "yaw": 45.0, "pitch": 0.0, "roll": 0.0}
#: Tag 2 na grzbiecie (uklad), tag 1 na maszcie; bok czarnego kwadratu do zmierzenia linijka.
DEFAULT_IDS = (2, 1)
DEFAULT_TAG_SIZE = 0.040


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def zyx(R: np.ndarray) -> tuple[float, float, float]:
    """(odchylenie, pochylenie, przechylenie) [st.] obrotu R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    R = np.asarray(R, float)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)


def mount_matrix(mount: Mapping[str, float]) -> np.ndarray:
    """4x4 T_tag2ant z montazu: przesuniecie [m] i Rz(odchylenie) @ Ry(pochylenie) @ Rx(przechylenie)."""
    m = {**DEFAULT_MOUNT, **dict(mount)}
    R = _rz(m["yaw"]) @ _ry(m["pitch"]) @ _rx(m["roll"])
    return pose(R, np.array([m["x"], m["y"], m["z"]], float))


def snap_yaw(yaw_deg: float, max_off_deg: float = 20.0) -> tuple[float, float]:
    """(odchylenie przyciagniete do 45 + k * 90 st., odchylka [st.]) - tag z krawedziami wzdluz plyty korpusu.

    Dalej niz `max_off_deg` od siatki - bez przyciagania (tag przyklejony inaczej), odchylka jak jest.
    """
    k = np.round((yaw_deg - 45.0) / 90.0)
    snapped = float(np.degrees(np.angle(np.exp(1j * np.radians(45.0 + 90.0 * k)))))
    off = float(np.degrees(np.angle(np.exp(1j * np.radians(yaw_deg - snapped)))))
    return (snapped, off) if abs(off) <= max_off_deg else (float(yaw_deg), off)


def _square_rotations() -> list[np.ndarray]:
    """24 obroty wlasciwe o katach wielokrotnosci 90 st. (macierze permutacji ze znakami, wyznacznik +1)."""
    from itertools import permutations, product
    out = []
    for perm in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for col, (row, sg) in enumerate(zip(perm, signs)):
                R[row, col] = sg
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


_SQUARE = _square_rotations()


def square_layout(layout: Mapping[int, np.ndarray], max_off_deg: float = 15.0) -> tuple[dict[int, np.ndarray],
                                                                                         dict[int, float]]:
    """Uklad z obrotami przyciagnietymi do prostopadlych (krawedzie tagow rownolegle): ({id: T}, {id: odchylka st.}).

    Maszt jest prostopadly do grzbietu, a tag na nim ma krawedzie rownolegle do taga na grzbiecie (zdjecie
    stanowiska: normalna taga 1 = -x taga 2, gora taga 1 = normalna taga 2). Nauka z kadrow daje obrot z bledem
    rzedu stopnia; przyciagniecie go usuwa, a odchylka jest sprawdzianem naklejenia. Tag dalej niz `max_off_deg`
    od kazdego prostopadlego obrotu zostaje bez zmian. Polozenie zostaje z nauki.
    """
    out, offs = {}, {}
    for i, T in layout.items():
        T = np.asarray(T, float)
        R = T[:3, :3]
        best = min(_SQUARE, key=lambda Q: np.linalg.norm(Q - R))
        off = float(np.degrees(np.arccos(np.clip((np.trace(best.T @ R) - 1) / 2, -1, 1))))
        offs[int(i)] = off
        T2 = T.copy()
        if off <= max_off_deg:
            T2[:3, :3] = best
        out[int(i)] = T2
    return out, offs


def mount_params(T_tag2ant: np.ndarray) -> dict[str, float]:
    T = np.asarray(T_tag2ant, float)
    yaw, pitch, roll = zyx(T[:3, :3])
    return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]), "yaw": yaw, "pitch": pitch, "roll": roll}


# ------------------------------------------------------------------ model
@lru_cache(maxsize=1)
def _model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_path(str(resolve_asset(MJCF)))
    return m, mujoco.MjData(m)


def rest_height(joints: Mapping[str, float]) -> float:
    """Wysokosc srodka korpusu nad blatem [m], gdy mrowka z tymi katami nog [st.] stoi/lezy poziomo.

    Nizsze z dwoch podparc: brzuch (`BELLY`) albo czubki stop (z zaokragleniem). Nogi poziomo
    ("lezy") maja stopy nad brzuchem - mrowka lezy na brzuchu.
    """
    m, d = _model()
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0                                      # wolny staw: korpus w zerze, bez obrotu
    for name, deg in joints.items():
        j = m.joint(name).id
        d.qpos[m.jnt_qposadr[j]] = np.clip(np.radians(deg), *m.jnt_range[j])
    mujoco.mj_kinematics(m, d)
    tips = [d.site_xpos[m.site(f"foot_tip_{k}").id][2] for k in range(1, 5)]
    return float(max(BELLY, TIP_RADIUS - min(tips)))


# ------------------------------------------------------------------ mrowka
def _T_or_none(x: Any) -> np.ndarray | None:
    return None if x is None else np.asarray(x, float).reshape(4, 4)


@dataclass
class Ant:
    """Mrowka na stanowisku: jej tagi (uklad z kamer), montaz taga na grzbiecie, nogi i ostatnia poza."""

    tags: TagBody = field(default_factory=lambda: TagBody("mrowka", ids=DEFAULT_IDS, tag_size=DEFAULT_TAG_SIZE))
    #: Gdzie tag `tags.ids[0]` siedzi na mrowce (`mount_matrix`): x, y, z [m], yaw, pitch, roll [st.].
    mount: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MOUNT))
    #: Katy nog [st.] (`JOINTS`) - do sceny i do wysokosci korpusu nad blatem.
    joints: dict[str, float] = field(default_factory=lambda: dict(LEG_POSES["lezy"]))
    #: Ostatnia poza z kamer (uklad mrowki w ukladzie podstawy ramienia); None = jeszcze nie szukano.
    T_ant2base: np.ndarray | None = None
    #: Tylko symulacja: mrowka NAPRAWDE w bliznaku - {"T_ant2base": 4x4, "mount": {...}, "layout": {id: 4x4}}.
    #: Scena rysuje prawde, a kalibracja (uklad, montaz, poza) liczy sie z kadrow jak dla prawdziwej,
    #: wiec wynik da sie z prawda porownac (jak `CameraRecord.sim_pose`). None = mrowka prawdziwa.
    sim: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.mount = {**DEFAULT_MOUNT, **{k: float(v) for k, v in dict(self.mount).items()}}
        self.joints = {**LEG_POSES["lezy"], **{k: float(v) for k, v in dict(self.joints).items()}}
        unknown = sorted(set(self.joints) - set(JOINTS))
        if unknown:
            raise ValueError(f"mrowka nie ma stawow {unknown} (sa: {', '.join(JOINTS)})")
        self.T_ant2base = _T_or_none(self.T_ant2base)
        if self.sim is not None:
            self.sim = {"T_ant2base": _T_or_none(self.sim["T_ant2base"]),
                        "mount": {**DEFAULT_MOUNT, **dict(self.sim.get("mount") or {})},
                        "layout": {int(i): np.asarray(T, float) for i, T in dict(self.sim.get("layout") or {}).items()}}

    def T_ref2ant(self) -> np.ndarray:
        """Tag na grzbiecie (`tags.ids[0]`) w ukladzie mrowki."""
        return mount_matrix(self.mount)

    def tag_poses(self) -> dict[int, np.ndarray]:
        """{id: T_tag2ant} tagow o znanym ukladzie."""
        T0 = self.T_ref2ant()
        return {i: T0 @ self.tags.layout[i] for i in self.tags.known()}

    @property
    def simulated(self) -> bool:
        return self.sim is not None

    def scene_pose(self) -> np.ndarray | None:
        """Gdzie mrowka blizniaka startuje w scenie: ostatnia poza z kamer, a bez niej - w symulacji - start mrowki
        udajacej prawdziwa (dalej chodzi w fizyce, a tagi ja dociagaja)."""
        if self.T_ant2base is not None:
            return self.T_ant2base
        return self.sim["T_ant2base"] if self.sim is not None else None

    def placement(self) -> AntPlacement | None:
        """Mrowka blizniaka do `SceneConfig.ant` (fizyka, nogi z odczytow serw, poza dociagana tagami).

        None, gdy nie wiadomo, gdzie stoi. Nosi tagi takie, jakie o nich wiadomo (panel 3D, Sim-Real); w symulacji
        jest schowana przed kamerami - te maja widziec tylko mrowke udajaca prawdziwa (`standin_placement`).
        """
        T = self.scene_pose()
        if T is None:
            return None
        return AntPlacement(MJCF, T.copy(), dict(self.joints),
                            [(i, T_tag, self.tags.tag_size) for i, T_tag in self.tag_poses().items()],
                            hidden=self.simulated)

    def standin_placement(self) -> AntPlacement | None:
        """Tylko symulacja: mrowka udajaca prawdziwa (`SceneConfig.ant_standin`) - z PRAWDZIWYMI tagami (`sim`)."""
        if self.sim is None:
            return None
        T0 = mount_matrix(self.sim["mount"])
        layout = {self.tags.ref: np.eye(4), **self.sim["layout"]}
        tags = {i: T0 @ layout[i] for i in self.tags.ids if i in layout}
        return AntPlacement(MJCF, self.sim["T_ant2base"].copy(), dict(self.joints),
                            [(i, T_tag, self.tags.tag_size) for i, T_tag in tags.items()], prefix=STANDIN_PREFIX)

    def rest_height(self) -> float:
        return rest_height(self.joints)

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        tags = self.tags.to_dict()
        for key in ("optical", "fov_deg", "resolution"):   # pola kamery PMD, mrowce niepotrzebne
            tags.pop(key, None)
        return {"tags": tags, "mount": dict(self.mount), "joints": dict(self.joints),
                "T_ant2base": None if self.T_ant2base is None else self.T_ant2base.tolist(),
                "sim": None if self.sim is None else {
                    "T_ant2base": self.sim["T_ant2base"].tolist(), "mount": dict(self.sim["mount"]),
                    "layout": {str(i): T.tolist() for i, T in self.sim["layout"].items()}}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ant:
        d = dict(data)
        tags = TagBody.from_dict({"name": "mrowka", "ids": DEFAULT_IDS, "tag_size": DEFAULT_TAG_SIZE,
                                  **dict(d.pop("tags", {}) or {})})
        return cls(tags=tags, **d)


# ------------------------------------------------------------------ poza z kamer
@dataclass
class AntFix:
    """Poza mrowki z kamer: uklad mrowki w ukladzie podstawy i to, z czego ja policzono."""

    T_ant2base: np.ndarray
    #: Poza ukladu tagow (taga na grzbiecie) z kamer - `tagbody.BodyFix`.
    tags: BodyFix

    @property
    def position(self) -> np.ndarray:
        return self.T_ant2base[:3, 3].copy()

    @property
    def heading_deg(self) -> float:
        """Kierunek przodu mrowki (x) w plaszczyznie blatu, od osi x podstawy [st.]."""
        x = self.T_ant2base[:3, 0]
        return float(np.degrees(np.arctan2(x[1], x[0])))

    @property
    def tilt_deg(self) -> float:
        """Odchylenie osi z mrowki od pionu [st.] - na plaskim blacie ~0, gdy montaz jest wypoziomowany."""
        return float(np.degrees(np.arccos(np.clip(self.T_ant2base[2, 2], -1.0, 1.0))))


def ant_from_tags(T_body2base: np.ndarray, ant: Ant) -> np.ndarray:
    """Uklad mrowki w ukladzie podstawy z pozy taga na grzbiecie i montazu."""
    return np.asarray(T_body2base, float) @ inverse(ant.T_ref2ant())


def locate(frames: Mapping[str, np.ndarray], cameras: Iterable, ant: Ant) -> AntFix | None:
    """Poza mrowki z kadrow kamer z poza i K (mediana kamer, `tagbody.locate`); None, gdy nikt nie widzi tagow."""
    fix = locate_body(frames, cameras, ant.tags)
    if fix is None:
        return None
    return AntFix(ant_from_tags(fix.T_body2base, ant), fix)


def level_mount(T_body2base: np.ndarray, ant: Ant, height: float | None = None,
                max_tilt_deg: float = 35.0) -> dict[str, float]:
    """Montaz z wysokoscia i pochyleniem taga wzietymi z blatu; x, y i odchylenie zostaja z `ant.mount`.

    Mrowka lezy/stoi na plaskim blacie, wiec jej uklad ma z w pionie, a srodek na wysokosci `height`
    (domyslnie `rest_height` z katow nog). Tag widziany przez kamery (`T_body2base`) rozklada sie na
    obrot korpusu wokol pionu i odchylenie montazu (znane), pochylenie i przechylenie montazu (szukane),
    a jego wysokosc nad blatem minus `height` to wysokosc taga nad srodkiem korpusu.
    Tag pochylony o wiecej niz `max_tilt_deg` - to nie jest tag na grzbiecie albo mrowka nie lezy plasko.
    """
    T = np.asarray(T_body2base, float)
    h = rest_height(ant.joints) if height is None else float(height)
    tilt = float(np.degrees(np.arccos(np.clip(T[2, 2], -1.0, 1.0))))
    if tilt > max_tilt_deg:
        raise ValueError(f"tag {ant.tags.ref} pochylony o {tilt:.0f} st. od pionu - to nie tag lezacy na grzbiecie "
                         f"(albo mrowka nie stoi plasko na blacie)")
    _, pitch, roll = zyx(T[:3, :3])
    return {**ant.mount, "z": float(T[2, 3] - h), "pitch": pitch, "roll": roll}


# ------------------------------------------------------------------ demo
#: Mrowka w `lerobot-twin demo --ant`: stoi na palcach 55 cm przed podstawa, 30 cm w lewo, przodem do +x.
DEMO_ANT_XY = (0.55, 0.30)
#: Prawdziwy montaz i uklad tagow mrowki w demo (kalibracja ich nie zna - wychodza z kadrow).
#: Osie taga wzdluz nog (jak na zdjeciu stanowiska), ale inna noga niz w `DEFAULT_MOUNT` - kalibracja ma co robic.
DEMO_MOUNT = {"x": 0.006, "y": -0.004, "z": TOP + 0.006, "yaw": 135.0, "pitch": 1.5, "roll": -1.0}
#: Obrot taga 1 w ukladzie taga 2 (zdjecie stanowiska): normalna = -x taga 2, gora = normalna taga 2.
MAST_IN_BACK_TAG = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def demo_layout() -> dict[int, np.ndarray]:
    """Tag 1 na maszcie w ukladzie taga 2 - geometria ze zdjecia stanowiska: prostopadle, krawedzie rownolegle,
    ~9,5 cm wzdluz +x taga 2 (nad noga), 7,5 cm nad nim, twarza w strone taga 2."""
    return {1: pose(MAST_IN_BACK_TAG, np.array([0.095, -0.005, 0.075]))}


def _sim_ant(data: dict[str, Any], xy: tuple[float, float]) -> dict[str, Any]:
    """`data` + symulowana mrowka stojaca w `xy` (uklad podstawy) i dwie kamery symulowane patrzace na nia."""
    from .calib.simulate import look_at

    out = dict(data)
    joints = dict(LEG_POSES["stoi"])
    T = pose(np.eye(3), np.array([xy[0], xy[1], rest_height(joints)]))
    ant = Ant(joints=joints, sim={"T_ant2base": T, "mount": DEMO_MOUNT, "layout": demo_layout()})
    out["ant"] = ant.to_dict()
    W, H, fovy = 960, 720, 60.0
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    K = [[f, 0.0, (W - 1) / 2], [0.0, f, (H - 1) / 2], [0.0, 0.0, 1.0]]
    target = np.array([xy[0], xy[1], 0.03])
    cams = [c for c in data.get("cameras", []) if not str(c.get("name", "")).startswith("sym-mrowka")]
    # Kamery wzgledem mrowki jak w demo (mrowka w DEMO_ANT_XY, kamery w (1,05, 0,40, 0,50) i (0,45, -0,20, 0,55)).
    for name, off in (("sym-mrowka-przod", (0.50, 0.10, 0.50)), ("sym-mrowka-bok", (-0.10, -0.50, 0.55))):
        eye = np.array([xy[0] + off[0], xy[1] + off[1], off[2]])
        Tc = look_at(eye, target).tolist()
        cams.append({"name": name, "source": "sim", "width": W, "height": H, "fps": 30, "K": K, "dist": None,
                     "T_cam2base": Tc, "intrinsics_from": "symulacja",
                     "calibration": {"trusted": True, "reason": "przyklad: poza prawdziwa z symulacji", "rms_px": 0.0},
                     "enabled": True, "sim_pose": Tc, "intrinsics_info": {}})
    out["cameras"] = cams
    return out


def add_demo_ant(data: dict[str, Any]) -> dict[str, Any]:
    """Stanowisko z pliku przykladowego + symulowana mrowka z tagami i dwie kamery patrzace na nia.

    Blat powiekszony do 1,6 x 1,2 m (mrowka chodzi 10 cm/s). Kamery symulowane maja zaufana poze
    (prawda) - tak jak te z przykladu; uklad tagow, montaz i kurs mrowki zostaja do wyznaczenia z kadrow.
    """
    out = dict(data)
    out["table"] = {**dict(data.get("table", {})), "size": [1.6, 1.2]}
    return _sim_ant(out, DEMO_ANT_XY)


# ------------------------------------------------------------------ mrowka na koncu blatu
#: Mrowka w `add_ant`: srodek tyle przed daleka krawedzia blatu [m].
TABLE_END_INSET = 0.20


def table_far_x(table: Mapping[str, Any]) -> float:
    """Gdzie os +x podstawy wychodzi z blatu [m, uklad podstawy] (`scene.Table`: rozmiar, base_xy, base_yaw)."""
    from .scene import Table

    t = Table()
    size = np.asarray(table.get("size", t.size), float)
    p = np.asarray(table.get("base_xy", t.base_xy), float)
    yaw = np.radians(float(table.get("base_yaw", t.base_yaw)))
    d = np.array([np.cos(yaw), np.sin(yaw)])
    hits = [((np.sign(d[k]) * size[k] / 2) - p[k]) / d[k] for k in range(2) if abs(d[k]) > 1e-9]
    return float(min(hits))


def ant_xy_at_table_end(data: Mapping[str, Any]) -> tuple[float, float]:
    """(x, y) mrowki na dalekim koncu blatu w ukladzie podstawy."""
    far = table_far_x(dict(data.get("table") or {}))
    return far - TABLE_END_INSET, 0.0


def add_ant(data: dict[str, Any], xy: tuple[float, float] | None = None) -> dict[str, Any]:
    """Stanowisko + symulowana mrowka stojaca na ISTNIEJACYM blacie, domyslnie na jego dalekim koncu
    (`ant_xy_at_table_end`: 20 cm przed krawedzia, y = 0), i dwie kamery
    symulowane patrzace na nia. Blat bez zmian (`add_demo_ant` go powieksza)."""
    return _sim_ant(data, ant_xy_at_table_end(data) if xy is None else (float(xy[0]), float(xy[1])))
