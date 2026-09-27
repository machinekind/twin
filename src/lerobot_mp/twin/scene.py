"""Scena MuJoCo skladana z opisu stanowiska: stol, ramie, kamery, karta, obiekty.

Scena jest kompilowana od zera z `MjSpec` przy kazdym `build` - tak jak
epizody w galaxeo-manipulators - wiec nic nie zostaje z poprzedniej konfiguracji,
a randomizacja wygladu czy polozenia kamer to po prostu inny `SceneConfig`.

Konwencje, ktorych trzeba pilnowac:

* **Poza kamery jest w konwencji OpenCV** (z do przodu, y w dol) i wzgledem
  PODSTAWY ramienia - tak ja zwraca kalibracja. MuJoCo patrzy wzdluz -z z y w
  gore, stad `R_mj = R_cv @ diag(1, -1, -1)` przy wstawianiu kamery.
* **Kamera dostaje pelna macierz K**, nie samo pole widzenia: `focal_pixel`,
  `principal_pixel` i `resolution`. Prawdziwa kamera ma punkt glowny poza
  srodkiem i czesto rozne fx/fy - z samym `fovy` symulowany kadr rozjezdzalby
  sie z prawdziwym o kilka pikseli, i to po cichu. Zgodnosc pilnuje test.
* Ramie jest wstawiane pod prefiksem `robot/`, wiec jego stawy, aktuatory
  i site'y maja w scenie nazwy `robot/<nazwa>`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .calib.card import Card
from .calib.tags import MARKER_FRACTION
from .calib.tags import render as render_tag
from .kinematics import RobotKinematics, inverse, pose
from .robots import RobotSpec

PREFIX = "robot/"
#: Mrowka RealAnt (`AntPlacement`) wchodzi do sceny pod tym prefiksem: `ant/torso`, `ant/hip_1`...
ANT_PREFIX = "ant/"
#: Tylko symulacja: mrowka udajaca prawdziwa (chodzi na wlasnej polityce, tagi widza kamery symulowane).
STANDIN_PREFIX = "antprawda/"
#: Grupa geomow niewidocznych dla kamer sceny (render MuJoCo pokazuje 0-2), a rysowanych w panelu 3D.
HIDDEN_GROUP = 4
#: Kolor kopii mrowki schowanej przed kamerami (demo: widac ja obok mrowki udajacej prawdziwa).
GHOST_RGBA = (0.25, 0.65, 1.0, 0.55)
#: Korpus modelu RealAnt w ukladzie mrowki: model ma nogi 1..4 na +x+y, -x+y, -x-y, +x-y korpusu
#: i startuje obrocony o 180 st. wokol z, zeby "do przodu" bylo +x swiata (realant-rl) - uklad
#: mrowki to wlasnie ten "przod = +x".
ANT_TORSO_IN_ANT = np.diag([-1.0, -1.0, 1.0, 1.0])
#: Stawy nog w kolejnosci realant-rl (akcja polityki, serwa po mapie z rollout_server).
ANT_JOINTS = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")
#: MuJoCo kamera patrzy wzdluz -z z y w gore; OpenCV wzdluz +z z y w dol.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraView:
    """Kamera w scenie: intrynsyki i poza wzgledem podstawy ramienia (OpenCV)."""

    name: str
    K: np.ndarray
    width: int
    height: int
    T_cam2base: np.ndarray

    @staticmethod
    def from_fov(name: str, width: int, height: int, fovy_deg: float, T_cam2base: np.ndarray) -> CameraView:
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        K = np.array([[f, 0.0, width / 2 - 0.5], [0.0, f, height / 2 - 0.5], [0.0, 0.0, 1.0]])
        return CameraView(name, K, width, height, np.asarray(T_cam2base, float))


@dataclass
class Table:
    #: Wymiary blatu [m] i wysokosc blatu nad podloga.
    size: tuple[float, float] = (0.9, 0.6)
    height: float = 0.75
    #: Gdzie na blacie stoi podstawa ramienia (od srodka blatu) i jak jest obrocona.
    #: Domyslnie przy krotszej krawedzi, przodem (+x) w strone blatu - wtedy cala
    #: przestrzen robocza (0,1-0,4 m przed podstawa, +-0,3 m w bok) lezy na stole.
    base_xy: tuple[float, float] = (-0.35, 0.0)
    base_yaw: float = 0.0
    rgba: tuple[float, float, float, float] = (0.55, 0.50, 0.44, 1.0)


@dataclass
class Box:
    """Prosty obiekt na stole - do zadan RL i do sprawdzania kadrow."""

    name: str
    size: tuple[float, float, float]            # polowy bokow [m]
    xy: tuple[float, float]                     # polozenie na blacie w ukladzie podstawy [m]
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)
    mass: float = 0.05
    free: bool = True
    #: Cialo mocap bez kolizji (tylko do ogladania) - np. cel odlozenia pickplace. Poze
    #: ustawia sie przez `data.mocap_pos` / `mocap_quat`; obiekty przez nie przenikaja.
    mocap: bool = False


@dataclass
class Panel:
    """Plaski obraz w scenie - tablica kalibracyjna, plakat, wydruk na blacie.

    `image` (H, W) albo (H, W, 3) uint8 wypelnia prostokat `size` [m] w plaszczyznie
    xy panelu, gorny wiersz obrazu na +y, patrzy wzdluz +z. Poza wzgledem podstawy.
    """

    name: str
    image: np.ndarray
    size: tuple[float, float]
    T_panel2base: np.ndarray


@dataclass
class AntPlacement:
    """Mrowka RealAnt w scenie (`twin.ant.Ant.placement`): gdzie stoi, jak ma nogi i gdzie ma tagi.

    Mrowka blizniaka (`ant/`) chodzi w fizyce sceny: nogi to stawy z serwami AX-12A z modelu, a ich cele
    przychodza z odczytow serw prawdziwej mrowki (`antmirror`); tagi tylko dociagaja jej poze. W symulacji
    obok stoi mrowka udajaca prawdziwa (`antprawda/`, `SceneConfig.ant_standin`). Uklad mrowki: srodek
    w polowie stosu plyt korpusu, x do przodu (kierunek chodu z realant-rl), z do gory.
    """

    mjcf: str
    T_ant2base: np.ndarray
    #: {staw modelu ("hip_1", "ankle_1", ...): kat [st.]}; brakujace = 0.
    joints: dict[str, float] = field(default_factory=dict)
    #: [(id taga, T_tag2ant, bok czarnego kwadratu [m])] - tagi naklejone na mrowke.
    tags: list[tuple[int, np.ndarray, float]] = field(default_factory=list)
    #: True: korpus na wolnym stawie `<prefiks>root` - mrowka chodzi w fizyce, `T_ant2base` to start.
    #: False: korpus mocap (poza ustawiana z zewnatrz).
    free: bool = True
    prefix: str = ANT_PREFIX
    #: Schowana przed kamerami sceny (grupa `HIDDEN_GROUP`, polprzezroczysta, bez tagow) - kopia w demo,
    #: ktorej kamery symulowane nie moga widziec obok mrowki udajacej prawdziwa.
    hidden: bool = False


@dataclass
class HandCamera:
    """Kamera jadaca na ciele dloni (`RobotSpec.hand_body`), np. kamera na nadgarstku A1X.

    `T_cam2hand` jest w osiach OpenCV (z do przodu, y w dol) wzgledem ciala dloni. Kamera to
    idealna kamera otworkowa dana przez `fovy_deg`: kwadratowe piksele, punkt glowny w srodku obrazu.
    """

    name: str                          # model name becomes PREFIX + name
    T_cam2hand: np.ndarray
    fovy_deg: float
    width: int = 160
    height: int = 120


@dataclass(frozen=True)
class HandCameraJitter:
    """Maly blad mocowania kamery na dloni, dla jednego renderu."""

    dpos: np.ndarray                   # (3,) m, added to the position in the hand frame
    rotvec: np.ndarray                 # (3,) rad, applied as R_mj @ exp(rotvec) (camera frame)
    dfovy_deg: float = 0.0


@dataclass(frozen=True)
class Light:
    """Swiatlo podane w calosci, dla sceny, ktora ma wygladac jak inna (scena treningu studenta
    chwytu). `pos` jest nad blatem jak w krotkiej formie; `dir` idzie bez zmian."""

    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    diffuse: float = 0.7
    directional: bool = False
    castshadow: bool = True


@dataclass
class SceneConfig:
    robot: RobotSpec
    table: Table = field(default_factory=Table)
    cameras: list[CameraView] = field(default_factory=list)
    #: Karta kalibracyjna w szczekach: geometria i PRAWDZIWA poza w ukladzie TCP.
    card: Card | None = None
    card_pose: np.ndarray | None = None
    #: Niewidzialna, pogrubiona bryla kolizyjna wokol karty - dla sprawdzacza
    #: kolizji, zeby fala trzymala prawdziwy odstep, a nie zera milimetrow.
    card_collider: float | None = None
    objects: list[Box] = field(default_factory=list)
    #: Obiekty, dla ktorych scena dostaje czujniki kontaktu z kazda ze szczek -
    #: `<obiekt>_jaw0` (stala) i `<obiekt>_jaw1` (ruchoma), 1 gdy sie stykaja.
    grasp_sensors: list[str] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    #: Mrowka RealAnt blizniaka (`twin.ant`, prefiks `ant/`); None = bez mrowki.
    ant: AntPlacement | None = None
    #: Tylko symulacja: mrowka udajaca prawdziwa (prefiks `antprawda/`, z tagami, na wlasnej polityce).
    ant_standin: AntPlacement | None = None
    #: Oswietlenie: lista (pozycja swiatla nad blatem, jasnosc) albo `Light`.
    lights: list[tuple[tuple[float, float, float], float] | Light] = field(
        default_factory=lambda: [((0.4, -0.6, 1.6), 0.55), ((-0.5, 0.3, 1.4), 0.35)]
    )
    #: Headlight (ambient, diffuse, specular); None keeps the twin's own.
    headlight: tuple[float, float, float] | None = None
    #: Cameras on the hand; never in `cameras`, which holds only cameras fixed to the base.
    hand_cameras: list[HandCamera] = field(default_factory=list)


class Scene:
    """Skompilowana scena: model, stan, kinematyka ramienia i render z kamer."""

    def __init__(self, cfg: SceneConfig, model: mujoco.MjModel):
        self.cfg = cfg
        self.model = model
        self.data = mujoco.MjData(model)
        self.kin = RobotKinematics(cfg.robot, model, PREFIX)
        spec = cfg.robot
        self.act_ids = np.array([model.actuator(PREFIX + j).id for j in spec.joints])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        mujoco.mj_forward(model, self.data)
        base = model.body(PREFIX + spec.base_body).id
        self.T_base2world = pose(self.data.xmat[base].reshape(3, 3), self.data.xpos[base])
        for ant in self.ants().values():
            self.place_ant(prefix=ant.prefix)

    # ------------------------------------------------------------ ramie
    def set_joints(self, joints: dict[str, float], *, hold: bool = True) -> None:
        """Ustawia poze ramienia natychmiast (bez dynamiki) i - domyslnie - trzyma ja serwami."""
        q = self.kin.to_q(joints)
        self.kin.write_qpos(self.data.qpos, q)
        self.kin.stop_qvel(self.data.qvel)
        if hold:
            self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def command(self, joints: dict[str, float]) -> None:
        """Zadaje cel serwom - ramie dojedzie tam w kolejnych krokach fizyki."""
        self.data.ctrl[self.act_ids] = self.kin.to_q(joints)

    def joints(self) -> dict[str, float]:
        """Zmierzone katy stawow w jednostkach aplikacji."""
        return self.kin.from_q(self.data.qpos[self.kin.qadr])

    # ------------------------------------------------------------ mrowka
    def ants(self) -> dict[str, AntPlacement]:
        """{prefiks: mrowka} w tej scenie."""
        return {a.prefix: a for a in (self.cfg.ant, self.cfg.ant_standin) if a is not None}

    def _ant(self, prefix: str) -> AntPlacement:
        ant = self.ants().get(prefix)
        if ant is None:
            raise RuntimeError(f"scena bez mrowki {prefix!r}")
        return ant

    def place_ant(self, T_ant2base: np.ndarray | None = None, joints: dict[str, float] | None = None,
                  prefix: str = ANT_PREFIX) -> None:
        """Mrowka w nowej pozie i/albo z nowymi katami nog [st.] - natychmiast, bez dynamiki.

        Bez argumentow: to, co mowi jej `AntPlacement` (start z pliku stanowiska).
        """
        ant = self._ant(prefix)
        if T_ant2base is not None:
            ant.T_ant2base = np.asarray(T_ant2base, float)
        if joints is not None:
            ant.joints = {**ant.joints, **{k: float(v) for k, v in joints.items()}}
        m, d = self.model, self.data
        T = self.T_base2world @ ant.T_ant2base @ ANT_TORSO_IN_ANT
        if ant.free:
            j = m.joint(prefix + "root")
            qa, va = m.jnt_qposadr[j.id], m.jnt_dofadr[j.id]
            d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = T[:3, 3], _quat(T[:3, :3])
            d.qvel[va:va + 6] = 0.0
        else:
            k = int(m.body_mocapid[m.body(prefix + "torso").id])
            d.mocap_pos[k] = T[:3, 3]
            d.mocap_quat[k] = _quat(T[:3, :3])
        for name, deg in ant.joints.items():
            j = m.joint(prefix + name)
            a = m.actuator(prefix + name).id
            q = float(np.clip(np.radians(deg), *m.jnt_range[j.id]))
            d.qpos[m.jnt_qposadr[j.id]] = q
            d.qvel[m.jnt_dofadr[j.id]] = 0.0
            d.ctrl[a] = float(np.clip(q, *m.actuator_ctrlrange[a]))
        mujoco.mj_forward(m, d)

    def ant_joints(self, prefix: str = ANT_PREFIX) -> tuple[np.ndarray, np.ndarray]:
        """(katy [rad], predkosci [rad/s]) nog w kolejnosci `twin.ant.JOINTS`."""
        m, d = self.model, self.data
        ids = [m.joint(prefix + n).id for n in ANT_JOINTS]
        return (np.array([d.qpos[m.jnt_qposadr[j]] for j in ids]), np.array([d.qvel[m.jnt_dofadr[j]] for j in ids]))

    def command_ant(self, ctrl: np.ndarray, prefix: str = ANT_PREFIX) -> None:
        """Cel serw nog [rad] w kolejnosci `twin.ant.JOINTS` (obciety do zakresu serw modelu)."""
        m = self.model
        for name, q in zip(ANT_JOINTS, ctrl):
            a = m.actuator(prefix + name).id
            self.data.ctrl[a] = float(np.clip(q, *m.actuator_ctrlrange[a]))

    def ant_pose(self, prefix: str = ANT_PREFIX) -> np.ndarray | None:
        """Poza mrowki wzgledem podstawy wyczytana z MODELU (None bez tej mrowki)."""
        if prefix not in self.ants():
            return None
        b = self.model.body(prefix + "torso").id
        T = pose(self.data.xmat[b].reshape(3, 3), self.data.xpos[b])
        return inverse(self.T_base2world) @ T @ inverse(ANT_TORSO_IN_ANT)

    def set_ant_pose(self, T_ant2base: np.ndarray, prefix: str = ANT_PREFIX) -> None:
        """Nowa poza korpusu mrowki na wolnym stawie; predkosc liniowa obraca sie razem z nia (korekta z tagow)."""
        m, d = self.model, self.data
        j = m.joint(prefix + "root")
        qa, va = m.jnt_qposadr[j.id], m.jnt_dofadr[j.id]
        R_old = np.zeros(9)
        mujoco.mju_quat2Mat(R_old, d.qpos[qa + 3:qa + 7])
        T = self.T_base2world @ np.asarray(T_ant2base, float) @ ANT_TORSO_IN_ANT
        dR = T[:3, :3] @ R_old.reshape(3, 3).T
        d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = T[:3, 3], _quat(T[:3, :3])
        d.qvel[va:va + 3] = dR @ d.qvel[va:va + 3]
        mujoco.mj_kinematics(m, d)

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=n)
        # Pozy cial i site'ow po `mj_step` sa sprzed ostatniego podkroku, a katy (qpos) - po
        # nim. Kostka dla polityki sim, HUD i uchwyt TCP czytaja xpos/site_xpos: bez tego
        # rozjezdzaly sie z katami o ~1 mm przy niesieniu (K7, jak w env i batch).
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------ kamery
    def render(self, camera: str) -> np.ndarray:
        """Kadr RGB z kamery sceny w jej natywnej rozdzielczosci."""
        view = self.camera(camera)
        r = self._renderer(view.width, view.height)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = r
        return r

    def camera(self, name: str) -> CameraView:
        for view in self.cfg.cameras:
            if view.name == name:
                return view
        raise KeyError(f"brak kamery {name!r} w scenie")

    def hand_camera(self, name: str) -> HandCamera:
        for cam in self.cfg.hand_cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"brak kamery na dloni {name!r} w scenie")

    def render_hand(self, name: str, jitter: HandCameraJitter | None = None,
                    *, segmentation: bool = False) -> np.ndarray:
        """RGB uint8 (H, W, 3) from a hand camera, at the camera's own size.

        With `segmentation` the result is int32 (H, W) of geom ids, -1 where no geom is seen.
        A `jitter` moves the camera for this one render; the model keeps its nominal mount.
        The model is shared with the physics thread, whose `cam_xpos` can pick up the jittered
        mount during that render; nothing in the loop reads it, but a reader of the camera's
        pose there should take the model's nominal values.
        """
        cam = self.hand_camera(name)
        r = self._renderer(cam.width, cam.height)
        full = PREFIX + name
        m = self.model
        if jitter is None:
            # `step` reruns only mj_kinematics after mj_step: the bodies move to the new joints,
            # camera poses stay one substep behind (millimetres on a fast move).
            mujoco.mj_camlight(m, self.data)
            return self._read(r, full, segmentation)

        c = m.camera(full).id
        pos, quat, fovy = m.cam_pos[c].copy(), m.cam_quat[c].copy(), float(m.cam_fovy[c])
        try:
            m.cam_pos[c] = pos + np.asarray(jitter.dpos, float)
            m.cam_quat[c] = _quat(_rotmat(quat) @ _exp_rotvec(jitter.rotvec))
            m.cam_fovy[c] = fovy + float(jitter.dfovy_deg)
            # The camera's world pose in `data` is not recomputed from the model by the render:
            # without this a new cam_pos or cam_quat changes no pixel (measured: 10 cm of
            # cam_pos, identical image).
            mujoco.mj_camlight(m, self.data)
            return self._read(r, full, segmentation)
        finally:
            m.cam_pos[c], m.cam_quat[c], m.cam_fovy[c] = pos, quat, fovy
            mujoco.mj_camlight(m, self.data)

    def _read(self, r: mujoco.Renderer, camera: str, segmentation: bool) -> np.ndarray:
        if not segmentation:
            r.update_scene(self.data, camera=camera)
            return r.render().copy()
        # The renderer is shared with RGB renders of the same size, so it must leave this mode.
        r.enable_segmentation_rendering()
        try:
            r.update_scene(self.data, camera=camera)
            seg = r.render()
        finally:
            r.disable_segmentation_rendering()
        ids = seg[:, :, 0].astype(np.int32)
        ids[seg[:, :, 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return ids

    def camera_pose(self, name: str) -> np.ndarray:
        """Poza kamery wzgledem podstawy w konwencji OpenCV, wyczytana z MODELU.

        Do sprawdzenia, ze scena wstawila kamere tam, gdzie kazal config.
        """
        c = self.model.camera(name).id
        T_world = pose(self.data.cam_xmat[c].reshape(3, 3) @ CV_TO_MJ, self.data.cam_xpos[c])
        return inverse(self.T_base2world) @ T_world

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------ budowa
def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _add_tag_plate_mesh(spec: mujoco.MjSpec, plate: float, name: str = "tag_plate") -> None:
    """Kwadratowa plytka z UV, wspolna dla wszystkich tagow - jak w galaxeo."""
    m = spec.add_mesh()
    m.name = name
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    h = plate / 2
    m.uservert = np.array([-h, -h, 0, h, -h, 0, h, h, 0, -h, h, 0], np.float32)
    m.userface = np.array([0, 1, 2, 0, 2, 3], np.int32)
    m.usertexcoord = np.array([0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    m.userfacetexcoord = np.array([0, 1, 2, 0, 2, 3], np.int32)


def _add_tag_material(spec: mujoco.MjSpec, tag_id: int, px: int = 400) -> str:
    img = render_tag(tag_id, px)
    tex = spec.add_texture()
    tex.name = f"tag{tag_id}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
    tex.data = np.repeat(img[:, :, None], 3, axis=2).tobytes()
    mat = spec.add_material()
    mat.name = f"tag{tag_id}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{tag_id}"
    mat.texuniform = False
    return mat.name


def _add_card(spec: mujoco.MjSpec, robot: mujoco.MjSpec, rspec: RobotSpec, card: Card,
              T_card2tcp: np.ndarray, collider: float | None = None) -> None:
    """Karta w szczekach: widoczna plytka, obudowa kolizyjna i tagi na obu stronach."""
    site = robot.site(rspec.tcp_site)
    T_tcp2hand = pose(_rotmat(site.quat), np.asarray(site.pos, float))
    T = T_tcp2hand @ T_card2tcp
    Rc, tc = T[:3, :3], T[:3, 3]
    hand = robot.body(rspec.hand_body)

    # Srodek pudelka cofniety od srodka taga: tylko `out` karty wystaje za
    # szczeki, reszta jest miedzy nimi.
    centre = tc + Rc @ np.array([card.out / 2 - card.length / 2, 0.0, 0.0])
    half = np.array([card.length / 2, card.width / 2, card.thickness / 2])
    body = hand.add_geom()
    body.name = "calib_card"
    body.type = mujoco.mjtGeom.mjGEOM_BOX
    # Rysowana karta minimalnie ciensza od prawdziwej: tagi lezace dokladnie na
    # jej powierzchni migotalyby (z-fighting) i detektor widzialby bialy prostokat.
    body.size = half - np.array([0.0, 0.0, 0.0006])
    body.pos, body.quat = centre, _quat(Rc)
    body.rgba = [0.97, 0.97, 0.97, 1.0]
    body.contype = body.conaffinity = 0
    body.mass = 0.0
    body.group = 1

    if collider is not None:
        pad = hand.add_geom()
        pad.name = "calib_card_pad"
        pad.type = mujoco.mjtGeom.mjGEOM_BOX
        pad.size = half + collider
        pad.pos, pad.quat = centre, _quat(Rc)
        pad.rgba = [0.0, 0.0, 0.0, 0.0]
        pad.contype = pad.conaffinity = 1
        pad.mass = 0.0
        pad.group = 3

    _add_tag_plate_mesh(robot, card.plate)
    for tag_id, T_tag in card.tag_poses().items():
        _add_tag_material(robot, tag_id)
        g = hand.add_geom()
        g.name = f"tag{tag_id}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "tag_plate"
        g.material = f"tag{tag_id}"
        g.pos = tc + Rc @ T_tag[:3, 3]
        g.quat = _quat(Rc @ T_tag[:3, :3])
        g.contype = g.conaffinity = 0
        g.mass = 0.0
        g.group = 1


def _add_panel(spec: mujoco.MjSpec, panel: Panel, T_base2world: np.ndarray) -> None:
    img = np.asarray(panel.image, np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    tex = spec.add_texture()
    tex.name = f"panel_{panel.name}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
    tex.data = np.ascontiguousarray(img).tobytes()
    mat = spec.add_material()
    mat.name = f"panel_{panel.name}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.texuniform = False
    w, h = panel.size[0] / 2, panel.size[1] / 2
    mesh = spec.add_mesh()
    mesh.name = f"panel_{panel.name}"
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    mesh.uservert = np.array([-w, -h, 0, w, -h, 0, w, h, 0, -w, h, 0], np.float32)
    mesh.userface = np.array([0, 1, 2, 0, 2, 3], np.int32)
    # Wiersz 0 obrazu na gorze (+y) - tak jak tagi karty.
    mesh.usertexcoord = np.array([0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    mesh.userfacetexcoord = np.array([0, 1, 2, 0, 2, 3], np.int32)
    T = T_base2world @ np.asarray(panel.T_panel2base, float)
    g = spec.worldbody.add_geom()
    g.name = f"panel_{panel.name}"
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh.name
    g.material = mat.name
    g.pos, g.quat = T[:3, 3], _quat(T[:3, :3])
    g.contype = g.conaffinity = 0
    g.group = 1


#: Tektura pod tagami mrowki i jej maszt (tak sa naklejone na prawdziwej).
CARDBOARD_RGBA = [0.72, 0.58, 0.40, 1.0]


def _add_ant(spec: mujoco.MjSpec, ant: AntPlacement, T_base2world: np.ndarray) -> None:
    """Mrowka RealAnt z pliku modelu: korpus na wolnym stawie (albo mocap), nogi ze stawami i serwami, tagi.

    Z modelu wypadaja podloga, swiatla i niebo (`attach_body` bierze tylko drzewo korpusu) oraz wydrukowana
    plytka z tagami ArUco oryginalnej RealAnt - na tej mrowce sa AprilTagi 36h11 na tekturze, dodawane tu
    w nauczonym ukladzie. Schowana (`hidden`): bez tagow, geomy w grupie `HIDDEN_GROUP`, polprzezroczyste.
    """
    from .robots import resolve_asset

    child = mujoco.MjSpec.from_file(str(resolve_asset(ant.mjcf)))
    # Opcje fizyki zostaja z ramienia (scena je juz przejela) - bez tego `attach` ostrzega przy kazdej budowie.
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations"):
        setattr(child.option, name, getattr(spec.option, name))
    torso = child.body("torso")
    if not ant.free:
        for j in list(torso.joints):
            child.delete(j)
        torso.mocap = True
    for g in list(torso.geoms):
        if g.material == "ink" or g.meshname == "tag_plate":
            child.delete(g)
    # Tarcie stop jak w treningu (0,6): kontakt bierze parametry geomu o wyzszym priorytecie, a stol ma 1,0.
    stack = [child.worldbody]
    while stack:
        b = stack.pop()
        stack.extend(b.bodies)
        for g in b.geoms:
            if g.contype:
                g.priority = 1
            elif ant.hidden and g.group in (0, 1, 2):
                g.group = HIDDEN_GROUP
                g.material = ""
                g.rgba = list(GHOST_RGBA)
    T = T_base2world @ np.asarray(ant.T_ant2base, float) @ ANT_TORSO_IN_ANT
    torso.pos, torso.quat = T[:3, 3], _quat(T[:3, :3])

    to_torso = inverse(ANT_TORSO_IN_ANT)
    top = 0.0244                                         # gorna plyta korpusu nad srodkiem stosu plyt
    for tag_id, T_tag2ant, size in ([] if ant.hidden else ant.tags):
        plate = size / MARKER_FRACTION
        mesh = f"tag_plate{tag_id}"
        _add_tag_plate_mesh(child, plate, mesh)
        _add_tag_material(child, tag_id)
        Tt = to_torso @ np.asarray(T_tag2ant, float)
        R, p = Tt[:3, :3], Tt[:3, 3]
        g = torso.add_geom()
        g.name = f"tag{tag_id}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname, g.material = mesh, f"tag{tag_id}"
        g.pos, g.quat = p + R[:, 2] * 0.0003, _quat(R)   # 0,3 mm nad tektura - bez migotania
        g.contype = g.conaffinity = 0
        g.mass, g.group = 0.0, 1
        card = torso.add_geom()                          # tektura pod tagiem, 2 mm, z zapasem wokol
        card.name = f"tag{tag_id}_karton"
        card.type = mujoco.mjtGeom.mjGEOM_BOX
        card.size = [plate * 0.6, plate * 0.6, 0.001]
        card.pos, card.quat = p - R[:, 2] * 0.001, _quat(R)
        card.rgba = CARDBOARD_RGBA
        card.contype = card.conaffinity = 0
        card.mass, card.group = 0.0, 1
        # Tag stojacy (normalna w poziomie, w ukladzie mrowki) siedzi na maszcie z tektury: pasek od
        # gornej plyty do srodka taga, za nim.
        n_ant = np.asarray(T_tag2ant, float)[:3, 2]
        centre_ant = np.asarray(T_tag2ant, float)[:3, 3]
        if abs(n_ant[2]) < 0.5 and centre_ant[2] > top + 0.01:
            n = np.array([n_ant[0], n_ant[1], 0.0])
            n /= np.linalg.norm(n)
            side = np.cross([0.0, 0.0, 1.0], n)
            h = (centre_ant[2] - top) / 2
            c = centre_ant - n * 0.0035
            c[2] = top + h
            Rm = np.column_stack([side, -n, [0.0, 0.0, 1.0]])   # prawoskretny: z x (z x n) = -n
            Tm = to_torso @ pose(Rm, c)
            mast = torso.add_geom()
            mast.name = f"tag{tag_id}_maszt"
            mast.type = mujoco.mjtGeom.mjGEOM_BOX
            mast.size = [plate * 0.3, 0.0015, h]
            mast.pos, mast.quat = Tm[:3, 3], _quat(Tm[:3, :3])
            mast.rgba = CARDBOARD_RGBA
            mast.contype = mast.conaffinity = 0
            mast.mass, mast.group = 0.0, 1
    spec.worldbody.add_frame().attach_body(torso, ant.prefix, "")


def _rotmat(quat) -> np.ndarray:
    R = np.zeros(9)
    q = np.asarray(quat, float)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


def _exp_rotvec(rotvec) -> np.ndarray:
    v = np.asarray(rotvec, float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-15:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _add_hand_camera(robot: mujoco.MjSpec, rspec: RobotSpec, hc: HandCamera) -> None:
    T = np.asarray(hc.T_cam2hand, float)
    cam = robot.body(rspec.hand_body).add_camera()
    cam.name = hc.name
    cam.pos = T[:3, 3]
    cam.quat = _quat(T[:3, :3] @ CV_TO_MJ)
    # MuJoCo reads fovy in degrees whatever the compiler's angle setting.
    cam.fovy = float(hc.fovy_deg)


def build(cfg: SceneConfig) -> Scene:
    """Kompiluje scene od zera."""
    t = cfg.table
    top = t.height
    # Ramie jako pierwsze: jego opcje fizyki (menagerie stroi je pod manipulacje -
    # stozek eliptyczny, impratio, iteracje solvera) sa czescia opisu ramienia
    # i scena ma je przejac. Bez tego `attach` zostawia domyslne sceny i tylko
    # ostrzega, a chwyt zachowuje sie inaczej niz w modelu, ktory ktos dostroil.
    robot = mujoco.MjSpec.from_file(str(cfg.robot.mjcf_path))
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations"):
        setattr(spec.option, name, getattr(robot.option, name))
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    if cfg.headlight is not None:
        ambient, diffuse, specular = (float(v) for v in cfg.headlight)
        spec.visual.headlight.ambient = [ambient] * 3
        spec.visual.headlight.diffuse = [diffuse] * 3
        spec.visual.headlight.specular = [specular] * 3
    width = max([v.width for v in cfg.cameras] + [h.width for h in cfg.hand_cameras] + [640])
    height = max([v.height for v in cfg.cameras] + [h.height for h in cfg.hand_cameras] + [480])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height

    world = spec.worldbody
    floor = world.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3.0, 3.0, 0.05]
    floor.rgba = [0.32, 0.34, 0.37, 1.0]

    tab = world.add_geom()
    tab.name, tab.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    tab.size = [t.size[0] / 2, t.size[1] / 2, top / 2]
    tab.pos = [0.0, 0.0, top / 2]
    tab.rgba = list(t.rgba)

    for k, item in enumerate(cfg.lights):
        light = world.add_light()
        light.name = f"light{k}"
        if isinstance(item, Light):
            light.pos = [item.pos[0], item.pos[1], top + item.pos[2]]
            light.dir = np.asarray(item.dir, float) / np.linalg.norm(item.dir)
            light.diffuse = [item.diffuse] * 3
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if item.directional \
                else mujoco.mjtLightType.mjLIGHT_SPOT
            light.castshadow = item.castshadow
            continue
        pos, strength = item
        light.pos = [pos[0], pos[1], top + pos[2]]
        light.dir = -np.asarray([pos[0], pos[1], pos[2]], float) / np.linalg.norm(pos)
        light.diffuse = [strength] * 3
        light.castshadow = k == 0

    # Karta doklejana do dloni PRZED wstawieniem ramienia, potem attach pod prefiksem.
    if cfg.card is not None:
        from .calib.card import pinch_point

        T_card = cfg.card_pose
        if T_card is None:
            T_card = cfg.card.nominal(pinch_point(RobotKinematics(cfg.robot)), cfg.robot)
        _add_card(spec, robot, cfg.robot, cfg.card, T_card, cfg.card_collider)
    for hc in cfg.hand_cameras:
        _add_hand_camera(robot, cfg.robot, hc)

    c, s = np.cos(t.base_yaw), np.sin(t.base_yaw)
    R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T_base2world = pose(R_base, np.array([t.base_xy[0], t.base_xy[1], top]))
    mount = world.add_frame()
    mount.pos = T_base2world[:3, 3]
    mount.quat = _quat(R_base)
    spec.attach(robot, prefix=PREFIX, frame=mount)
    for ant in (cfg.ant, cfg.ant_standin):
        if ant is not None:
            _add_ant(spec, ant, T_base2world)

    for obj in cfg.objects:
        p = T_base2world @ np.array([obj.xy[0], obj.xy[1], obj.size[2], 1.0])
        body = world.add_body()
        body.name = obj.name
        body.pos = p[:3]
        if obj.mocap:
            body.mocap = True
        elif obj.free:
            body.add_freejoint()
        g = body.add_geom()
        g.name = obj.name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(obj.size)
        g.rgba = list(obj.rgba)
        g.mass = obj.mass
        if obj.mocap:
            g.contype = g.conaffinity = 0
            g.group = 1
            continue
        g.condim = 4
        g.friction = [1.0, 0.02, 0.002]

    for panel in cfg.panels:
        _add_panel(spec, panel, T_base2world)

    for obj in cfg.grasp_sensors:
        for k, jaw in enumerate(cfg.robot.jaw_bodies):
            s = spec.add_sensor()
            s.name = f"{obj}_jaw{k}"
            s.type = mujoco.mjtSensor.mjSENS_CONTACT
            s.objtype, s.objname = mujoco.mjtObj.mjOBJ_BODY, PREFIX + jaw
            s.reftype, s.refname = mujoco.mjtObj.mjOBJ_BODY, obj
            s.intprm = [1, 0, 1]                 # dane: "found"; bez redukcji; jeden kontakt

    for view in cfg.cameras:
        T_world = T_base2world @ np.asarray(view.T_cam2base, float)
        cam = world.add_camera()
        cam.name = view.name
        cam.pos = T_world[:3, 3]
        cam.quat = _quat(T_world[:3, :3] @ CV_TO_MJ)
        K = np.asarray(view.K, float)
        cam.resolution = [view.width, view.height]
        # Rozmiar matrycy jest tylko po to, zeby MuJoCo przeszlo z `fovy` na
        # model ogniskowej: przy ogniskowej w pikselach sam sie skraca.
        cam.sensor_size = [view.width * 1e-6, view.height * 1e-6]
        cam.focal_pixel = [K[0, 0], K[1, 1]]
        # Punkt glowny jako przesuniecie srodka obrazu wzgledem niego, w obu osiach
        # "srodek minus K". Srodek piksela (0, 0) to w OpenCV 0, wiec srodek
        # obrazu to (W - 1) / 2 i (H - 1) / 2. Znak ZMIERZONY na renderze, nie
        # wziety z dokumentacji: przy odwrotnym kadr przesuwal sie o dwukrotnosc
        # przesuniecia (62 px przy 30 px), zgodnie w calym kadrze.
        #
        # Obie osie sa symetryczne. Wczesniej w y bylo dodatkowe pol piksela,
        # zmierzone na srodkach CZERWONYCH plam - a dolna polowa kulki jest
        # w cieniu, odpada na progu koloru i podnosi srodek plamy o ~0,4 px.
        # Maska segmentacji (czysta geometria) pokazala, ze to pol piksela
        # przesuwalo caly kadr o 0,5 px w dol wzgledem rzutu K - to samo
        # w OpenGL i w rendererze MuJoCo Warp, ktory liczy promienie przez
        # srodki pikseli z tych samych intrynsyk. Przy f = 600 px to 0,05 st
        # pochylenia kamery w kalibracji.
        # Pilnuje tego `test_twin_scene.py` na maskach kulek w znanych punktach.
        cam.principal_pixel = [(view.width - 1) / 2 - K[0, 2], (view.height - 1) / 2 - K[1, 2]]

    return Scene(cfg, spec.compile())
