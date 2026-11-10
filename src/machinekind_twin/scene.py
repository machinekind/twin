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
* Other robots of the station (`PassiveRobot`) come in under their own name as prefix
  (`<name>/<body>`), welded in place: the twin never drives them.
* Fixtures (`Fixture`, e.g. a camera mount) come in the same way (`<name>/<part>`): static bodies with
  visual meshes and box colliders that keep `PASSIVE_CLEARANCE`, like another robot.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np

from .calib.card import Card
from .calib.tags import MARKER_FRACTION
from .calib.tags import render as render_tag
from .kinematics import RobotKinematics, inverse, pose
from .robots import RobotCamera, RobotSpec

PREFIX = "robot/"
#: MuJoCo kamera patrzy wzdluz -z z y w gore; OpenCV wzdluz +z z y w dol.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
#: A passive robot's collision geoms report contacts this far out [m] but push only on touch
#: (margin = gap), so every collision check keeps this clearance from it.
PASSIVE_CLEARANCE = 0.02
#: Physics options the scene takes from the driven arm's model.
_OPTIONS = ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations")
#: Contact of the table top and of every object that collides: stiff, so a pressing gripper does not
#: push an object into the table or into another object. MuJoCo's default contact is softer.
OBJECT_SOLREF = (0.01, 1.0)
OBJECT_SOLIMP = (0.95, 0.995, 0.001, 0.5, 2.0)
#: Friction of the objects: sliding, torsional, rolling.
OBJECT_FRICTION = (1.0, 0.02, 0.002)


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
    #: Height of its bottom above the table top [m]: 0 rests on the table, the height of another
    #: box's top stacks it on that box.
    z: float = 0.0
    #: Turn about the vertical [rad], relative to the base; 0 leaves the box aligned with the world.
    yaw: float = 0.0


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


@dataclass(frozen=True)
class Light:
    """One light of the scene. `pos` is relative to the table top's centre; `dir` may be any length."""

    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    #: Grey: the same on all three channels.
    diffuse: float = 0.7
    #: True: a directional light (the sun), `pos` only places it; False: a spot light.
    directional: bool = False
    castshadow: bool = True

    def __post_init__(self) -> None:
        if not any(self.dir):
            raise ValueError("swiatlo: `dir` nie moze byc wektorem zerowym")


@dataclass(frozen=True)
class Headlight:
    """MuJoCo's light at the viewer, grey: each value on all three channels."""

    ambient: float = 0.35
    diffuse: float = 0.35
    #: MuJoCo's own default.
    specular: float = 0.5


@dataclass(frozen=True)
class Look:
    """How the scene is lit. The defaults are the twin's own look: a key light with shadows and a
    fill, both aimed at the table top's centre. Another look (a training scene's, so that a policy
    sees the pixels it learned on) is a workspace setting."""

    lights: tuple[Light, ...] = (
        Light(pos=(0.4, -0.6, 1.6), dir=(-0.4, 0.6, -1.6), diffuse=0.55, castshadow=True),
        Light(pos=(-0.5, 0.3, 1.4), dir=(0.5, -0.3, -1.4), diffuse=0.35, castshadow=False),
    )
    headlight: Headlight = Headlight()


@dataclass
class PassiveRobot:
    """Another robot of the station, one the twin does not drive: a full copy of its model.

    Its base stands at `T_base` in the driven arm's base frame. It has no actuators and physics never
    moves it, but contacts with it are real. `joints` [app units] is the pose it is built in; a live
    source moves it later (`Scene.set_passive_joints`).
    """

    name: str
    spec: RobotSpec
    T_base: np.ndarray
    joints: dict[str, float] = field(default_factory=dict)
    #: Where `Scene.set_passive_base` last put its base body, in its own fixed frame; None = as in the
    #: model file.
    base: np.ndarray | None = None

    @property
    def prefix(self) -> str:
        return self.name + "/"


@dataclass(frozen=True)
class Collider:
    """A box that stands for a fixture part in collision checks: centre and half sizes [m] in the part's frame."""

    pos: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass
class FixturePart:
    """One part of a fixture: an optional visual mesh in metres and its colliders, in the part's frame."""

    name: str
    #: The part's frame in the fixture's frame.
    T: np.ndarray
    mesh: Path | None = None
    colliders: dict[str, Collider] = field(default_factory=dict)
    rgba: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)


@dataclass
class Fixture:
    """A static body of the station that is not a robot, e.g. a camera mount. Its frame stands at `T_base`
    in the driven arm's base frame. Only the colliders collide; meshes cannot grow by a clearance."""

    name: str
    parts: list[FixturePart]
    T_base: np.ndarray

    @property
    def prefix(self) -> str:
        return self.name + "/"


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
    look: Look = field(default_factory=Look)
    #: Robots the twin does not drive (`PassiveRobot`); empty = the driven arm alone.
    passive: list[PassiveRobot] = field(default_factory=list)
    #: Static bodies of the station (`Fixture`); empty = none.
    fixtures: list[Fixture] = field(default_factory=list)


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
        self.passive = {p.name: p for p in cfg.passive}
        bodies = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        #: Geoms of the passive robots, and of every robot (driven and passive), per geom id.
        self.passive_geom = np.array([any(n.startswith(p.prefix) for p in cfg.passive) for n in bodies], bool)
        self.robot_geom = np.array([n.startswith(PREFIX) for n in bodies], bool) | self.passive_geom
        #: Geoms of the fixtures, and every geom each collision check keeps `PASSIVE_CLEARANCE` from.
        self.fixture_geom = np.array([any(n.startswith(f.prefix) for f in cfg.fixtures) for n in bodies],
                                     bool)
        self.kept_geom = self.passive_geom | self.fixture_geom

    # ------------------------------------------------------------ inne roboty
    def set_passive_joints(self, name: str, joints: Mapping[str, float]) -> None:
        """Moves the passive robot `name` to `joints` [app units]; no physics involved."""
        robot = self.passive[name]
        for body, (pos, quat) in passive_poses(robot.spec, joints).items():
            b = self.model.body(robot.prefix + body).id
            self.model.body_pos[b] = pos
            self.model.body_quat[b] = quat
        robot.joints = dict(joints)
        mujoco.mj_kinematics(self.model, self.data)

    def set_passive_base(self, name: str, T: np.ndarray) -> None:
        """Puts the passive robot `name`'s base body at `T` in its own fixed frame, in place of its
        pose in the model file (`RobotBackend.base_pose`); its limbs follow."""
        robot = self.passive[name]
        b = self.model.body(robot.prefix + robot.spec.base_body).id
        # `_add_passive`'s mount frame compiles into a top-level body's own pose: parent world, world
        # coordinates. A lower body's pose is relative to its parent, which stays where it is.
        if self.model.body_parentid[b] != 0:
            raise ValueError(f"{robot.spec.name}: podstawa {robot.spec.base_body!r} nie jest cialem "
                             "najwyzszego poziomu modelu - nie moze sie przesuwac")
        T_world = self.T_base2world @ robot.T_base @ T
        self.model.body_pos[b] = T_world[:3, 3]
        self.model.body_quat[b] = _quat(T_world[:3, :3])
        robot.base = T
        mujoco.mj_kinematics(self.model, self.data)

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

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=n)
        # Pozy cial i site'ow po `mj_step` sa sprzed ostatniego podkroku, a katy (qpos) - po
        # nim. Kostka dla polityki sim, HUD i uchwyt TCP czytaja xpos/site_xpos: bez tego
        # rozjezdzaly sie z katami o ~1 mm przy niesieniu (K7, jak w env i batch).
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------ kamery
    def render(self, camera: str) -> np.ndarray:
        """An RGB frame from a camera of the scene at its own size: a fixed one (`cfg.cameras`) or
        one on the arm (`RobotSpec.cameras`)."""
        if camera in self.cfg.robot.cameras:
            return self._render_robot_camera(camera)

        view = self.camera(camera)
        r = self._renderer(view.width, view.height)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def _render_robot_camera(self, name: str) -> np.ndarray:
        camera = self.cfg.robot.cameras[name]

        # `step` reruns only mj_kinematics after mj_step: the bodies follow the new joints, but
        # camera poses stay one substep behind (millimetres on a fast move) until mj_camlight.
        mujoco.mj_camlight(self.model, self.data)

        r = self._renderer(camera.width, camera.height)
        r.update_scene(self.data, camera=PREFIX + name)
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


def _add_robot_camera(robot: mujoco.MjSpec, name: str, camera: RobotCamera) -> None:
    T = np.asarray(camera.pose, float)
    cam = robot.body(camera.body).add_camera()
    cam.name = name
    cam.pos = T[:3, 3]
    cam.quat = _quat(T[:3, :3] @ CV_TO_MJ)
    # MuJoCo reads fovy in degrees whatever the compiler's angle setting.
    cam.fovy = float(camera.fovy_deg)


def _set_headlight(spec: mujoco.MjSpec, headlight: Headlight) -> None:
    spec.visual.headlight.ambient = [headlight.ambient] * 3
    spec.visual.headlight.diffuse = [headlight.diffuse] * 3
    spec.visual.headlight.specular = [headlight.specular] * 3


def _add_light(spec: mujoco.MjSpec, name: str, light: Light, top: float) -> None:
    mj = spec.worldbody.add_light()
    mj.name = name
    mj.pos = [light.pos[0], light.pos[1], top + light.pos[2]]
    mj.dir = np.asarray(light.dir, float) / np.linalg.norm(light.dir)
    mj.diffuse = [light.diffuse] * 3
    if light.directional:
        mj.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        mj.type = mujoco.mjtLightType.mjLIGHT_SPOT
    mj.castshadow = light.castshadow


#: Compiled models of robots alone, by MJCF path: forward kinematics of passive robots.
_ROBOT_MODELS: dict[str, mujoco.MjModel] = {}


def passive_poses(spec: RobotSpec, joints: Mapping[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pose (pos, quat) of each moving body of `spec` in its parent body at `joints` [app units]."""
    key = str(spec.mjcf_path)
    model = _ROBOT_MODELS.get(key)
    if model is None:
        model = _ROBOT_MODELS[key] = mujoco.MjModel.from_xml_path(key)
    kin = RobotKinematics(spec, model)
    kin.write_qpos(kin.data.qpos, kin.to_q(joints))
    mujoco.mj_kinematics(model, kin.data)
    d = kin.data
    out = {}
    for b in range(1, model.nbody):
        parent = int(model.body_parentid[b])
        if model.body_jntnum[b] == 0 or parent == 0:          # fixed, or a floating root: as mounted
            continue
        R = d.xmat[parent].reshape(3, 3)
        out[model.body(b).name] = (R.T @ (d.xpos[b] - d.xpos[parent]), _quat(R.T @ d.xmat[b].reshape(3, 3)))
    return out


def _check_passive(cfg: SceneConfig) -> None:
    """Passive robots and fixtures need names that are unique, plain and not the driven arm's prefix."""
    names = [p.name for p in cfg.passive] + [f.name for f in cfg.fixtures]
    bad = [n for n in names if not re.fullmatch(r"[A-Za-z0-9_-]+", n) or n + "/" == PREFIX]
    if bad:
        raise ValueError(f"nazwy robotow {bad}: tylko litery, cyfry, _ i -, inne niz {PREFIX[:-1]!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"nazwy robotow powtarzaja sie: {names}")


def _add_passive(spec: mujoco.MjSpec, other: PassiveRobot, T_base2world: np.ndarray, sticker: bool) -> None:
    """A copy of `other`'s model, welded at its pose: no joints, actuators, sensors or cameras. With
    `sticker`, its base tag on its base body (`RobotSpec.base_tag`)."""
    robot = mujoco.MjSpec.from_file(str(other.spec.mjcf_path))
    for body, (pos, quat) in passive_poses(other.spec, other.joints).items():
        b = robot.body(body)
        if b.frame is not None:
            raise ValueError(f"{other.spec.name}: ruchome cialo {body!r} lezy w <frame> - nieobslugiwane")
        b.pos, b.quat = pos, quat
        b.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    for group in (robot.actuators, robot.sensors, robot.tendons, robot.equalities, robot.keys,
                  robot.cameras, robot.lights, robot.joints):
        for element in list(group):
            robot.delete(element)
    for g in robot.geoms:
        if g.contype or g.conaffinity:
            g.margin = g.gap = PASSIVE_CLEARANCE
    for name in _OPTIONS:                                   # the scene keeps the driven arm's options
        setattr(robot.option, name, getattr(spec.option, name))
    tag = other.spec.base_tag
    if sticker and tag is not None:
        _add_tag_plate_mesh(robot, tag.size / MARKER_FRACTION)
        T_tag = np.asarray(tag.pose, float)
        g = robot.body(other.spec.base_body).add_geom()
        g.name = "base_tag"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "tag_plate"
        g.material = _add_tag_material(robot, tag.id)
        # Lying exactly on the body the plate would flicker (z-fighting) and a detector would see white.
        # The card draws its body thinner; a robot's body stays as modelled, so the sticker floats 0.5 mm out.
        g.pos = T_tag[:3, 3] + 0.0005 * T_tag[:3, 2]
        g.quat = _quat(T_tag[:3, :3])
        g.contype = g.conaffinity = 0
        g.mass = 0.0
        g.group = 1
    T = T_base2world @ np.asarray(other.T_base, float)
    mount = spec.worldbody.add_frame()
    mount.pos, mount.quat = T[:3, 3], _quat(T[:3, :3])
    spec.attach(robot, prefix=other.prefix, frame=mount)


def _add_fixture(spec: mujoco.MjSpec, fixture: Fixture, T_base2world: np.ndarray) -> None:
    """A static body per part: the mesh to look at, the colliders to keep clear of."""
    for part in fixture.parts:
        T = T_base2world @ np.asarray(fixture.T_base, float) @ np.asarray(part.T, float)
        body = spec.worldbody.add_body()
        body.name = fixture.prefix + part.name
        body.pos, body.quat = T[:3, 3], _quat(T[:3, :3])
        if part.mesh is not None:
            mesh = spec.add_mesh()
            mesh.name = body.name
            mesh.file = str(Path(part.mesh).resolve())
            g = body.add_geom()
            g.name = body.name
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = mesh.name
            g.rgba = list(part.rgba)
            g.contype = g.conaffinity = 0
            g.group = 1
        for name, box in part.colliders.items():
            g = body.add_geom()
            g.name = f"{body.name}/{name}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos, g.size = list(box.pos), list(box.size)
            # Seen only when the part has no mesh to show.
            g.rgba = list(part.rgba) if part.mesh is None else [0.0, 0.0, 0.0, 0.0]
            g.group = 1 if part.mesh is None else 3
            g.margin = g.gap = PASSIVE_CLEARANCE
            _stiff_contact(g)


def _check_camera_names(cfg: SceneConfig) -> None:
    """The scene renders cameras by name, so a fixed one may not take the name of one on the arm."""
    clash = sorted({view.name for view in cfg.cameras} & set(cfg.robot.cameras))
    if clash:
        raise ValueError(f"kamery stanowiska {clash} nazywaja sie jak kamery na ramieniu "
                         f"({cfg.robot.name}) - zmien ich nazwy")


def _stiff_contact(g: mujoco.MjsGeom) -> None:
    g.solref = list(OBJECT_SOLREF)
    g.solimp = list(OBJECT_SOLIMP)


def _rotmat(quat) -> np.ndarray:
    R = np.zeros(9)
    q = np.asarray(quat, float)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


def compose(cfg: SceneConfig, *, stickers: bool = True) -> mujoco.MjSpec:
    """Composes the scene from scratch as an `MjSpec`, without compiling it. `stickers=False` leaves out
    the other robots' base tags."""
    _check_camera_names(cfg)
    _check_passive(cfg)
    t = cfg.table
    top = t.height
    # Ramie jako pierwsze: jego opcje fizyki (menagerie stroi je pod manipulacje -
    # stozek eliptyczny, impratio, iteracje solvera) sa czescia opisu ramienia
    # i scena ma je przejac. Bez tego `attach` zostawia domyslne sceny i tylko
    # ostrzega, a chwyt zachowuje sie inaczej niz w modelu, ktory ktos dostroil.
    robot = mujoco.MjSpec.from_file(str(cfg.robot.mjcf_path))
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    for name in _OPTIONS:
        setattr(spec.option, name, getattr(robot.option, name))
    _set_headlight(spec, cfg.look.headlight)

    views = [*cfg.cameras, *cfg.robot.cameras.values()]
    spec.visual.global_.offwidth = max([v.width for v in views] + [640])
    spec.visual.global_.offheight = max([v.height for v in views] + [480])

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
    _stiff_contact(tab)

    for k, light in enumerate(cfg.look.lights):
        _add_light(spec, f"light{k}", light, top)

    # Karta doklejana do dloni PRZED wstawieniem ramienia, potem attach pod prefiksem.
    if cfg.card is not None:
        from .calib.card import pinch_point

        T_card = cfg.card_pose
        if T_card is None:
            T_card = cfg.card.nominal(pinch_point(RobotKinematics(cfg.robot)), cfg.robot)
        _add_card(spec, robot, cfg.robot, cfg.card, T_card, cfg.card_collider)
    for name, camera in cfg.robot.cameras.items():
        _add_robot_camera(robot, name, camera)

    c, s = np.cos(t.base_yaw), np.sin(t.base_yaw)
    R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T_base2world = pose(R_base, np.array([t.base_xy[0], t.base_xy[1], top]))
    mount = world.add_frame()
    mount.pos = T_base2world[:3, 3]
    mount.quat = _quat(R_base)
    spec.attach(robot, prefix=PREFIX, frame=mount)

    for obj in cfg.objects:
        p = T_base2world @ np.array([obj.xy[0], obj.xy[1], obj.z + obj.size[2], 1.0])
        body = world.add_body()
        body.name = obj.name
        body.pos = p[:3]
        if obj.yaw:
            c, s = np.cos(obj.yaw), np.sin(obj.yaw)
            body.quat = _quat(R_base @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
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
        g.friction = list(OBJECT_FRICTION)
        _stiff_contact(g)

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

    # Last, so every id of the scene without them stays as it was.
    for other in cfg.passive:
        _add_passive(spec, other, T_base2world, stickers)
    for fixture in cfg.fixtures:
        _add_fixture(spec, fixture, T_base2world)

    return spec


def build(cfg: SceneConfig) -> Scene:
    """Composes and compiles the scene from scratch."""
    return Scene(cfg, compose(cfg).compile())


def export(cfg: SceneConfig, folder: str | Path) -> Path:
    """Writes the scene as MJCF into `folder`: `scene.xml` and `assets/` with the arm's meshes.

    For packages that run on the measured station without importing the twin (training):
    `mujoco.MjModel.from_xml_path(folder / "scene.xml")` gives the model `build(cfg)` compiles,
    and `MjSpec.from_file` lets them add their task's objects. A scene with the calibration card
    or panels is refused: their textures live in memory, not in files. Returns the path of
    `scene.xml`.
    """
    if cfg.card is not None or cfg.panels:
        raise ValueError("eksport sceny: karta i panele nie wchodza do MJCF - zbuduj scene bez nich")
    folder = Path(folder)
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    def mesh_dir(spec: RobotSpec) -> Path:
        robot = mujoco.MjSpec.from_file(str(spec.mjcf_path))
        return Path(robot.modelfiledir) / robot.meshdir

    meshes = mesh_dir(cfg.robot)
    # A passive robot's or a fixture's meshes go to `assets/<name>/`: another may use the same file names.
    # A fixture's mesh files are absolute already.
    others = {p.prefix: (mesh_dir(p.spec), Path(p.name)) for p in cfg.passive}
    others.update({f.prefix: (Path(), Path(f.name)) for f in cfg.fixtures})
    # A base tag's texture lives in memory, like the card's: MJCF cannot hold it.
    spec = compose(cfg, stickers=False)
    for mesh in spec.meshes:
        source, sub = next((v for k, v in others.items() if mesh.name.startswith(k)), (meshes, Path()))
        (assets / sub).mkdir(exist_ok=True)
        shutil.copyfile(source / mesh.file, assets / sub / Path(mesh.file).name)
        # Absolute while `to_xml` compiles: an attached model looks for its files in its own folder,
        # where `assets/<name>/` does not exist. The XML gets them relative to `assets/` below.
        mesh.file = (assets.resolve() / sub / Path(mesh.file).name).as_posix()
    spec.modelfiledir = str(folder)
    spec.meshdir = "assets"
    absolute = quoteattr(assets.resolve().as_posix() + "/")[:-1]     # `file="/.../assets/`, as XML writes it
    xml = spec.to_xml().replace(f"file={absolute}", 'file="')
    path = folder / "scene.xml"
    path.write_text(xml, encoding="utf-8")
    return path
