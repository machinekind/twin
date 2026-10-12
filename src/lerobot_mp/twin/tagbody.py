"""Sztywna bryla z kilkoma tagami 36h11: uklad tagow nauczony z kadrow i poza bryly z kamer.

    body = TagBody("bryla", ids=(0, 1, 2), tag_size=0.020)
    obs = observe(frames, ws.cameras, body)               # [Observation] - jeden kadr kazdej kamery
    body.layout = learn_layout(obs_a + obs_b + ..., body).layout
    fix = locate(frames, ws.cameras, body)                # T_body2base z kamer, ktore maja poze

Uklad bryly to uklad PIERWSZEGO taga (`ids[0]`): srodek w srodku tego taga, x w prawo, y w
gore, z wzdluz jego normalnej (konwencja `calib.tags`). Gdzie na obudowie siedza pozostale
tagi, nikt nie mierzy linijka: bryla jest sztywna, wiec kazdy kadr, w ktorym jedna kamera
widzi dwa tagi naraz, daje ich wzajemna poze (`inv(T_a2cam) @ T_b2cam`), a mediana z kilku
kadrow i kamer to punkt startowy, a wszystkie rogi ze wszystkich kadrow naraz (uklad i poza
bryly w kazdym kadrze razem, Levenberg-Marquardt w numpy) - uklad (`learn_layout`). Sam PnP
jednego taga ma blad glebokosci rzedu procenta (tag 50 px, rog 0,3 px -> 4 mm na 0,4 m), ktory
mediana par przenosi na uklad; wspolne dopasowanie go zdejmuje, bo bryla jest jedna.
Do nauki wystarczy K kamery - poza kamery jest niepotrzebna, wiec uklad da sie zebrac jeszcze
przed kalibracja. Tag, ktorego nie widziano razem z zadnym juz znanym, zostaje bez pozy
i nie liczy sie w `locate`.

Poza bryly z jednej kamery: jeden PnP ze wszystkich rogow widzianych tagow (start z IPPE na
najwiekszym tagu, potem iteracyjnie po wszystkich rogach). Kilka kamer skleja mediana odporna
na zawiniecie obrotu (`handeye._median_T`); rozrzut miedzy kamerami to od razu sprawdzian
ich kalibracji - jak w `tagreach`.

Uwaga na identyfikatory: karta kalibracyjna ma tagi 0 i 1 (`Card.ids`). Te same numery na
bryle, gdy jest w kadrze podczas fali, sa dla sesji kalibracji nie do odroznienia
od karty (`tags.detect` odrzuca id widziane dwa razy w kadrze, ale tag z bryly widziany
sam za karte uchodzi). Bezpieczniej nakleic inne numery; `ids` sa dowolne.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .calib import tags
from .calib.handeye import _median_T
from .kinematics import inverse

#: pmd[vision] CamBoard nano: matryca 165x120 px, pole widzenia 90 x 69 stopni.
NANO_RESOLUTION = (165, 120)
NANO_FOV_DEG = (90.0, 69.0)

#: Gdzie patrzy kamera wzgledem taga na jej obudowie: os -> (z kamery, y kamery = "dol" obrazu)
#: w ukladzie taga. "-z": tag na tylnej sciance, kamera patrzy od niego; "+x"/"-x": tag na
#: bocznej sciance (naklejony prosto, y taga do gory); "+y"/"-y": tag na gornej sciance
#: (normalna taga to "gora" kamery, kamera patrzy wzdluz y taga).
OPTICAL_AXES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "-z": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "+x": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "-x": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
}


def optical_in_tag(axis: str) -> np.ndarray:
    """4x4 uklad optyczny kamery (OpenCV: z do przodu, y w dol) w ukladzie taga, srodek w srodku taga."""
    z, y = (np.asarray(v, float) for v in OPTICAL_AXES[axis])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    return T


def _T(x: Any) -> np.ndarray:
    T = np.asarray(x, float)
    if T.shape != (4, 4):
        raise ValueError(f"oczekiwano macierzy 4x4, jest {T.shape}")
    return T


@dataclass
class TagBody:
    """Sztywna bryla z tagami 36h11 tej samej wielkosci (np. kamera z tagami na obudowie)."""

    name: str = "pmd"
    ids: tuple[int, ...] = (0, 1, 2)
    #: Bok CZARNEGO kwadratu taga na obudowie [m] - zmierzyc linijka po naklejeniu.
    tag_size: float = 0.020
    #: {id taga: 4x4 T_tag2body}; tag `ids[0]` to jednostka. Uczone z kamer (`learn_layout`).
    layout: dict[int, np.ndarray] = field(default_factory=dict)
    #: (id taga, os z `OPTICAL_AXES`): gdzie patrzy kamera wzgledem tego taga; None = nieznane
    #: (rysuja sie same tagi, bez piramidy pola widzenia).
    optical: tuple[int, str] | None = None
    fov_deg: tuple[float, float] = NANO_FOV_DEG
    resolution: tuple[int, int] = NANO_RESOLUTION

    def __post_init__(self) -> None:
        self.ids = tuple(int(i) for i in self.ids)
        if not self.ids:
            raise ValueError("bryla bez tagow")
        self.layout = {int(i): _T(T) for i, T in self.layout.items()}
        self.layout.setdefault(self.ids[0], np.eye(4))

    @property
    def ref(self) -> int:
        return self.ids[0]

    def known(self) -> list[int]:
        """Tagi, ktore maja poze w ukladzie bryly (w kolejnosci `ids`)."""
        return [i for i in self.ids if i in self.layout]

    def T_optical2body(self) -> np.ndarray | None:
        if self.optical is None:
            return None
        tag_id, axis = int(self.optical[0]), str(self.optical[1])
        if tag_id not in self.layout:
            return None
        return self.layout[tag_id] @ optical_in_tag(axis)

    def corners_in_body(self, tag_id: int) -> np.ndarray:
        """(4, 3) rogi taga w ukladzie bryly, w kolejnosci `tags.corners_in_tag`."""
        T = self.layout[int(tag_id)]
        return (T[:3, :3] @ tags.corners_in_tag(self.tag_size).T).T + T[:3, 3]

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ids": list(self.ids), "tag_size": float(self.tag_size),
                "layout": {str(i): T.tolist() for i, T in self.layout.items()},
                "optical": None if self.optical is None else [int(self.optical[0]), str(self.optical[1])],
                "fov_deg": list(self.fov_deg), "resolution": list(self.resolution)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TagBody:
        d = dict(data)
        d.pop("T_body2base", None)                        # ostatnia poza zapisana obok, nie czesc bryly
        layout = {int(i): _T(T) for i, T in dict(d.pop("layout", {}) or {}).items()}
        opt = d.pop("optical", None)
        return cls(layout=layout, optical=None if opt is None else (int(opt[0]), str(opt[1])),
                   ids=tuple(d.pop("ids", (0, 1, 2))), fov_deg=tuple(d.pop("fov_deg", NANO_FOV_DEG)),
                   resolution=tuple(d.pop("resolution", NANO_RESOLUTION)), **d)


# ------------------------------------------------------------------ obserwacje
def _camera_intrinsics(cam) -> tuple[np.ndarray, np.ndarray | None] | None:
    if cam.K is None:
        return None
    K = np.asarray(cam.K, float)
    dist = np.asarray(cam.dist, float) if cam.dist is not None else None
    return K, dist


@dataclass
class Observation:
    """Tagi bryly w jednym kadrze jednej kamery: rogi, K i poza kazdego taga z osobna (IPPE)."""

    camera: str
    K: np.ndarray
    dist: np.ndarray | None
    corners: dict[int, np.ndarray]
    poses: dict[int, np.ndarray]

    @property
    def ids(self) -> list[int]:
        return sorted(self.corners)


def observe(frames: Mapping[str, np.ndarray], cameras: Iterable, body: TagBody) -> list[Observation]:
    """Tagi bryly widziane w tym kadrze przez kazda WLACZONA kamere z K - po jednej obserwacji na kamere.

    Poza kamery jest tu niepotrzebna (uklad tagow liczy sie wzgledem siebie), wiec nauka ukladu
    idzie takze z kamer jeszcze nieskalibrowanych - byle mialy K z szachownicy.
    """
    out: list[Observation] = []
    for cam in cameras:
        img = frames.get(cam.name)
        if img is None or not cam.enabled:
            continue
        intr = _camera_intrinsics(cam)
        if intr is None:
            continue
        K, dist = intr
        seen = tags.detect(img)
        corners = {i: np.asarray(seen[i], float) for i in body.ids if i in seen}
        if corners:
            poses = {i: tags.tag_pose(c, body.tag_size, K, dist) for i, c in corners.items()}
            out.append(Observation(cam.name, K, dist, corners, poses))
    return out


@dataclass
class LayoutFit:
    #: {id: T_tag2body}; tylko tagi polaczone z `ref` lancuchem wspolnych kadrow.
    layout: dict[int, np.ndarray]
    #: Ile kadrow widzialo dana pare tagow naraz.
    pairs: dict[tuple[int, int], int] = field(default_factory=dict)
    #: Rozrzut wzajemnej pozy pary miedzy kadrami [mm] (0 przy jednym kadrze).
    spread_mm: dict[tuple[int, int], float] = field(default_factory=dict)

    def missing(self, ids: Iterable[int]) -> list[int]:
        return [int(i) for i in ids if int(i) not in self.layout]


def learn_layout(observations: Iterable[Observation], body: TagBody, refine: bool = True) -> LayoutFit:
    """Uklad tagow bryly z obserwacji (kazda z jednego kadru jednej kamery).

    Para tagow widziana w jednym kadrze daje `T_b2a = inv(T_a2cam) @ T_b2cam`; z kilku kadrow
    bierze sie mediane. Uklad = uklad taga `ids[0]`; pozostale dochodza lancuchem przez pary
    (tag 2 widziany tylko z tagiem 1 dostaje poze przez tag 1). Potem, gdy `refine`, wspolne
    dopasowanie ukladu i pozy bryly w kazdym kadrze do wszystkich rogow (`refine_layout`).
    """
    observations = list(observations)
    ids = [int(i) for i in body.ids]
    ref = ids[0]
    samples: dict[tuple[int, int], list[np.ndarray]] = {}
    for obs in observations:
        seen = [i for i in ids if i in obs.poses]
        for a, b in combinations(seen, 2):
            samples.setdefault((a, b), []).append(inverse(_T(obs.poses[a])) @ _T(obs.poses[b]))
    rel = {k: _median_T(v) for k, v in samples.items()}
    spread = {}
    for k, v in samples.items():
        p = np.array([T[:3, 3] for T in v])
        spread[k] = float(np.linalg.norm(p - rel[k][:3, 3], axis=1).max() * 1000.0) if len(v) > 1 else 0.0
    layout = {ref: np.eye(4)}
    frontier = [ref]
    while frontier:
        a = frontier.pop()
        for (p, q), T_q2p in rel.items():
            if p == a and q not in layout:
                layout[q] = layout[a] @ T_q2p
                frontier.append(q)
            elif q == a and p not in layout:
                layout[p] = layout[a] @ inverse(T_q2p)
                frontier.append(p)
    if refine and len(layout) > 1:
        layout = refine_layout(observations, TagBody(body.name, tuple(ids), body.tag_size, layout))
    return LayoutFit(layout, {k: len(v) for k, v in samples.items()}, spread)


def _rodrigues(v: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))
    return R


def refine_layout(observations: list[Observation], body: TagBody, iters: int = 30) -> dict[int, np.ndarray]:
    """Uklad tagow (poza `ref`) i poza bryly w kazdym kadrze dopasowane naraz do wszystkich rogow.

    Levenberg-Marquardt na malym zadaniu (6 parametrow na tag, 6 na kadr) bez scipy. Jakobian
    z roznic centralnych, ale po blokach: parametry kadru wplywaja tylko na jego rogi, wiec
    liczy sie je z samego tego kadru (30 kadrow to ~50 ms na iteracje, nie sekundy). Poprawki
    obrotow sa lokalne (`R0 @ exp(d)`), wiec zadna poza nie siedzi blisko osobliwosci wektora
    obrotu. Zwraca nowy slownik ukladu (`ref` = jednostka).
    """
    ids = [i for i in body.known() if i != body.ref]
    frames = [o for o in observations if any(i in o.corners for i in body.known())]
    if not ids or not frames:
        return dict(body.layout)
    n_tag, n_fr = len(ids), len(frames)
    lay0 = [body.layout[i].copy() for i in ids]                              # T_tag2body
    cam0 = [body_pose_in_camera(o.corners, body, o.K, o.dist)[0] for o in frames]  # T_body2cam
    corners0 = tags.corners_in_tag(body.tag_size)
    per_frame = [[i for i in body.known() if i in o.corners] for o in frames]

    def bump(T0: np.ndarray, d: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ _rodrigues(d[:3])
        T[:3, 3] = T0[:3, 3] + d[3:]
        return T

    def frame_res(k: int, lay: dict[int, np.ndarray], T_b2c: np.ndarray) -> np.ndarray:
        o = frames[k]
        out = []
        for i in per_frame[k]:
            T = T_b2c @ lay[i]
            pts = (T[:3, :3] @ corners0.T).T + T[:3, 3]
            out.append((tags.project(pts, o.K, o.dist) - o.corners[i]).ravel())
        return np.concatenate(out)

    def unpack(x: np.ndarray):
        lay = {body.ref: np.eye(4), **{i: bump(lay0[k], x[6 * k:6 * k + 6]) for k, i in enumerate(ids)}}
        cams = [bump(cam0[k], x[6 * (n_tag + k):6 * (n_tag + k) + 6]) for k in range(n_fr)]
        return lay, cams

    def residuals(x: np.ndarray) -> np.ndarray:
        lay, cams = unpack(x)
        return np.concatenate([frame_res(k, lay, cams[k]) for k in range(n_fr)])

    sizes = [8 * len(per_frame[k]) for k in range(n_fr)]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    h = 1e-6
    x = np.zeros(6 * (n_tag + n_fr))
    lam = 1e-3
    r = residuals(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.zeros((r.size, x.size))
        lay, cams = unpack(x)
        for j in range(6 * n_tag):                       # uklad: kazdy kadr, w ktorym ten tag jest
            e = np.zeros_like(x)
            e[j] = h
            lp, _ = unpack(x + e)
            lm, _ = unpack(x - e)
            tag = ids[j // 6]
            for k in range(n_fr):
                if tag in per_frame[k]:
                    J[offs[k]:offs[k + 1], j] = (frame_res(k, lp, cams[k]) - frame_res(k, lm, cams[k])) / (2 * h)
        for k in range(n_fr):                            # poza kadru: tylko jego rogi
            for jj in range(6):
                d = np.zeros(6)
                d[jj] = h
                col = 6 * (n_tag + k) + jj
                J[offs[k]:offs[k + 1], col] = (frame_res(k, lay, bump(cams[k], d))
                                               - frame_res(k, lay, bump(cams[k], -d))) / (2 * h)
        A, g = J.T @ J, J.T @ r
        improved, step = False, np.zeros_like(x)
        for _try in range(8):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = residuals(x + step)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                x, r, cost, lam = x + step, r_new, c_new, max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved:
            break
        # Poprawki wchodza do punktu startowego - obrot znow liczony lokalnie od zera.
        lay, cams = unpack(x)
        lay0, cam0 = [lay[i] for i in ids], cams
        x = np.zeros_like(x)
        if np.abs(step).max() < 1e-9:
            break
    lay, _ = unpack(x)
    return lay


# ------------------------------------------------------------------ poza bryly
def body_pose_in_camera(seen: Mapping[int, np.ndarray], body: TagBody, K: np.ndarray,
                        dist: np.ndarray | None) -> tuple[np.ndarray, float, list[int]] | None:
    """(T_body2cam, RMS reprojekcji [px], uzyte tagi) z rogow tagow o znanym ukladzie; None, gdy zaden."""
    used = [i for i in body.known() if i in seen]
    if not used:
        return None
    dist_ = np.zeros(5) if dist is None else np.asarray(dist, float)
    # Start: IPPE na najwiekszym w kadrze tagu (jak `tagreach`), potem wszystkie rogi naraz.
    biggest = max(used, key=lambda i: cv2.contourArea(np.asarray(seen[i], np.float32)))
    T0 = tags.tag_pose(seen[biggest], body.tag_size, K, dist_) @ inverse(body.layout[biggest])
    obj = np.vstack([body.corners_in_body(i) for i in used])
    img = np.vstack([np.asarray(seen[i], float) for i in used])
    if len(used) > 1:
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T0[:3, :3]))
        tvec = T0[:3, 3].reshape(3, 1).copy()
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist_, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.ravel()
    else:
        T = T0
    pts_cam = (T[:3, :3] @ obj.T).T + T[:3, 3]
    err = tags.project(pts_cam, K, dist_) - img
    return T, float(np.sqrt((err ** 2).sum(axis=1).mean())), used


@dataclass
class BodyFix:
    """Poza bryly w ukladzie podstawy z kilku kamer naraz."""

    T_body2base: np.ndarray
    per_camera: dict[str, np.ndarray] = field(default_factory=dict)
    reproj_px: dict[str, float] = field(default_factory=dict)
    tags_seen: dict[str, list[int]] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.T_body2base[:3, 3].copy()

    @property
    def spread_mm(self) -> float:
        """Najwiekszy rozjazd polozenia bryly miedzy kamerami [mm] (0 przy jednej)."""
        if len(self.per_camera) < 2:
            return 0.0
        ps = np.array([T[:3, 3] for T in self.per_camera.values()])
        return float(np.linalg.norm(ps - self.position, axis=1).max() * 1000.0)


def locate(frames: Mapping[str, np.ndarray], cameras: Iterable, body: TagBody) -> BodyFix | None:
    """Poza bryly w ukladzie podstawy z kadrow kamer, ktore maja poze (`T_cam2base`) i K.

    None, gdy zadna taka kamera nie widzi zadnego taga o znanym ukladzie.
    """
    per, rms, used = {}, {}, {}
    for cam in cameras:
        img = frames.get(cam.name)
        if img is None or not cam.enabled or cam.T_cam2base is None:
            continue
        intr = _camera_intrinsics(cam)
        if intr is None:
            continue
        K, dist = intr
        res = body_pose_in_camera(tags.detect(img), body, K, dist)
        if res is None:
            continue
        T_body2cam, rms[cam.name], used[cam.name] = res
        per[cam.name] = np.asarray(cam.T_cam2base, float) @ T_body2cam
    if not per:
        return None
    return BodyFix(_median_T(list(per.values())), per, rms, used)
