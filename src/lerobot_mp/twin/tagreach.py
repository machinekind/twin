"""Tag na biurku: poza z kamer w ukladzie podstawy i punkt dojazdu koncowki nad nim.

    fix = tag_from_frames(frames, ws.cameras, tag_id=3, tag_size=0.05)
    p = hover_point(fix.T_tag2base, above=0.08)          # 8 cm nad srodkiem taga, uklad podstawy

Kazda wlaczona kamera z zaufana poza (`T_cam2base`) i intrynsykami daje jedna
poze taga: `T_tag2base = T_cam2base @ T_tag2cam` (PnP z czterech rogow, jak
karta w fali kalibracyjnej). Kilka kamer skleja mediana odporna na zawiniecie
obrotu (`handeye._median_T`). Rozrzut miedzy kamerami to od razu sprawdzian
kalibracji: dobrze skalibrowane kamery widza ten sam tag w tym samym miejscu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .calib import tags
from .calib.handeye import _median_T
from .workspace import require_physical_cameras


@dataclass
class TagFix:
    """Poza taga w ukladzie podstawy z kilku kamer naraz."""

    T_tag2base: np.ndarray
    #: Poza z kazdej kamery, ktora tag widziala.
    per_camera: dict[str, np.ndarray] = field(default_factory=dict)
    #: RMS reprojekcji rogow [px] na kamere.
    reproj_px: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.T_tag2base[:3, 3].copy()

    @property
    def spread_mm(self) -> float:
        """Najwiekszy rozjazd polozenia taga miedzy kamerami [mm] (0 przy jednej)."""
        if len(self.per_camera) < 2:
            return 0.0
        ps = np.array([T[:3, 3] for T in self.per_camera.values()])
        return float(np.linalg.norm(ps - self.position, axis=1).max() * 1000.0)


def tag_from_frames(frames: Mapping[str, np.ndarray], cameras: Iterable, tag_id: int,
                    tag_size: float, *, physical: bool = False) -> TagFix | None:
    """Poza taga `tag_id` (bok czarnego kwadratu `tag_size` [m]) z kadrow zaufanych kamer.

    `cameras` to rekordy stanowiska (`CameraRecord`): liczy sie kamera wlaczona,
    z `T_cam2base` i `K`, dla ktorej jest kadr w `frames`. None, gdy zadna nie widzi taga.
    """
    per, rms = {}, {}
    for cam in cameras:
        img = frames.get(cam.name)
        if img is None or not cam.enabled or cam.T_cam2base is None or cam.K is None:
            continue
        if not cam.trusted and not cam.simulated:
            continue
        if physical:
            try:
                require_physical_cameras([cam], [cam.name])
            except RuntimeError:
                continue
        seen = tags.detect(img)
        if int(tag_id) not in seen:
            continue
        corners = seen[int(tag_id)]
        K = np.asarray(cam.K, float)
        dist = np.asarray(cam.dist, float) if cam.dist is not None else None
        T_tag2cam = tags.tag_pose(corners, tag_size, K, dist)
        pts_cam = (T_tag2cam[:3, :3] @ tags.corners_in_tag(tag_size).T).T + T_tag2cam[:3, 3]
        err = tags.project(pts_cam, K, dist) - np.asarray(corners, float)
        rms[cam.name] = float(np.sqrt((err ** 2).sum(axis=1).mean()))
        per[cam.name] = np.asarray(cam.T_cam2base, float) @ T_tag2cam
    if not per:
        return None
    return TagFix(_median_T(list(per.values())), per, rms)


def hover_point(T_tag2base: np.ndarray, above: float) -> np.ndarray:
    """Punkt `above` [m] nad srodkiem taga wzdluz osi z podstawy (tag lezy na blacie)."""
    p = np.asarray(T_tag2base, float)[:3, 3].copy()
    p[2] += float(above)
    return p
