"""Independent SO-101 reference from Articulus, for `test_twin_robots.py`.

Copied from the 3D preview of the deleted hand-control app (`lerobot_mp/preview/model.py`).
`przemeknowak781/articulus` rebuilds the SO-101 from the vendor's STEP solids
and URDF; `scripts/import_articulus_model.py` turns its export into `assets/so101_preview.npz`.
The chain is stored as `pre @ motion @ post` steps rather than derived again as formulas, so the
test compares the MJCF with the source geometry and not with a second derivation of it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Default location of the asset, relative to the repository root.
DEFAULT_ASSET = "assets/so101_preview.npz"


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """4x4 rotation matrix about an axis through the origin."""
    matrix = np.eye(4, dtype=np.float64)
    if abs(angle_rad) < 1e-12:
        return matrix

    a = axis / (np.linalg.norm(axis) or 1.0)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = a
    # Rodrigues' formula.
    matrix[:3, :3] = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    return matrix


@dataclass
class ArmModel:
    """Meshes and kinematic chain of the real SO-101."""

    link_names: list[str]
    link_colors: np.ndarray          # (L, 3) uint8, RGB
    vertices: np.ndarray             # (N, 3) float32, metres, link-local frame
    faces: np.ndarray                # (M, 3) int32
    face_link: np.ndarray            # (M,)   int16
    vertex_link: np.ndarray          # (N,)   int16
    chain_link: np.ndarray           # (K,)   int16
    chain_parent: np.ndarray         # (K,)   int16, -1 = root
    chain_joint: list[str]
    chain_axis: np.ndarray           # (K, 3)
    chain_pre: np.ndarray            # (K, 4, 4)
    chain_post: np.ndarray           # (K, 4, 4)
    dof_names: list[str]
    dof_min: np.ndarray
    dof_max: np.ndarray
    source: dict
    reference: dict

    @classmethod
    def load(cls, path: str | Path) -> "ArmModel":
        data = np.load(Path(path), allow_pickle=True)

        faces = data["faces"].astype(np.int32)
        face_link = data["face_link"].astype(np.int16)
        vertices = data["vertices"].astype(np.float32)

        # Every vertex belongs to exactly one link (the meshes are joined separately), so
        # assigning from the triangles is unambiguous.
        vertex_link = np.zeros(len(vertices), dtype=np.int16)
        vertex_link[faces.reshape(-1)] = np.repeat(face_link, 3)

        model = cls(
            link_names=[str(n) for n in data["link_names"]],
            link_colors=data["link_colors"].astype(np.uint8),
            vertices=vertices,
            faces=faces,
            face_link=face_link,
            vertex_link=vertex_link,
            chain_link=data["chain_link"].astype(np.int16),
            chain_parent=data["chain_parent"].astype(np.int16),
            chain_joint=[str(j) for j in data["chain_joint"]],
            chain_axis=data["chain_axis"].astype(np.float64),
            chain_pre=data["chain_pre"].astype(np.float64),
            chain_post=data["chain_post"].astype(np.float64),
            dof_names=[str(n) for n in data["dof_names"]],
            dof_min=data["dof_min"].astype(np.float64),
            dof_max=data["dof_max"].astype(np.float64),
            source=json.loads(str(data["source"])),
            reference=json.loads(str(data["reference"])) if "reference" in data else {},
        )
        model._validate()
        return model

    def _validate(self) -> None:
        """The chain must be topologically sorted: parent before child."""
        seen: set[int] = set()
        for step_index, parent in enumerate(self.chain_parent):
            if parent >= 0 and int(parent) not in seen:
                raise ValueError(
                    f"Lancuch nie jest posortowany topologicznie przy kroku {step_index} "
                    f"({self.link_names[int(self.chain_link[step_index])]}) - "
                    "rodzic wystepuje po dziecku."
                )
            seen.add(int(self.chain_link[step_index]))

    def link_transforms(self, joints_deg: dict[str, float]) -> np.ndarray:
        """World<-link transforms (L, 4, 4) for the given angles in degrees."""
        world = np.repeat(np.eye(4, dtype=np.float64)[None], len(self.link_names), axis=0)

        for step in range(len(self.chain_link)):
            link = int(self.chain_link[step])
            parent = int(self.chain_parent[step])
            base = world[parent] if parent >= 0 else np.eye(4)

            joint = self.chain_joint[step]
            angle = 0.0
            if joint:
                angle = np.radians(float(joints_deg.get(joint, 0.0)))
            move = _rotation(self.chain_axis[step], angle) if joint else np.eye(4)

            world[link] = base @ self.chain_pre[step] @ move @ self.chain_post[step]
        return world

    def from_lerobot(self, joints: dict[str, float]) -> dict[str, float]:
        """Converts LeRobot joint positions to model angles [degrees].

        LeRobot already gives the arm joints in degrees of the same calibration, so they pass
        unchanged. The gripper is the exception: LeRobot always normalizes it to 0..100, while
        the model has a real angular range, so it is mapped linearly using the limits stored
        in the model.
        """
        out: dict[str, float] = {}
        limits = dict(zip(self.dof_names, zip(self.dof_min, self.dof_max)))

        for name, value in joints.items():
            if name not in limits:
                continue
            low, high = limits[name]
            if name == "gripper":
                fraction = min(max(float(value) / 100.0, 0.0), 1.0)
                out[name] = float(low + fraction * (high - low))
            else:
                out[name] = float(min(max(value, low), high))
        return out

    @property
    def title(self) -> str:
        return str(self.source.get("title") or "SO-101")


def load_model(path: str | Path | None = None) -> ArmModel | None:
    """Load the reference model; None when the asset is missing (the test then skips)."""
    candidate = Path(path) if path else Path(DEFAULT_ASSET)
    if not candidate.is_absolute():
        # Also relative to the repository root, one level above tests/.
        for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
            resolved = root / candidate
            if resolved.is_file():
                candidate = resolved
                break

    if not candidate.is_file():
        logger.info(
            "Brak modelu 3D (%s) - uzywam podgladu schematycznego. "
            "Wygeneruj go: python scripts/import_articulus_model.py --articulus ../articulus",
            candidate,
        )
        return None

    try:
        model = ArmModel.load(candidate)
    except Exception:
        logger.exception("Nie udalo sie wczytac modelu 3D z %s", candidate)
        return None

    logger.info(
        "Model 3D: %s (%d czlonow, %d trojkatow, zrodlo: %s)",
        model.title,
        len(model.link_names),
        len(model.faces),
        model.source.get("origin", "?"),
    )
    return model
