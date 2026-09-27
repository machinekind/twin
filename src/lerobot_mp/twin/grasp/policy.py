"""The grasp student's checkpoint, and the policy that runs it on the CPU.

    path = find_checkpoint(a.checkpoint)    # --checkpoint, $LEROBOT_GRASP_STUDENT, workspace/policies/grasp/
    pol = GraspStudentPolicy.load(path)     # sha256 first, then a weights-only load, strict weights
    a = pol.act(image, obs, uv)             # uint8 (120, 160, 3), raw 19-vector, believed pixel -> (5,)

The checkpoint is never fetched by a run: `lerobot-twin grasp-model fetch` downloads it
over HTTPS with the standard library, checks its sha256 and renames it into place. Every
load checks the sha256 before anything is unpickled, and unpickles with
`weights_only=True`. Another checkpoint needs its hash in `LEROBOT_GRASP_SHA256`; only
`GraspStudentPolicy.load` reads that variable.

`act` is training's `Student._prep` + `Student.act` at batch 1: the image scaled to [0, 1],
the heat channel at the believed pixel appended, the 19-vector normalised with the
checkpoint's statistics in float32 without clipping. Nothing outside `act` normalises.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import logging
import os
import re
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ...paths import data_path
from .contract import (
    ACT_DIM,
    CAMERA_SPEC_URL,
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    FOVY_DEG,
    HEAT_SIGMA_FRAC,
    IMG_H,
    IMG_W,
    OBS_DIM,
    T_GRIPPER_CAMERA,
)
from .net import StudentNet

log = logging.getLogger(__name__)

CHECKPOINT_ENV = "LEROBOT_GRASP_STUDENT"
SHA256_ENV = "LEROBOT_GRASP_SHA256"
DEFAULT_CHECKPOINT = Path("workspace") / "policies" / "grasp" / "student.pt"
CAMERA_SPEC_NAME = "wrist_camera_spec.json"
CAMERA_SPEC_TOL = 1e-9
TIME_MEDIAN_MAX_MS = 30.0
TIME_P95_MAX_MS = 45.0

OBS_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gap x10",
             "believed-tcp x x10", "believed-tcp y x10", "believed-tcp z x10", "sin 2e", "cos 2e", "g_cmd",
             "prev x", "prev y", "prev z", "prev yaw", "prev grip", "attempts/3")

_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")


class CheckpointError(RuntimeError):
    """The checkpoint is missing, has another hash, or does not have the student's schema."""


class _Pinned(str):
    """`load`'s default hash. Only this object gives way to `$LEROBOT_GRASP_SHA256`: the same
    hash passed explicitly is the caller's choice and wins over the environment."""


_PINNED = _Pinned(CHECKPOINT_SHA256)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _hex(value: str, what: str) -> str:
    v = str(value).strip().lower()
    if not _HEX64.fullmatch(v):
        raise CheckpointError(f"{what}: {value!r} is not a sha256 (64 hex characters)")
    return v


def find_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """First existing file of: `explicit`, `$LEROBOT_GRASP_STUDENT`, `workspace/policies/grasp/student.pt`."""
    env = os.environ.get(CHECKPOINT_ENV, "").strip()
    for cand, where in ((explicit, "given"), (env or None, CHECKPOINT_ENV),
                        (data_path(DEFAULT_CHECKPOINT), "default")):
        if cand is None or str(cand) == "":
            continue
        p = Path(cand).expanduser()
        if p.is_file():
            return p
        if where != "default":
            log.warning("checkpoint %s (%s) does not exist; looking further", p, where)
    return None


def _download(url: str, dest: Path, *, sha256: str | None, timeout_s: float) -> Path:
    """Stream `url` to `<dest>.part`, compare the sha256 when one is given, rename into place."""
    if not url.lower().startswith("https://"):
        raise CheckpointError(f"refusing {url}: only https:// URLs are fetched")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            final = resp.geturl() if hasattr(resp, "geturl") else url
            if not str(final).lower().startswith("https://"):
                raise CheckpointError(f"refusing {url}: redirected to {final}, not https")
            with open(part, "wb") as f:
                while chunk := resp.read(_CHUNK):
                    h.update(chunk)
                    f.write(chunk)
                    n += len(chunk)
    except BaseException as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, (OSError, http.client.HTTPException)):
            raise CheckpointError(f"download of {url} failed: {exc}") from exc
        raise
    digest = h.hexdigest()
    if sha256 is not None and digest != sha256:
        part.unlink(missing_ok=True)
        raise CheckpointError(f"{url}: sha256 {digest} over {n} bytes, expected {sha256}; nothing was saved")
    os.replace(part, dest)
    log.info("saved %s (%d bytes, sha256 %s)", dest, n, digest)
    return dest


def fetch_checkpoint(dest: Path | None = None, *, url: str = CHECKPOINT_URL,
                     sha256: str = CHECKPOINT_SHA256, timeout_s: float = 60.0) -> Path:
    """Download the checkpoint over HTTPS; only `grasp-model fetch` calls this."""
    dest = Path(dest).expanduser() if dest is not None else data_path(DEFAULT_CHECKPOINT)
    return _download(url, dest, sha256=_hex(sha256, "sha256"), timeout_s=timeout_s)


def _safe_globals() -> list[Any]:
    import numpy

    return [numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype, type(numpy.dtype("float32"))]


def _check_schema(ckpt: Any, path: Path) -> None:
    if not isinstance(ckpt, Mapping):
        raise CheckpointError(f"{path}: not a checkpoint dict ({type(ckpt).__name__})")

    def need(key: str) -> Any:
        if key not in ckpt:
            raise CheckpointError(f"{path}: key {key} missing")
        return ckpt[key]

    if need("arch") != "resnet18":
        raise CheckpointError(f"{path}: arch {ckpt['arch']!r}, expected 'resnet18'")
    if need("heat") is not True:
        raise CheckpointError(f"{path}: heat {ckpt['heat']!r}, expected True")
    for key, want in (("obs_dim", OBS_DIM), ("act_dim", ACT_DIM)):
        if need(key) != want:
            raise CheckpointError(f"{path}: {key} {ckpt[key]!r}, expected {want}")
    if not isinstance(need("net"), Mapping):
        raise CheckpointError(f"{path}: net is {type(ckpt['net']).__name__}, not a state dict")
    for key in ("obs_mean", "obs_std"):
        try:
            arr = np.asarray(need(key), np.float64)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"{path}: {key} is not numeric: {exc}") from exc
        if arr.shape != (OBS_DIM,):
            raise CheckpointError(f"{path}: {key} has shape {arr.shape}, expected ({OBS_DIM},)")
        if not np.all(np.isfinite(arr)):
            raise CheckpointError(f"{path}: {key} is not finite")
        if key == "obs_std" and not np.all(arr > 0):
            raise CheckpointError(f"{path}: obs_std has entries <= 0")


def load_checkpoint(path: Path, *, sha256: str | None = CHECKPOINT_SHA256) -> dict:
    """Verify the sha256 (`None` skips it, with a warning), load weights-only, check the schema."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"{path}: no such file (lerobot-twin grasp-model fetch)")
    if sha256 is None:
        log.warning("%s: sha256 NOT verified (sha256=None)", path)
    else:
        want = _hex(sha256, "sha256")
        size = path.stat().st_size
        if want == CHECKPOINT_SHA256 and size != CHECKPOINT_BYTES:
            raise CheckpointError(f"{path}: {size} bytes, the published checkpoint has {CHECKPOINT_BYTES} "
                                  f"(incomplete copy or another file)")
        got = sha256_of(path)
        if got != want:
            raise CheckpointError(f"{path}: sha256 {got}, expected {want} (another checkpoint needs "
                                  f"{SHA256_ENV}=<its sha256>)")
    try:
        with torch.serialization.safe_globals(_safe_globals()):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"{path}: weights-only load failed: {exc}") from exc
    _check_schema(ckpt, path)
    return ckpt


def heat_channel(uv: torch.Tensor, H: int = IMG_H, W: int = IMG_W) -> torch.Tensor:
    """(B, 1, H, W) Gaussian blob at each (u, v), training's function.

    Sigma comes from the width. Only NaN gives a zero channel: an off-frame point keeps
    the tail of its blob inside the frame. Not clipped, not normalised, peak 1.0.
    """
    uv = torch.as_tensor(np.array(uv, np.float32))
    ys = torch.arange(H, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, W)
    u, v = uv[:, 0].view(-1, 1, 1), uv[:, 1].view(-1, 1, 1)
    s2 = 2 * (HEAT_SIGMA_FRAC * W) ** 2
    blob = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / s2)
    return torch.nan_to_num(blob, nan=0.0).unsqueeze(1)


class GraspStudentPolicy:
    """The student network with its observation statistics, at batch 1 on the CPU."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    path: Path

    def __init__(self, net: StudentNet, obs_mean: np.ndarray, obs_std: np.ndarray, path: str | Path, *,
                 sha256: str | None = None, schema: Mapping[str, Any] | None = None) -> None:
        self.net = net.eval()
        self.obs_mean = np.array(obs_mean, np.float32)
        self.obs_std = np.array(obs_std, np.float32)
        if self.obs_mean.shape != (OBS_DIM,) or self.obs_std.shape != (OBS_DIM,):
            raise ValueError(f"obs_mean / obs_std must be ({OBS_DIM},)")
        self.path = Path(path)
        self.sha256 = sha256
        self.schema = dict(schema or {})

    @classmethod
    def load(cls, path: str | Path | None = None, *, sha256: str | None = _PINNED,
             threads: int = 2) -> GraspStudentPolicy:
        """Load and verify the checkpoint; `path` None looks where `find_checkpoint` does.

        Left at its default, `sha256` is `$LEROBOT_GRASP_SHA256` when that is set, else the
        published hash. An explicit `sha256`, or None (no verification), is used as given.
        """
        found = Path(path).expanduser() if path is not None else find_checkpoint()
        if found is None:
            raise CheckpointError(f"no checkpoint: lerobot-twin grasp-model fetch (or --checkpoint PATH, "
                                  f"{CHECKPOINT_ENV})")
        if sha256 is _PINNED:
            env = os.environ.get(SHA256_ENV, "").strip()
            sha256 = _hex(env, SHA256_ENV) if env else CHECKPOINT_SHA256
        if sha256 is not None:
            sha256 = _hex(sha256, "sha256")
            if sha256 != CHECKPOINT_SHA256:
                log.warning("checkpoint %s is verified against sha256 %s, not the published %s",
                            found, sha256, CHECKPOINT_SHA256)
        ckpt = load_checkpoint(found, sha256=sha256)
        net = StudentNet(OBS_DIM, ACT_DIM)
        try:
            net.load_state_dict(ckpt["net"], strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointError(f"{found}: net does not fit the student network: {exc}") from exc
        torch.set_num_threads(threads)
        schema = {k: v for k, v in ckpt.items() if k not in ("net", "obs_mean", "obs_std")}
        return cls(net, ckpt["obs_mean"], ckpt["obs_std"], found, sha256=sha256, schema=schema)

    @staticmethod
    def _image(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype != np.uint8 or img.shape != (IMG_H, IMG_W, 3):
            raise ValueError(f"image must be uint8 ({IMG_H}, {IMG_W}, 3), got {img.dtype} {img.shape}")
        # A private writable copy: torch warns on read-only buffers, and a camera may reuse its own.
        return np.array(img, order="C", copy=True)

    @staticmethod
    def _uv(uv: np.ndarray) -> np.ndarray:
        u = np.asarray(uv)
        if u.shape != (2,) or u.dtype.kind not in "fiu":
            raise ValueError(f"uv must be a numeric (2,), got {u.dtype} {u.shape}")
        return u.astype(np.float32)

    @staticmethod
    def _stack(image: np.ndarray, uv: np.ndarray) -> torch.Tensor:
        # Same tensor operations as training's `_prep`, so the convolutions see the same memory layout.
        x = torch.as_tensor(image[None]).permute(0, 3, 1, 2).float() / 255.0
        return torch.cat([x, heat_channel(uv[None], x.shape[2], x.shape[3])], 1)

    def inputs(self, image: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """The (4, 120, 160) float32 image input `act` gives the network, for the dry run dump."""
        with torch.no_grad():
            return np.ascontiguousarray(self._stack(self._image(image), self._uv(uv))[0].numpy())

    def act(self, image: np.ndarray, obs: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """(5,) float32 action for the raw 19-vector; a non-finite output is returned as it is."""
        img = self._image(image)
        o = np.asarray(obs)
        if o.shape != (OBS_DIM,) or o.dtype.kind not in "fiu":
            raise ValueError(f"obs must be a numeric ({OBS_DIM},), got {o.dtype} {o.shape}")
        with np.errstate(over="ignore"):             # an overflow becomes inf and is refused just below
            o = o.astype(np.float32)
        if not np.all(np.isfinite(o)):
            raise ValueError(f"obs is not finite at {np.flatnonzero(~np.isfinite(o)).tolist()}")
        u = self._uv(uv)
        with torch.no_grad():
            x = self._stack(img, u)
            on = torch.as_tensor(((o - self.obs_mean) / self.obs_std)[None])
            a = self.net(x, on)
        return np.array(a[0].numpy(), np.float32)

    def time(self, n: int = 200) -> dict[str, float]:
        """Wall time of `act` on zeros: the first call, then median and p95 of the next `n`."""
        image = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        obs = np.zeros(OBS_DIM, np.float32)
        uv = np.zeros(2, np.float32)
        ms = []
        for _ in range(max(1, int(n)) + 1):
            t0 = perf_counter()
            self.act(image, obs, uv)
            ms.append((perf_counter() - t0) * 1e3)
        rest = np.asarray(ms[1:])
        return {"median_ms": float(np.median(rest)), "p95_ms": float(np.percentile(rest, 95)),
                "first_ms": float(ms[0])}


def camera_spec_differences(spec: Mapping[str, Any]) -> list[str]:
    """What in a published `wrist_camera_spec.json` differs from the contract (tolerance 1e-9)."""
    out = []
    try:
        T = np.asarray(spec["T_gripper_camera_opencv"], np.float64)
        if T.shape != (4, 4):
            out.append(f"T_gripper_camera_opencv has shape {T.shape}")
        else:
            d = float(np.abs(T - T_GRIPPER_CAMERA).max())
            if not d <= CAMERA_SPEC_TOL:
                out.append(f"T_gripper_camera_opencv differs by {d:.3g}")
    except (KeyError, TypeError, ValueError) as exc:
        out.append(f"T_gripper_camera_opencv unreadable: {exc!r}")
    try:
        fovy = float(spec["lens"]["recommended"]["fov_deg"][1])
        if not abs(fovy - FOVY_DEG) <= CAMERA_SPEC_TOL:
            out.append(f"lens.recommended.fov_deg[1] = {fovy}, contract {FOVY_DEG}")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        out.append(f"lens.recommended.fov_deg[1] unreadable: {exc!r}")
    return out


def _model_fetch(a: argparse.Namespace) -> int:
    given = getattr(a, "checkpoint", None)
    dest = Path(given).expanduser() if given else data_path(DEFAULT_CHECKPOINT)
    if dest.is_file():
        got = sha256_of(dest)
        if got != CHECKPOINT_SHA256:
            print(f"{dest} juz istnieje i ma inny sha256 ({got}) - nie nadpisuje.\n"
                  f"Usun go albo podaj inny plik: lerobot-twin grasp-model fetch --checkpoint <plik>",
                  file=sys.stderr)
            return 1
        print(f"Model juz jest: {dest} (sha256 zgodny) - nie pobieram drugi raz.")
    else:
        print(f"Pobieram {CHECKPOINT_URL}\n  -> {dest} ({CHECKPOINT_BYTES / 1e6:.1f} MB)", flush=True)
        try:
            fetch_checkpoint(dest)
        except CheckpointError as exc:
            print(f"Nie pobrano modelu: {exc}", file=sys.stderr)
            return 1
        print(f"Zapisano {dest}, sha256 zgodny.")
    spec_path = dest.with_name(CAMERA_SPEC_NAME)
    try:
        _download(CAMERA_SPEC_URL, spec_path, sha256=None, timeout_s=60.0)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (CheckpointError, OSError, ValueError) as exc:
        print(f"Nie pobrano opisu kamery {CAMERA_SPEC_URL}: {exc}", file=sys.stderr)
        return 1
    diffs = camera_spec_differences(spec)
    if diffs:
        bar = "!" * 72
        print(f"{bar}\nUWAGA: opublikowany opis kamery na nadgarstku ({spec_path}) ROZNI SIE od kontraktu\n"
              f"blizniaka (grasp/contract.py). Model mogl byc uczony z inna kamera - nie uruchamiaj go,\n"
              f"dopoki kontrakt nie zostanie sprawdzony:", file=sys.stderr)
        for d in diffs:
            print(f"  - {d}", file=sys.stderr)
        print(bar, file=sys.stderr)
        return 1
    print(f"Opis kamery ({spec_path}): poza i pole widzenia zgodne z kontraktem.")
    return 0


def _find_for_cli(a: argparse.Namespace) -> Path | None:
    path = find_checkpoint(getattr(a, "checkpoint", None))
    if path is None:
        print(f"Brak modelu. Pobierz: lerobot-twin grasp-model fetch  "
              f"(albo --checkpoint PATH, {CHECKPOINT_ENV})", file=sys.stderr)
    return path


def _load_for_cli(path: Path) -> GraspStudentPolicy | None:
    try:
        return GraspStudentPolicy.load(path)
    except CheckpointError as exc:
        print(f"Model odrzucony: {exc}", file=sys.stderr)
        return None


def _model_check(a: argparse.Namespace) -> int:
    path = _find_for_cli(a)
    if path is None:
        return 1
    digest = sha256_of(path)
    print(f"Model: {path.resolve()}")
    print(f"  rozmiar {path.stat().st_size} B, sha256 {digest}"
          f" ({'opublikowany' if digest == CHECKPOINT_SHA256 else 'INNY niz opublikowany'})")
    pol = _load_for_cli(path)
    if pol is None:
        return 1
    print(f"  wczytany bez wykonywania kodu (weights_only), {len(pol.net.state_dict())} wpisow wag, "
          f"dopasowanie scisle")
    for key in sorted(pol.schema):
        print(f"  {key}: {pol.schema[key]!r}")
    print("  statystyki obserwacji (z pliku):")
    for i, (name, m, s) in enumerate(zip(OBS_NAMES, pol.obs_mean, pol.obs_std)):
        print(f"    {i:2d} {name:<20s} srednia {m:+.6f}  odchylenie {s:.6f}")
    return 0


def _model_time(a: argparse.Namespace) -> int:
    path = _find_for_cli(a)
    pol = _load_for_cli(path) if path is not None else None
    if pol is None:
        return 1
    n = int(getattr(a, "n", 200) or 200)
    r = pol.time(n)
    ok = r["median_ms"] < TIME_MEDIAN_MAX_MS and r["p95_ms"] < TIME_P95_MAX_MS
    print(f"Inferencja na CPU ({torch.get_num_threads()} watki, {n} wywolan po pierwszym): "
          f"mediana {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, pierwsze {r['first_ms']:.1f} ms")
    limits = f"mediana < {TIME_MEDIAN_MAX_MS:.0f} ms, p95 < {TIME_P95_MAX_MS:.0f} ms"
    print(f"Werdykt: OK ({limits})" if ok else f"Werdykt: ZA WOLNO (wymagane: {limits})")
    return 0 if ok else 1


def main_model(a: argparse.Namespace) -> int:
    """`lerobot-twin grasp-model fetch|check|time`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    what = getattr(a, "action", None) or getattr(a, "what", None)
    if what == "fetch":
        return _model_fetch(a)
    if what == "check":
        return _model_check(a)
    if what == "time":
        return _model_time(a)
    print(f"lerobot-twin grasp-model: nieznane polecenie {what!r} (fetch, check, time)", file=sys.stderr)
    return 2
