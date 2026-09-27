"""Grasp student: the network rebuilt without torchvision, the heat channel, the checkpoint and its fetch.

The recorded fixtures come from the training environment (`scripts/record_grasp_fixtures.py`).
Tests that need the weights look where `find_checkpoint` looks (`LEROBOT_GRASP_STUDENT` for
local runs) and skip without a checkpoint. Nothing here touches the network: the fetch tests
use a local plain-HTTP server and a patched opener.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot_mp.twin.grasp import contract as C  # noqa: E402
from lerobot_mp.twin.grasp import policy as P  # noqa: E402
from lerobot_mp.twin.grasp.net import StudentNet  # noqa: E402
from lerobot_mp.twin.grasp.policy import (  # noqa: E402
    CheckpointError,
    GraspStudentPolicy,
    fetch_checkpoint,
    find_checkpoint,
    heat_channel,
    load_checkpoint,
    sha256_of,
)

FIX = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
CKPT = find_checkpoint()
RERECORD = "re-record the fixtures (docs/GRASP.md)"


def fixture(name: str, *keys: str) -> dict[str, np.ndarray]:
    path = FIX / name
    if not path.exists():
        pytest.fail(f"{path} is missing: {RERECORD}")
    with np.load(path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            pytest.fail(f"{path} lacks {', '.join(missing)}: {RERECORD}")
        return {k: z[k] for k in keys}


def dtype_name(d: object) -> str:
    return str(d).removeprefix("torch.")


def flip_one_weight_byte(src: Path, dst: Path) -> None:
    """Copy `src` with one bit changed in the middle of its largest tensor, so the copy still loads."""
    shutil.copyfile(src, dst)
    with zipfile.ZipFile(dst) as z:
        info = max(z.infolist(), key=lambda i: i.file_size)
    assert info.compress_type == zipfile.ZIP_STORED
    with open(dst, "r+b") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        name_len, extra_len = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
        start = info.header_offset + 30 + name_len + extra_len
        at = start + info.file_size // 2
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0x01]))


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
    if CKPT is None:
        pytest.skip("no checkpoint: lerobot-twin grasp-model fetch, or LEROBOT_GRASP_STUDENT")
    return CKPT


def _refuse_network(*args: object, **kw: object) -> None:
    raise AssertionError("the network was opened")


@pytest.fixture(scope="module")
def policy(ckpt_path: Path) -> GraspStudentPolicy:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(P.SHA256_ENV, raising=False)
        mp.setattr(P.urllib.request, "urlopen", _refuse_network)
        return GraspStudentPolicy.load(ckpt_path)


@pytest.fixture(scope="module")
def changed_copy(ckpt_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    dst = tmp_path_factory.mktemp("changed") / "student.pt"
    flip_one_weight_byte(ckpt_path, dst)
    return dst


@pytest.fixture
def toy_policy() -> GraspStudentPolicy:
    torch.manual_seed(0)
    return GraspStudentPolicy(StudentNet(), np.zeros(C.OBS_DIM, np.float32),
                              np.full(C.OBS_DIM, 0.5, np.float32), "toy.pt")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P.urllib.request, "urlopen", _refuse_network)


# ---------------------------------------------------------------- network, no checkpoint

def test_the_rebuilt_network_has_the_recorded_state_dict_names_shapes_and_dtypes() -> None:
    path = FIX / "state_keys.json"
    if not path.exists():
        pytest.fail(f"{path} is missing: {RERECORD}")
    want = json.loads(path.read_text())
    sd = StudentNet().state_dict()
    assert len(sd) == 130
    assert [row[0] for row in want] == list(sd)
    for name, shape, dtype in want:
        assert list(sd[name].shape) == list(shape), name
        assert dtype_name(sd[name].dtype) == dtype_name(dtype), name


def test_the_network_uses_exact_gelu_default_batch_norm_and_imagenet_statistics() -> None:
    net = StudentNet()
    assert net.prop[1].approximate == "none"
    assert all(m.approximate == "none" for m in net.head if isinstance(m, torch.nn.GELU))
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(m.eps == 1e-5 and m.momentum == 0.1 and m.affine for m in bns)
    assert net.mean.shape == net.std.shape == (1, 3, 1, 1)
    assert torch.equal(net.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.equal(net.std.flatten(), torch.tensor([0.229, 0.224, 0.225]))
    for i in (2, 3, 8, 9):
        assert not list(net.trunk[i].parameters())


def test_importing_the_network_and_the_policy_asks_for_neither_torchvision_nor_the_hub_client() -> None:
    # A fresh interpreter: in this one the modules are already imported. The finder sees every
    # attempt, also one that a try/except would hide or that fails because the package is absent.
    code = textwrap.dedent("""
        import sys
        asked = []
        class Spy:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in ("torchvision", "huggingface_hub"):
                    asked.append(name)
                return None
        sys.meta_path.insert(0, Spy())
        import lerobot_mp.twin.grasp.net, lerobot_mp.twin.grasp.policy
        print(asked, [m for m in sys.modules if m.partition(".")[0] in ("torchvision", "huggingface_hub")])
    """)
    src = Path(P.__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[] []"


def test_the_network_maps_a_batch_to_five_actions_in_the_tanh_range() -> None:
    net = StudentNet().eval()
    torch.manual_seed(1)
    img, obs = torch.rand(2, 4, C.IMG_H, C.IMG_W), torch.randn(2, C.OBS_DIM)
    with torch.no_grad():
        a = net(img, obs)
    assert a.shape == (2, C.ACT_DIM) and a.dtype == torch.float32
    assert bool(torch.all(a.abs() <= 1.0))


def test_the_heat_channel_equals_the_training_function() -> None:
    h = fixture("heat.npz", "uv", "heat")
    got = heat_channel(h["uv"]).numpy()
    assert got.shape == h["heat"].shape and got.dtype == np.float32
    assert float(np.abs(got - h["heat"]).max()) <= 1e-6
    for (u, v), blob in zip(h["uv"], got):
        if np.isnan(u) or np.isnan(v):
            assert not blob.any()
        elif not (0 <= u < C.IMG_W and 0 <= v < C.IMG_H):
            assert 0.0 < float(blob.max()) < 1.0, (u, v)


def test_the_heat_sigma_comes_from_the_width() -> None:
    blob = heat_channel(np.array([[80.0, 60.0]], np.float32))[0, 0].numpy()
    sigma = C.HEAT_SIGMA_FRAC * C.IMG_W
    assert blob[60, 80] == 1.0
    assert blob[60, 90] == pytest.approx(np.exp(-100.0 / (2 * sigma ** 2)), rel=1e-6)
    assert blob[70, 80] == pytest.approx(np.exp(-100.0 / (2 * sigma ** 2)), rel=1e-6)


# ---------------------------------------------------------------- act and inputs, no checkpoint

def test_act_refuses_a_wrong_image_or_observation(toy_policy: GraspStudentPolicy) -> None:
    img = np.zeros((C.IMG_H, C.IMG_W, 3), np.uint8)
    obs = np.zeros(C.OBS_DIM, np.float32)
    uv = np.array([40.0, 70.0], np.float32)
    for bad in (img.astype(np.float32), np.zeros((C.IMG_W, C.IMG_H, 3), np.uint8),
                np.zeros((C.IMG_H, C.IMG_W, 4), np.uint8), img[None]):
        with pytest.raises(ValueError):
            toy_policy.act(bad, obs, uv)
    for bad_obs in (np.zeros(18, np.float32), np.zeros((1, C.OBS_DIM), np.float32)):
        with pytest.raises(ValueError):
            toy_policy.act(img, bad_obs, uv)
    for value in (np.nan, np.inf, -np.inf, 1e39):
        o = obs.astype(np.float64)
        o[4] = value
        with pytest.raises(ValueError, match="not finite"):
            toy_policy.act(img, o, uv)
    with pytest.raises(ValueError):
        toy_policy.act(img, obs, np.zeros(3, np.float32))


def test_a_nan_pixel_is_legal_and_gives_an_empty_heat_channel(toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(0).integers(0, 256, (C.IMG_H, C.IMG_W, 3), dtype=np.uint8)
    a = toy_policy.act(img, np.zeros(C.OBS_DIM), np.array([np.nan, np.nan]))
    assert a.shape == (C.ACT_DIM,) and a.dtype == np.float32 and np.all(np.isfinite(a))
    x = toy_policy.inputs(img, np.array([np.nan, np.nan]))
    assert not x[3].any()


def test_inputs_are_the_scaled_image_and_the_heat_channel(toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(1).integers(0, 256, (C.IMG_H, C.IMG_W, 3), dtype=np.uint8)
    img.setflags(write=False)
    uv = np.array([-12.0, 30.0])
    x = toy_policy.inputs(img, uv)
    assert x.shape == (4, C.IMG_H, C.IMG_W) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:3], img.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0))
    np.testing.assert_array_equal(x[3], heat_channel(uv[None])[0, 0].numpy())
    assert x[3].max() > 0.0


def test_act_normalises_with_the_checkpoint_statistics_without_clipping(
        toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(2).integers(0, 256, (C.IMG_H, C.IMG_W, 3), dtype=np.uint8)
    uv = np.array([44.6, 75.8], np.float32)
    obs = np.full(C.OBS_DIM, 50.0)
    a = toy_policy.act(img, obs, uv)
    x = torch.as_tensor(toy_policy.inputs(img, uv)[None])
    with torch.no_grad():
        want = toy_policy.net(x, torch.full((1, C.OBS_DIM), 100.0)).numpy()[0]
        clipped = toy_policy.net(x, torch.full((1, C.OBS_DIM), 5.0)).numpy()[0]
    np.testing.assert_allclose(a, want, rtol=0, atol=1e-6)
    assert np.abs(a - clipped).max() > 1e-3


def test_act_runs_without_gradients_and_changes_neither_the_network_nor_its_inputs(
        toy_policy: GraspStudentPolicy) -> None:
    grad = []
    toy_policy.net.register_forward_hook(lambda m, i, o: grad.append(torch.is_grad_enabled()))
    before = {k: v.clone() for k, v in toy_policy.net.state_dict().items()}
    img = np.random.default_rng(3).integers(0, 256, (C.IMG_H, C.IMG_W, 3), dtype=np.uint8)
    obs = np.random.default_rng(4).normal(size=C.OBS_DIM).astype(np.float32)
    uv = np.array([44.6, 75.8], np.float32)
    kept = img.copy(), obs.copy(), uv.copy()
    a1 = toy_policy.act(img, obs, uv)
    a2 = toy_policy.act(img, obs, uv)
    assert grad == [False, False]
    np.testing.assert_array_equal(a1, a2)
    for name, value in toy_policy.net.state_dict().items():
        assert torch.equal(value, before[name]), name        # BatchNorm statistics: eval mode
    for arr, orig in zip((img, obs, uv), kept):
        np.testing.assert_array_equal(arr, orig)


def test_a_non_finite_network_output_is_returned_as_it_is(toy_policy: GraspStudentPolicy) -> None:
    with torch.no_grad():
        toy_policy.net.head[4].bias[:] = float("nan")
    a = toy_policy.act(np.zeros((C.IMG_H, C.IMG_W, 3), np.uint8), np.zeros(C.OBS_DIM), np.zeros(2))
    assert a.shape == (C.ACT_DIM,) and np.isnan(a).all()


# ---------------------------------------------------------------- finding the checkpoint

def test_find_checkpoint_takes_explicit_alone_else_the_first_existing_of_environment_and_default(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit, env = tmp_path / "explicit.pt", tmp_path / "env.pt"
    monkeypatch.setattr(P, "data_path", lambda rel: tmp_path / "data" / rel)
    monkeypatch.delenv(P.CHECKPOINT_ENV, raising=False)
    assert find_checkpoint() is None
    assert find_checkpoint(explicit) is None
    placed = tmp_path / "data" / P.DEFAULT_CHECKPOINT
    placed.parent.mkdir(parents=True)
    placed.write_bytes(b"d")
    assert find_checkpoint() == placed
    env.write_bytes(b"e")
    monkeypatch.setenv(P.CHECKPOINT_ENV, str(env))
    assert find_checkpoint() == env
    assert find_checkpoint(explicit) is None
    explicit.write_bytes(b"x")
    assert find_checkpoint(explicit) == explicit
    assert find_checkpoint(str(explicit)) == explicit


def test_load_without_any_checkpoint_names_the_fetch_command_and_opens_no_connection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None) -> None:
    monkeypatch.setattr(P, "data_path", lambda rel: tmp_path / rel)
    monkeypatch.delenv(P.CHECKPOINT_ENV, raising=False)
    with pytest.raises(CheckpointError, match="grasp-model fetch"):
        GraspStudentPolicy.load()
    with pytest.raises(CheckpointError, match="grasp-model fetch"):
        load_checkpoint(tmp_path / "nothing.pt")


# ---------------------------------------------------------------- the checkpoint

def test_the_checkpoint_has_the_published_sha256_and_size(ckpt_path: Path) -> None:
    assert ckpt_path.stat().st_size == C.CHECKPOINT_BYTES
    assert sha256_of(ckpt_path) == C.CHECKPOINT_SHA256


def test_the_checkpoint_loads_weights_only_with_the_student_schema(
        ckpt_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = torch.load

    def spy(*args: object, **kw: object) -> object:
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(torch, "load", spy)
    ckpt = load_checkpoint(ckpt_path)
    assert calls and all(c.get("weights_only") is True and c.get("map_location") == "cpu" for c in calls)
    assert ckpt["arch"] == "resnet18" and ckpt["heat"] is True
    assert ckpt["obs_dim"] == C.OBS_DIM and ckpt["act_dim"] == C.ACT_DIM
    assert len(ckpt["net"]) == 130
    for key in ("obs_mean", "obs_std"):
        assert ckpt[key].shape == (C.OBS_DIM,) and ckpt[key].dtype == np.float32
    assert np.all(ckpt["obs_std"] > 0)


def test_the_checkpoint_weights_load_strictly_into_the_rebuilt_network(ckpt_path: Path) -> None:
    res = StudentNet().load_state_dict(load_checkpoint(ckpt_path)["net"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def test_load_gives_an_eval_network_on_the_cpu_with_the_asked_threads_and_the_file_statistics(
        ckpt_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None) -> None:
    monkeypatch.delenv(P.SHA256_ENV, raising=False)
    threads = torch.get_num_threads()
    try:
        pol = GraspStudentPolicy.load(ckpt_path, threads=1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)
    assert pol.path == ckpt_path and pol.sha256 == C.CHECKPOINT_SHA256
    assert not any(m.training for m in pol.net.modules())
    assert all(t.device.type == "cpu" for t in pol.net.state_dict().values())
    ckpt = load_checkpoint(ckpt_path)
    for got, key in ((pol.obs_mean, "obs_mean"), (pol.obs_std, "obs_std")):
        assert got.dtype == np.float32 and got.shape == (C.OBS_DIM,)
        np.testing.assert_array_equal(got, ckpt[key])
    for name, value in ckpt["net"].items():
        assert torch.equal(pol.net.state_dict()[name], value), name


def test_a_corrupted_copy_is_refused_before_anything_is_unpickled(
        ckpt_path: Path, changed_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: object, **kw: object) -> None:
        raise AssertionError("torch.load ran on an unverified file")

    monkeypatch.setattr(torch, "load", never)
    monkeypatch.setenv(P.SHA256_ENV, sha256_of(changed_copy))   # load_checkpoint never reads it
    with pytest.raises(CheckpointError, match="sha256"):
        load_checkpoint(changed_copy)
    short = tmp_path / "short.pt"
    with open(ckpt_path, "rb") as f:
        short.write_bytes(f.read(C.CHECKPOINT_BYTES // 2))
    with pytest.raises(CheckpointError, match=f"{C.CHECKPOINT_BYTES}"):
        load_checkpoint(short)


def test_a_file_that_is_no_checkpoint_is_refused_by_the_weights_only_load(tmp_path: Path) -> None:
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="weights-only load failed"):
        load_checkpoint(junk, sha256=hashlib.sha256(b"not a checkpoint").hexdigest())


def test_a_checkpoint_with_another_schema_is_refused_naming_the_key(ckpt_path: Path, tmp_path: Path) -> None:
    good = {**load_checkpoint(ckpt_path), "net": {}}      # small files: the schema check needs no weights

    def saved(ckpt: dict, name: str) -> Path:
        path = tmp_path / name
        torch.save(ckpt, path)
        return path

    path = saved(good, "good.pt")
    assert load_checkpoint(path, sha256=sha256_of(path))["arch"] == "resnet18"
    for i, (key, value) in enumerate((("arch", "cnn"), ("heat", False), ("heat", 1), ("obs_dim", 18),
                                      ("act_dim", 4), ("net", None), ("obs_mean", np.zeros(18, np.float32)),
                                      ("obs_mean", np.full(19, np.nan, np.float32)),
                                      ("obs_std", np.zeros(19, np.float32)))):
        path = saved({**good, key: value}, f"bad{i}.pt")
        with pytest.raises(CheckpointError, match=key):
            load_checkpoint(path, sha256=sha256_of(path))
    path = saved({k: v for k, v in good.items() if k != "obs_std"}, "missing.pt")
    with pytest.raises(CheckpointError, match="obs_std missing"):
        load_checkpoint(path, sha256=sha256_of(path))


def test_the_policy_gives_the_recorded_actions_at_batch_1_from_the_recorded_inputs(
        policy: GraspStudentPolicy) -> None:
    s = fixture("shard.npz", "img", "obs", "uv", "act")
    assert len(s["act"]) == 59
    assert np.isfinite(s["uv"]).all(axis=1).any()
    got = np.stack([policy.act(s["img"][i], s["obs"][i], s["uv"][i]) for i in range(len(s["act"]))])
    assert got.dtype == np.float32 and got.shape == s["act"].shape
    err = float(np.abs(got - s["act"]).max())
    print(f"max |act - recorded| over {len(got)} ticks at batch 1: {err:.3g}")
    assert err <= 1e-5


def test_lerobot_grasp_sha256_lets_load_verify_a_changed_copy_and_without_it_the_copy_is_refused(
        ckpt_path: Path, changed_copy: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    digest = sha256_of(changed_copy)
    assert digest != C.CHECKPOINT_SHA256
    monkeypatch.delenv(P.SHA256_ENV, raising=False)
    with pytest.raises(CheckpointError, match=C.CHECKPOINT_SHA256):
        GraspStudentPolicy.load(changed_copy)
    monkeypatch.setenv(P.SHA256_ENV, f" {digest.upper()} ")
    with caplog.at_level(logging.WARNING, logger=P.__name__):
        pol = GraspStudentPolicy.load(changed_copy)
    assert pol.sha256 == digest and pol.path == changed_copy
    assert any(digest in r.getMessage() and C.CHECKPOINT_SHA256 in r.getMessage() for r in caplog.records)
    with pytest.raises(CheckpointError, match=digest):
        GraspStudentPolicy.load(ckpt_path)


def test_an_explicit_sha256_is_used_as_given_and_a_malformed_variable_is_refused(
        ckpt_path: Path, changed_copy: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    digest = sha256_of(changed_copy)
    monkeypatch.setenv(P.SHA256_ENV, "f" * 64)
    assert GraspStudentPolicy.load(changed_copy, sha256=digest).sha256 == digest
    with pytest.raises(CheckpointError, match="f" * 64):
        GraspStudentPolicy.load(ckpt_path)
    assert GraspStudentPolicy.load(ckpt_path, sha256=C.CHECKPOINT_SHA256).sha256 == C.CHECKPOINT_SHA256
    with caplog.at_level(logging.WARNING, logger=P.__name__):
        pol = GraspStudentPolicy.load(ckpt_path, sha256=None)
    assert pol.sha256 is None and any("NOT verified" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv(P.SHA256_ENV, "05e5aaf6")
    with pytest.raises(CheckpointError, match=P.SHA256_ENV):
        GraspStudentPolicy.load(ckpt_path)
    monkeypatch.setenv(P.SHA256_ENV, "")
    assert GraspStudentPolicy.load(ckpt_path).sha256 == C.CHECKPOINT_SHA256


def test_the_policy_is_timed_on_zeros_input(policy: GraspStudentPolicy) -> None:
    r = policy.time(200)
    print(f"student on the CPU ({torch.get_num_threads()} threads): median {r['median_ms']:.1f} ms, "
          f"p95 {r['p95_ms']:.1f} ms, first {r['first_ms']:.1f} ms")
    assert set(r) == {"median_ms", "p95_ms", "first_ms"} and np.isfinite(list(r.values())).all()
    # The budget is 30 / 45 ms; the factor 2 absorbs other test runs on a shared machine.
    assert r["median_ms"] < 2 * P.TIME_MEDIAN_MAX_MS
    assert r["p95_ms"] < 2 * P.TIME_P95_MAX_MS


# ---------------------------------------------------------------- fetch

class _Response(io.BytesIO):
    def __init__(self, data: bytes, url: str, on_read: object = None) -> None:
        super().__init__(data)
        self.url = url
        self.on_read = on_read

    def geturl(self) -> str:
        return self.url

    def read(self, n: int | None = -1) -> bytes:
        if self.on_read is not None:
            self.on_read(self.tell())
        return super().read(n)


def test_fetch_refuses_a_plain_http_url_without_asking_the_server(tmp_path: Path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    (served / "student.pt").write_bytes(b"weights")
    requests: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kw: object) -> None:
            super().__init__(*args, directory=str(served), **kw)

        def do_GET(self) -> None:
            requests.append(self.path)
            super().do_GET()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dest = tmp_path / "out" / "student.pt"
        url = f"http://127.0.0.1:{server.server_address[1]}/student.pt"
        with pytest.raises(CheckpointError, match="https"):
            fetch_checkpoint(dest, url=url, sha256=hashlib.sha256(b"weights").hexdigest())
    finally:
        server.shutdown()
        server.server_close()
    assert requests == []
    assert not dest.exists() and not dest.with_name("student.pt.part").exists()


def test_fetch_streams_to_a_part_file_verifies_the_hash_and_renames_into_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.random.default_rng(0).bytes(2_500_000)
    dest = tmp_path / "models" / "student.pt"
    part = dest.with_name("student.pt.part")
    seen: list[tuple[str, float, bool]] = []

    def opener(url: str, timeout: float) -> _Response:
        def on_read(pos: int) -> None:
            seen.append((url, timeout, part.exists()))
        return _Response(payload, url, on_read)

    monkeypatch.setattr(P.urllib.request, "urlopen", opener)
    url = "https://example.invalid/student.pt"
    assert fetch_checkpoint(dest, url=url, sha256=hashlib.sha256(payload).hexdigest(), timeout_s=7.0) == dest
    assert dest.read_bytes() == payload and not part.exists()
    assert seen and all(u == url and t == 7.0 for u, t, _ in seen)
    assert seen[-1][2], "the download was not streamed into the .part file"


def test_fetch_removes_the_part_file_and_keeps_the_old_file_on_a_hash_mismatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.random.default_rng(1).bytes(1_500_000)
    dest = tmp_path / "student.pt"
    dest.write_bytes(b"old")
    part = dest.with_name("student.pt.part")
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda url, timeout: _Response(payload, url))
    with pytest.raises(CheckpointError, match="expected 0{64}"):
        fetch_checkpoint(dest, url="https://example.invalid/student.pt", sha256="0" * 64)
    assert dest.read_bytes() == b"old" and not part.exists()
    monkeypatch.setenv(P.SHA256_ENV, hashlib.sha256(payload).hexdigest())   # fetch never reads it
    with pytest.raises(CheckpointError, match=f"expected {C.CHECKPOINT_SHA256}"):
        fetch_checkpoint(dest, url="https://example.invalid/student.pt")
    assert dest.read_bytes() == b"old" and not part.exists()


def test_fetch_cleans_up_after_a_broken_stream_or_a_redirect_to_plain_http(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.random.default_rng(2).bytes(3_000_000)
    dest = tmp_path / "student.pt"
    part = dest.with_name("student.pt.part")

    def broken(pos: int) -> None:
        if pos > 0:
            raise ConnectionResetError("peer went away")

    url, digest = "https://example.invalid/student.pt", hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda url, timeout: _Response(payload, url, broken))
    with pytest.raises(CheckpointError, match="peer went away"):
        fetch_checkpoint(dest, url=url, sha256=digest)
    assert not dest.exists() and not part.exists()
    monkeypatch.setattr(P.urllib.request, "urlopen",
                        lambda url, timeout: _Response(payload, "http://example.invalid/student.pt"))
    with pytest.raises(CheckpointError, match="not https"):
        fetch_checkpoint(dest, url=url, sha256=digest)
    assert not dest.exists() and not part.exists()


# ---------------------------------------------------------------- lerobot-twin grasp-model

def _camera_spec(**change: float) -> bytes:
    T = np.array(C.T_GRIPPER_CAMERA, float)
    T[0, 3] += change.get("dx", 0.0)
    fov = [58.62, C.FOVY_DEG + change.get("dfov", 0.0)]
    return json.dumps({"frame": "gripper_link", "T_gripper_camera_opencv": T.tolist(),
                       "lens": {"recommended": {"fov_deg": fov}}}).encode()


def _namespace(action: str, checkpoint: Path | None = None, n: int = 200) -> argparse.Namespace:
    return argparse.Namespace(action=action, checkpoint=None if checkpoint is None else str(checkpoint), n=n)


def test_grasp_model_fetch_downloads_verifies_and_compares_the_camera_spec(
        ckpt_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture) -> None:
    weights = ckpt_path.read_bytes()
    spec = {"body": _camera_spec()}
    asked: list[str] = []

    def opener(url: str, timeout: float) -> _Response:
        asked.append(url)
        return _Response(weights if url == C.CHECKPOINT_URL else spec["body"], url)

    monkeypatch.setattr(P.urllib.request, "urlopen", opener)
    dest = tmp_path / "grasp" / "student.pt"
    assert P.main_model(_namespace("fetch", dest)) == 0
    assert sha256_of(dest) == C.CHECKPOINT_SHA256
    assert json.loads(dest.with_name(P.CAMERA_SPEC_NAME).read_text())["frame"] == "gripper_link"
    assert asked == [C.CHECKPOINT_URL, C.CAMERA_SPEC_URL]

    asked.clear()
    spec["body"] = _camera_spec(dx=1e-6)
    assert P.main_model(_namespace("fetch", dest)) == 1
    assert asked == [C.CAMERA_SPEC_URL]
    assert "ROZNI SIE" in capsys.readouterr().err
    spec["body"] = _camera_spec(dfov=0.01)
    assert P.main_model(_namespace("fetch", dest)) == 1
    assert "fov_deg" in capsys.readouterr().err


def test_grasp_model_fetch_never_overwrites_a_file_with_another_hash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None) -> None:
    dest = tmp_path / "student.pt"
    dest.write_bytes(b"someone else's weights")
    assert P.main_model(_namespace("fetch", dest)) == 1
    assert dest.read_bytes() == b"someone else's weights"


def test_grasp_model_check_prints_the_hash_the_schema_and_the_statistics(
        ckpt_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        no_network: None) -> None:
    monkeypatch.delenv(P.SHA256_ENV, raising=False)
    assert P.main_model(_namespace("check", ckpt_path)) == 0
    out = capsys.readouterr().out
    assert C.CHECKPOINT_SHA256 in out and "'resnet18'" in out and "130" in out
    assert out.count("srednia") == C.OBS_DIM


def test_grasp_model_check_and_time_without_a_checkpoint_name_the_fetch_command(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        no_network: None) -> None:
    monkeypatch.setattr(P, "data_path", lambda rel: tmp_path / rel)
    monkeypatch.delenv(P.CHECKPOINT_ENV, raising=False)
    for action in ("check", "time"):
        assert P.main_model(_namespace(action)) == 1
        assert "lerobot-twin grasp-model fetch" in capsys.readouterr().err
    assert P.main_model(argparse.Namespace(what="check", checkpoint=None, n=200)) == 1
    assert "lerobot-twin grasp-model fetch" in capsys.readouterr().err
    assert P.main_model(_namespace("train")) == 2


def test_grasp_model_check_with_a_missing_checkpoint_names_the_file_not_the_fetch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        no_network: None) -> None:
    env = tmp_path / "student_v4.pt"
    env.write_bytes(b"e")
    monkeypatch.setenv(P.CHECKPOINT_ENV, str(env))
    missing = tmp_path / "student_v5.pt"
    assert P.main_model(_namespace("check", missing)) == 1
    err = capsys.readouterr().err
    assert f"Nie ma pliku --checkpoint {missing}" in err and "fetch" not in err


def test_grasp_model_time_prints_the_numbers_and_the_verdict(
        ckpt_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        no_network: None) -> None:
    monkeypatch.delenv(P.SHA256_ENV, raising=False)
    rc = P.main_model(_namespace("time", ckpt_path, n=20))
    out = capsys.readouterr().out
    assert "mediana" in out and "p95" in out and "Werdykt" in out
    assert rc == (0 if "Werdykt: OK" in out else 1)
