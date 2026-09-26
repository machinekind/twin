"""PolicyRunner na (udawanym) prawdziwym ramieniu: straznicy, ktorych trening nie ma.

Wszystko w czasie symulowanym: petla blizniaka bez watku (`connect(threaded=False)`
+ `Twin.step`), polityka bez watku (`start(threaded=False)` + `step_once`) - wynik
nie zalezy od obciazenia procesora.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from lerobot_mp.twin.rl import task as tk  # noqa: E402
from lerobot_mp.twin.rl.policy import Policy, PolicyMeta  # noqa: E402
from lerobot_mp.twin.rl.runner import PolicyRunner  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

from test_twin_runtime import FakeArm, connect_fake  # noqa: E402


def constant_policy(task: tk.TaskConfig, action: dict[int, float] | None = None) -> Policy:
    """Polityka, ktora zawsze daje te sama akcje (zera + `action`)."""
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=task.act_dim))
    with torch.no_grad():
        for p in pol.actor.parameters():
            p.zero_()
        for k, v in (action or {}).items():
            pol.actor[-1].bias[k] = v
    return pol


@pytest.fixture
def twin():
    tw = Twin(Workspace())
    try:
        yield tw
    finally:
        tw.close()


def drive(twin, runner, seconds: float, t0: float = 0.0) -> float:
    """Petla blizniaka 50 Hz i polityka 20 Hz na wspolnym zegarze symulowanym (krok 10 ms)."""
    n = int(round(seconds / 0.01))
    for k in range(1, n + 1):
        if k % 2 == 0:
            twin.step(0.02)
        if k % 5 == 0 and runner.status.running:
            runner.step_once(t0 + k * 0.01)
    return t0 + n * 0.01

# ------------------------------------------------------------ rozjazd = kolizja
def test_physical_policy_refuses_command_only_gripper_state(twin, monkeypatch):
    arm = FakeArm()
    arm.gripper_measured = lambda: False
    connect_fake(twin, monkeypatch, arm)
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach")))
    with pytest.raises(RuntimeError, match="calibrated gripper feedback"):
        runner.start(threaded=False)
    assert twin.owner is None and not runner.status.running


def test_tracking_stop_holds_the_measured_pose_instead_of_pressing_on(twin, monkeypatch):
    """Staw zablokowany (kolizja): polityka staje, a ramie trzyma ZMIERZONA poze.

    Przed poprawka `_halt` robil tylko `set_engaged(False)`, a nadzor trzymal ostatni
    rozkaz - ~25 st. za przeszkoda, czyli serwo dociskalo do niej z pelnym momentem.
    """
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    arm.blocked["shoulder_pan"] = 10.0
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach"), {0: 1.0}), episode_limit=False)
    runner.start(threaded=False)
    drive(twin, runner, 3.0)
    assert not runner.status.running
    assert "nie nadaza" in runner.status.stopped_because
    drive(twin, runner, 0.3)
    assert arm.sent[-1]["shoulder_pan"] == pytest.approx(10.0, abs=0.5)
    assert twin.status.command["shoulder_pan"] == pytest.approx(10.0, abs=0.5)
    assert twin.owner is None


# ------------------------------------------------------------ wlasnosc ramienia
def test_start_does_not_switch_off_the_clutch_of_the_next_owner(twin, monkeypatch):
    """STOP miedzy `claim` a sprzeglem runnera, a zaraz potem panel bierze ramie i wlacza sprzeglo:
    runner wylaczal sprzeglo panelu (pole wyboru w panelu dalej "wlaczone")."""
    connect_fake(twin, monkeypatch, FakeArm())
    real_claim = twin.claim

    def claim_then_race(owner, preempt=None):
        real_claim(owner, preempt)
        twin.estop("STOP z panelu")
        twin.clear_estop()
        real_claim("panel")
        twin.set_engaged(True)
    monkeypatch.setattr(twin, "claim", claim_then_race)
    runner = PolicyRunner(twin, constant_policy(tk.make_task("reach")))
    with pytest.raises(RuntimeError, match="odebrane"):
        runner.start(threaded=False)
    assert twin.owner == "panel"
    twin.step(0.02)
    assert twin.status.engaged


# ------------------------------------------------------------ polityka z innego ramienia
def test_runner_refuses_a_policy_trained_for_another_robot(twin, monkeypatch):
    """Polityka z innego ramienia (inne stawy, inne jednostki chwytaka) nie rusza ramienia."""
    arm = FakeArm()
    connect_fake(twin, monkeypatch, arm)
    claims = []
    monkeypatch.setattr(twin, "claim", lambda *a, **kw: claims.append(a))
    task = tk.make_task("reach")
    pol = constant_policy(task)
    pol.meta.robot = "a1x"
    with pytest.raises(RuntimeError, match="uczona na a1x"):
        PolicyRunner(twin, pol)
    pol = constant_policy(task)
    pol.meta.act_dim = 7
    with pytest.raises(RuntimeError, match="7 wyjsc"):
        PolicyRunner(twin, pol)
    assert claims == [] and twin.owner is None
    PolicyRunner(twin, constant_policy(task))                 # SO-101 na SO-101 - bez zmian
