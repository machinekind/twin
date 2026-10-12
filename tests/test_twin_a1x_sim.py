"""Galaxea A1X w blizniaku w symulacji: dom etapami i zatrzymanie na kolizji."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin.robots import A1X  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402


def test_a1x_home_goes_in_stages_elbow_first_and_stops_on_a_collision(monkeypatch):
    """Dom A1X etapami: najpierw lokiec, potem nadgarstek, potem bark; etap z kolizja zatrzymuje dom.

    Prosta rampa wszystkich stawow z pozy zlozonej obracala nadgarstek w kolumne barku
    (zmierzone na prawdziwym ramieniu, odtworzone w blizniaku: path_clear z J4 przed J3
    daje "arm_link1 z gripper_link w 7% drogi").

    Blizniak na zegarze wirtualnym, krokami testu (`twinbench.Bench`): watek domu (`dom`) i `move`
    jada w tym samym czasie symulowanym.
    """
    from twinbench import Bench

    from lerobot_mp.twin import collision

    bench = Bench.sim("a1x")
    tw = bench.twin
    try:
        rest = {"arm_joint1": 4.0, "arm_joint2": 3.5, "arm_joint3": -3.5, "arm_joint4": 0.5,
                "arm_joint5": -2.0, "arm_joint6": -8.0, "gripper": 60.0}
        bench.call(lambda: tw.move(rest, 2.0, owner="test"))
        tw.release("test")
        moves = []
        real_move = tw.move

        def spy(joints, duration, *a, **kw):
            moves.append({k: round(v, 1) for k, v in joints.items()})
            return real_move(joints, duration, *a, **kw)
        monkeypatch.setattr(tw, "move", spy)
        tw.home()
        assert bench.pump(until=lambda: tw.owner != "dom" and len(moves) >= 3, max_sim_s=40.0), moves
        bench.pump(sim_s=0.5)
        assert tw.home_note == "", tw.home_note
        assert len(moves) >= 3
        first = moves[0]
        changed = [k for k in first if abs(first[k] - rest[k]) > 0.5]
        assert changed == ["arm_joint3"], changed                       # lokiec sam, reszta stoi
        assert abs(first["arm_joint3"] - A1X.home["arm_joint3"]) < 0.6
        home = A1X.home
        for k in ("arm_joint2", "arm_joint3", "arm_joint4"):
            assert abs(tw.status.measured[k] - home[k]) < 2.0, (k, tw.status.measured[k])

        # Etap z kolizja: ramie nie rusza, powod w home_note.
        bench.call(lambda: tw.move(rest, 2.0, owner="test"))
        tw.release("test")
        moves.clear()
        monkeypatch.setattr(collision.CollisionChecker, "path_clear", lambda self, a, b, step_rad=0.04: "kolizja: test")
        tw.home()
        assert bench.pump(until=lambda: tw.home_note, max_sim_s=5.0)
        assert "kolizja: test" in tw.home_note
        assert moves == []
        assert abs(tw.status.measured["arm_joint3"] - rest["arm_joint3"]) < 2.0
    finally:
        tw.disconnect()
        bench.clock.close()
    bench.assert_on_time()


def test_a1x_sim_starts_closed_and_set_home_here_closes_the_scene_gripper(tmp_path):
    """Sim startuje z zamknietym chwytakiem (`A1X.home`), a "Ustaw dom = tu" zamyka chwytak sceny
    od razu i nadzor rusza od 0 - jak na A1X bez pomiaru (dom = szczeki zamkniete)."""
    from lerobot_mp.twin import runtime as rt

    tw = rt.Twin(Workspace(robot="a1x", path=tmp_path / "twin.json"))
    try:
        tw.connect("sim", threaded=False)
        assert A1X.home["gripper"] == 0.0
        assert tw.joints()["gripper"] == pytest.approx(0.0, abs=0.5)
        tw.claim("test")
        tw.set_engaged(True, owner="test")
        tw.set_target({"gripper": 60.0}, owner="test")
        for _ in range(50):
            tw.step(0.02)
        assert tw.joints()["gripper"] == pytest.approx(60.0, abs=2.0)
        arm_before = {k: v for k, v in tw.joints().items() if k != "gripper"}
        home = tw.set_home_here()
        assert home["gripper"] == 0.0 and tw.workspace.home["gripper"] == 0.0
        assert tw.joints()["gripper"] == pytest.approx(0.0, abs=0.5)          # scena zamknieta od razu
        assert tw._supervisor.command["gripper"] == 0.0
        for _ in range(25):
            tw.step(0.02)
        assert tw.joints()["gripper"] == pytest.approx(0.0, abs=0.5)          # serwo trzyma 0, cel nie wraca
        assert tw.status.command["gripper"] == pytest.approx(0.0, abs=0.5)
        for k, v in arm_before.items():
            assert tw.joints()[k] == pytest.approx(v, abs=0.5), k
        # Nastepne "otworz" to rampa od 0.
        tw.set_target({"gripper": 100.0}, owner="test")
        tw.step(0.02)
        assert 0.0 < tw.status.command["gripper"] < 10.0
    finally:
        tw.disconnect()
