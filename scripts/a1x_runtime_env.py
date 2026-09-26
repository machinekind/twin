"""Simulation-only adapter for collecting demonstrations through the real controller.

Implements the small interface used by collect_a1x_lift_demos. It deliberately
uses PolicyRunner, Twin and SafetySupervisor instead of setting physics controls.
The returned reward is unused: this adapter is for demonstrations, not PPO.
"""

from dataclasses import asdict, replace
from types import SimpleNamespace

import mujoco
import numpy as np

from evaluate_a1x_runtime import PhysicsMonitor
from lerobot_mp.twin.rl.env import TwinEnv, scene_config
from lerobot_mp.twin.rl.policy import PolicyMeta
from lerobot_mp.twin.rl.randomize import Randomization
from lerobot_mp.twin.rl.runner import PolicyRunner
from lerobot_mp.twin.rl.task import Limits
from lerobot_mp.twin.runtime import Twin
from lerobot_mp.twin.workspace import Workspace


class RuntimeLiftEnv:
    def __init__(self, task):
        if task.name != "lift" or task.n_joints != 7 or task.control_hz != 20 or task.timestep != .005:
            raise ValueError("This adapter supports only A1X lift at 20 Hz / 5 ms physics")
        self.task = task
        workspace = Workspace(robot="a1x")
        self.rand = replace(Randomization.none(), fold_yaw=True)
        self.resetter = TwinEnv(task, workspace=workspace, randomization=self.rand)
        self.twin = Twin(workspace)
        config = scene_config(task, workspace)
        self.twin.configure(objects=config.objects, grasp_sensors=config.grasp_sensors)
        self.scene, self.kin = self.twin.scene, self.twin.scene.kin
        self.limits = Limits.of(self.kin)
        self.transform = np.linalg.inv(self.scene.T_base2world)
        self.action = np.zeros(7, np.float32)
        self.policy = SimpleNamespace(task=task, meta=PolicyMeta(task=asdict(task), robot="a1x", hidden=[],
            obs_dim=task.obs_dim, act_dim=7, randomization=asdict(self.rand)), act=self._act)
        self.runner = self.monitor = None
        self.t = 0

    def _act(self, obs):
        # Assert that recorded observations are the inputs actually seen by the
        # controller policy, not a separately reconstructed training observation.
        np.testing.assert_array_equal(obs, self.observation)
        return self.action.copy()

    @property
    def q_cmd(self):
        return self.runner.q_cmd

    def reset(self, *, seed):
        self._stop()
        self.resetter.reset(seed=seed)
        m, d = self.scene.model, self.scene.data
        source = self.resetter.scene.data
        m.opt.timestep = self.task.timestep
        mujoco.mj_resetData(m, d)
        d.qpos[:] = source.qpos
        d.qvel[:] = source.qvel
        d.ctrl[:] = source.ctrl
        mujoco.mj_forward(m, d)
        self.twin.connect("sim", threaded=False)
        if not self.twin.status.simulated:
            raise RuntimeError("Demonstration collection must never connect to physical hardware")
        self.monitor = PhysicsMonitor(self.scene, self.task)
        self.scene.step = self.monitor.step
        self.runner = PolicyRunner(self.twin, self.policy, cube_provider=self.monitor.cube)
        self.runner.start(threaded=False)
        self.observation = self.runner.observation(dict(self.twin.status.measured))
        self.t = 0
        return self.observation.copy(), {}

    def state(self):
        d, R, translation = self.scene.data, self.transform[:3, :3], self.transform[:3, 3]
        cube, rotation, _ = self.monitor.cube()
        return dict(q=d.qpos[self.kin.qadr].copy(), cube_pos=cube, cube_rot=rotation,
                    tcp=R @ d.site_xpos[self.kin.site_id] + translation,
                    jaws=np.asarray([d.sensordata[a] for a in self.monitor.jaw_adr]))

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Demonstration action must have seven finite entries")
        self.action = np.clip(action, -1, 1)
        keep_running = self.runner.step_once(self.t / self.task.control_hz)
        if keep_running:
            for tick in range(self.t * 5 + 1, (self.t + 1) * 5 + 1):
                if tick % 2 == 0:
                    self.twin.step(.02)
                    if not self.twin.connected or self.monitor.contacts or self.monitor.numerical:
                        break
        self.t += 1
        reason = self.runner.status.stopped_because
        controller_finished = reason == "zadanie wykonane"
        normal_end = controller_finished or reason == "koniec epizodu"
        guard_fault = not self.twin.connected or (not keep_running and not normal_end)
        numerical = self.monitor.numerical or any(w.number for w in self.scene.data.warning)
        unsafe = bool(self.monitor.contacts)
        terminated = bool(guard_fault or numerical or unsafe)
        required_hold_s = max(2.0, self.task.end_on_success / self.task.control_hz)
        finished = bool(controller_finished and not terminated
                        and self.monitor.max_hold_s >= required_hold_s - 1e-8)
        self.observation = self.runner.observation(dict(self.twin.status.measured))
        state = self.state()
        info = dict(finished=finished, unsafe=unsafe, broken=bool(numerical),
                    height=float(state["cube_pos"][2] - self.task.cube_half),
                    grasped=bool(state["jaws"].min() > .5), runtime_stop=reason or self.twin.status.error,
                    guard_rejection=bool(guard_fault), max_continuous_hold_s=self.monitor.max_hold_s,
                    max_coupling_mm=self.monitor.max_coupling_mm)
        return self.observation.copy(), 0.0, terminated, normal_end or self.t >= self.task.episode_steps, info

    def _stop(self):
        if self.runner is not None:
            self.runner.stop()
        if self.monitor is not None:
            self.scene.step = self.monitor.original_step
        self.twin.disconnect()

    def close(self):
        self._stop()
        self.twin.close()
        self.resetter.close()
