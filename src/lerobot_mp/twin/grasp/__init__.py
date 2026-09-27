"""Grasp student in the twin: a wrist-camera policy trained in simulation, driven on the A1X.

Modules, from the bottom (each imports only modules named before it):

* `contract` - the training contract: constants, units and the three small helpers,
* `rig`      - per-rig measured values (`workspace/grasp.json`) and the refusal reasons,
* `command`  - command bookkeeping, a verbatim port of the training environment's step,
* `keys`     - the one stdin reader: once motion has started a bare Enter is always STOP,
* `net`, `policy`                  - the network and its checkpoint,
* `camera`, `hover`, `gripper`, `observer` - what the policy sees,
* `motion`   - taking, moving and holding the arm, without a policy,
* `driver`   - one episode tick by tick,
* `evaluate`, `run`, `bench`       - sim evaluation, the four run modes and the bench tools.

Nothing is imported here, so every module imports on its own.
"""
