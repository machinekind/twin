# Spectacles driver

The Spectacles driver lets a person wearing Snap Spectacles point at a spot and
have the Galaxea A1X arm move there. This page covers the twin's side: the
`lerobot-twin specs` command, the Spectacles folder in the panel, and the code
in `src/lerobot_mp/twin/specsreach.py`. For the full two-machine setup with
the glasses, follow [RIG.md](RIG.md).

Nothing here has run on the glasses or on the real arm yet. It has been tested
in simulation, with a fake relay and a fake CAN bus.

## 1. What it is

- The glasses send a point in the arm's base frame. The point is either a hand
  pinch inside the reach box or the centre of an object the glasses detected.
- The twin checks the point and, if every check passes, moves the arm there.
- The arm-level work lives in the `galaxeo.arm` package (repository
  galaxeo-manipulators): the A1X model, IK (inverse kinematics: joint angles
  for a tool position), collision checks, the reach box, and the guarded move
  (a ramp with an impact guard and a backoff).
- The twin adds the rest: arm ownership, STOP, the three modes, consent for
  the real arm, the relay protocol and the panel folder.
- Only the A1X is supported. The command and the panel refuse a workspace
  with another arm (SO-101).

```
glasses (Lens) --/specs--> relay (server.js) --/twin--> twin driver --> Twin --> sim or A1X on CAN
```

The relay and the Lens live in the specs-lens repository
([README](https://github.com/machinekind/specs-lens/blob/reach-project/README.md),
sections "Robot frame" and "Reach box").

**Base frame.** Origin at the centre of the arm base on the joint 1 axis,
table top at z = 0, x forward, y to the arm's left, z up, metres.

**Reach box.** `galaxeo.arm` computes it from the A1X model. For the A1X
gripper it is x 0.20 to 0.50 m, y −0.30 to 0.30 m, z 0.05 to 0.30 m.

**Obstacles.** Before each target the twin gives galaxeo everything the arm
must not hit: every object in the twin scene that has collision, a 12 cm cube
at each placed camera, and the fixed `obstacles` listed in
`workspace/twin.json`. For an object target, the object within 6 cm of the
target is left out, because the arm is going to it.

## 2. Run it

Only one process may own the arm: either the CLI or the panel, never both.
On macOS only one process can open the XCAN adapter. On Linux two processes
would both send commands and the arm would follow neither.

### The command

```bash
lerobot-twin specs [--relay URL] [--mode dry|sim|real] [--i-know] [--start-center]
                   [--backend NAME] [--port PORT] [--trusted-only] [--max-jump-deg DEG]
                   [--grasp] [--workspace FILE] [--seconds S] [--config FILE]
```

| flag | what it does |
|---|---|
| `--relay URL` | The relay address, path `/twin`. Default `ws://127.0.0.1:8080/twin`. When the relay runs on the glasses Mac and the arm is on another machine, give that Mac's address: `--relay ws://192.168.1.105:8080/twin`. |
| `--mode dry\|sim\|real` | Default `dry`. See [Modes and consent](#4-modes-and-consent). |
| `--i-know` | Required with `--mode real`. Without it the command prints a message and exits with code 2. |
| `--start-center` | Before taking targets, move the arm above the centre of the reach box. See below. |
| `--backend NAME` | Which arm backend: `sim`, `a1x`, ... Default: `sim` in `dry` and `sim`, so nothing touches hardware by accident; in `real`, the backend saved in the workspace. |
| `--port PORT` | The arm port: `xcan` on macOS, `can0` on Linux, `PCAN_USBBUS1` on Windows. Default: the port saved in the workspace. |
| `--trusted-only` | Accept targets only from a trusted glasses registration. Without it, any registration is enough. |
| `--max-jump-deg DEG` | Refuse a target (`jump`) when any joint would turn more than this on one leg. Default 150. Long legs are still driven in pieces of at most 40 degrees. |
| `--grasp` | Object targets: go down, close the gripper and come back up. Without it the arm only stops above the object. |
| `--workspace FILE` | The workspace file. Default `workspace/twin.json`. |
| `--seconds S` | Stop after S seconds (used by tests). |
| `--config FILE` | YAML config such as `configs/local.yaml` (A1X joint offsets and gripper settings). |

**Why `--start-center`.** The A1X home pose is far from the reach box. From
home, every point in the box needs 45 to 90 degrees on some joint. That is
below the default `--max-jump-deg`, so without `--start-center` the first
target is a long drive along a straight line in joint space. With
`--start-center` the arm first moves above the box centre in stages (joint
groups one after another, as the panel's Home does), at 10 degrees per second,
under the impact guard with tighter limits. Every later target then starts
close to its goal.

- In `real` mode it asks for Enter first. Ctrl-C at that prompt ends the
  command and the arm holds its pose.
- On a real arm it is only allowed with `--mode real`. With `--mode dry` on
  the real arm the command refuses.
- If the move fails, the command prints the reason and, on a real arm, exits
  with the arm holding its pose.
- On a simulated arm without `--start-center`, the command moves the arm
  above the box centre on its own, in 2 s, without asking.

What the command prints: the arm line
(`Ramie: a1x (prawdziwe), tryb real, przekaznik ws://...`), then every
`targetState` as one JSON line. Refusals, impacts, a missing relay and (in
`dry`) joint efforts go to the log.

### The panel

```bash
SPECS_RELAY=ws://192.168.1.105:8080/twin lerobot-twin ui --port 8765
```

`SPECS_RELAY` pre-fills the relay field. `--port 8765` keeps the panel off
8080, which is the relay's port. Open `http://localhost:8765`.

1. Tab **Ramie**: connect the arm (A1X on its port, or `sim`).
2. Tab **Polityki**, folder **Spectacles**:
   - **Przekaznik**: the relay address, path `/twin`.
   - **Tryb okularow**: `dry`, `sim` or `real`.
   - **Okulary moga ruszac prawdziwym ramieniem**: the consent box for `real`.
   - **Polacz z okularami**: start the driver and connect to the relay.
   - **Rozlacz okulary**: stop the driver. A move in progress holds the
     measured pose. It also clears the consent box.
   - The status line: `Okulary: polaczone, tryb dry, sesja: <id>`, then the
     last target, for example `ostatni cel [35, 0, 17] cm: reached`.

The panel differs from the command in three ways. It has no
`--start-center`: move the arm near the box first with the panel. It does not
grasp. It does not use `--max-jump-deg` (it uses the default, 150). Like the
command without `--trusted-only`, it accepts any registration the glasses use.

While the panel is armed (**UZBROJ**), the panel owns the arm and every
target gets `busy`. Disarm the panel before you pinch.

### STOP

- The red **STOP** button at the top of the panel.
- **Enter** or **Ctrl-C** in the terminal running `lerobot-twin specs`.

STOP makes the arm hold its measured pose. It does not cut the motors: the A1X
has no brakes. **The glasses cannot stop the arm.** Keep a hand on the keyboard
or the panel.

## 3. What happens to a target

The network thread only receives targets and puts them in a queue. One worker
thread does everything else. A malformed target is logged and dropped with no
reply. If several targets are waiting, the newest one runs and the older ones
get `rejected` with `busy`.

### The checks, in order

The first check that fails ends the target with `rejected` and this reason:

| reason | when |
|---|---|
| `not connected` | the twin has no arm connected |
| `stop` | STOP is active |
| `busy` | another owner has the arm (the armed panel, a policy, a calibration wave) |
| `not armed` | `real` without consent, or `sim` on a hardware backend |
| `untrusted` | no registration yet, or (with `--trusted-only`) the target or the registration is not trusted |
| `session` | the target's session is not the session of the registration the twin holds |
| `outside volume` | the point is outside the reach box |
| `no depth` | an object target whose depth was not measured (the glasses put it at a guessed 150 cm) |
| `stale` | an object target from a detection older than 0.5 s (`detail` gives the age) |
| `ik` | IK found no joint angles within 3 mm, or, for an object, the gripper would point more than about 15 degrees away from straight down |
| `jump` | a joint would turn more than `--max-jump-deg` on one leg |
| `collision` | the straight joint-space path from the current command to the goal hits the arm itself, the table, or an obstacle |

`ik`, `jump` and `collision` carry a `detail` (IK error in mm, the jump in
degrees, the colliding pair).

These words are the same as the refusals of galaxeo's `Arm.plan`, but the
twin does not call `Arm.plan`. It runs the same galaxeo parts itself (reach
box, IK, collision checker) because it plans several legs, checks the
straight-down approach, and uses its own jump limit.

**Legs.** A hand target is one leg: the tool goes to the point, in any
orientation. An object target goes to a point 8 cm above the object (at most
3 cm below the top of the box), with the gripper pointing down. With `--grasp`
two more legs follow: down to 1 cm above the centre at half speed, then close
the gripper and go back up along the same path.

### After the checks

1. The driver sends `accepted` and shows a green `target` sphere in the
   glasses. In `dry` it then sends `hold` and stops there.
2. **Hold-off, `real` mode, hand targets only.** The driver waits 1 s. A new
   target, STOP or a mode change in that second cancels the target.
3. It takes the arm as owner `specs`. If it cannot, the target gets
   `rejected` with `busy` or `stop`.
4. **Re-check.** If the arm's command changed by 0.5 degrees or more since the
   checks (someone moved a panel slider), the target ends with `hold`, reason
   `moved`. It is not planned again: pinch again.
5. It sends `moving` and drives the legs. It sends `moving` again at the start
   of each later leg.

### The move

Each leg is `galaxeo.arm.Motion.move`, sent through the twin
(`twin/armlink.py`): every command is `Twin.set_target` as owner `specs`. The
twin's `a1x` backend stays the only writer on the CAN bus, and STOP, Home and
the panel work as for any other owner.

- The move starts from the **measured** pose.
- Speed: 20 degrees per second on the fastest joint (10 when going down to an
  object). The command follows a smoothstep ramp.
- galaxeo refuses any single move over 45 degrees, so the driver cuts each leg
  into equal pieces of at most 40 degrees along the same checked path.
- While the arm moves, the twin's speed limit drops to 35 degrees per second.
  It drops to 15 during a backoff, and goes back to the panel's value
  afterwards.
- Arrival: every joint within 2 degrees of the goal and slower than 10 degrees
  per second. A joint that does not get there in 3 s ends the target with
  `hold`, reason `not arrived`.
- Feedback older than 150 ms, or frozen, ends the target with `hold`, reason
  `stale feedback` or `frozen telemetry`.

### The impact guard

It watches every tick of the move. Two rules:

| rule | trips when |
|---|---|
| lag | the command keeps moving but the joint does not: command and joint more than 6 degrees apart for 0.15 s while the joint moves slower than 15 degrees per second. 3 degrees when going down to an object. 2.5 degrees for 0.3 s during arrival. `--start-center` uses 4 degrees and 0.2 s at arrival. |
| effort | a joint's effort is above the threshold for 3 samples in a row. A1X: 20 on moves, 12 when going down, in the units of the `0x052` feedback frame. Simulation: 0.8 (0.5 going down) times the joint's force range. |

A joint that is already above half its effort threshold at the start of a move
gets only the lag rule.

**The A1X effort thresholds are guesses.** Nobody has measured them on this
arm. Until they are measured, the lag rule is the main protection. See
[The effort log](#5-the-effort-log-in-dry-mode).

The guard sits under the twin's own supervisor and the backend's own fault
("deaf or blocked": 3 degrees for 1 s, then STOP). The guard is tuned to trip
first.

### The backoff

When the guard trips:

1. The arm holds its measured pose. It must not keep pressing.
2. It backs off at 8 degrees per second to the newest good pose from at least
   0.4 s before the trip that is at least 3 degrees away.
3. The target ends with `backoff`, reason `impact lag arm_joint2` or
   `impact effort arm_joint2` (the rule and the joint).
4. Targets that were waiting in the queue get `rejected` with `busy`, so the
   wearer points again.

The backoff has its own guard, armed after 0.3 s. A second trip during the
backoff only holds: the target ends with `hold`, the same `impact ...`
reason, and `detail` `second trip during backoff: ...`. A backoff is not a
STOP: the next pinch works.

### Hold reasons

`hold` means the arm stopped where it is, holding its measured pose.

| reason | cause |
|---|---|
| (none) | `dry` mode: checks passed, nothing moved |
| `retargeted` | a new target arrived; the new one runs from where the arm is |
| `moved` | the command changed during the hold-off |
| `mode` | the mode or the consent changed in the panel during the move |
| `relay` | the relay connection dropped during a move in `real` mode |
| `not connected` | the arm disconnected during the move |
| `stop` | STOP, or the driver was stopped |
| `preempted` | something else took the arm: Home, the panel, another owner |
| `not arrived`, `stale feedback`, `frozen telemetry`, `no feedback` | from galaxeo's move |
| `impact ...` | a second trip during a backoff |
| `error` | an exception in the driver (`detail` has it) |

After `reached` the driver releases the arm, which keeps holding the target.

### States sent back

| state | meaning |
|---|---|
| `accepted` | all checks passed |
| `rejected` | refused; nothing moved |
| `moving` | a leg started |
| `reached` | the arm is at the target |
| `backoff` | the impact guard tripped and the arm backed off |
| `hold` | the arm stopped where it is |

Usual order: `accepted`, `moving` (once per leg), then `reached`, `backoff` or
`hold`. In `dry`: `accepted`, `hold`. After `accepted` a target can still get
`rejected` (`busy` or `stop`) if the arm could not be taken.

## 4. Modes and consent

| mode | what moves | needs |
|---|---|---|
| `dry` | nothing. All checks run and the planned target shows in the glasses. | nothing; works on any backend |
| `sim` | the simulated arm only | a simulated backend; on hardware every target gets `not armed` |
| `real` | the real arm | consent |

**Consent, command.** `--mode real` without `--i-know` exits with code 2. With
it, consent holds until the command ends. The command also prints a warning.

**Consent, panel.** The box *Okulary moga ruszac prawdziwym ramieniem* must be
ticked. It holds while it is ticked. **Rozlacz okulary** unticks it.
**Polacz z okularami** again (a reconnect) keeps it.

Changing the mode or the box in the panel takes effect at once. A move in
progress, including the 1 s hold-off, ends with `hold`, reason `mode`.

**Trust.** The relay forwards the glasses' registration (the transform from
the glasses' world to the base frame) with its session id and a `trusted`
flag.

- Trusted-only (command with `--trusted-only`; the panel never): the target must
  say `trusted: true`, and its session must match the last trusted
  registration. Once a session is trusted it stays trusted, even if a later
  registration in the same session is not. A registration for a new session
  (the Lens restarted) starts without trust.
- Otherwise (command default): the target's session must match the last
  registration of any kind.

In `real`, losing the relay during a move stops the move (`hold`, `relay`).
In `dry` and `sim` a move carries on.

## 5. The effort log in dry mode

In `dry` mode, while no target is running, the driver logs the arm's joint
efforts once a second:

```
... INFO lerobot_mp.twin.specsreach: Wysilek stawow (dry): {'arm_joint1': 0.4, 'arm_joint2': 6.1, ...}
```

On a real arm the values come from the `0x052` feedback frames, the same
units as the thresholds. With the `sim` backend they are simulated forces and
say nothing about the real arm.

To turn the log into thresholds:

1. Run the panel with the real arm, connect the Spectacles folder in `dry`.
   The log goes to the terminal running `lerobot-twin ui`.
2. Arm the panel and move the arm calmly through the reach box with the
   sliders or the 3D handle. Note the largest effort each joint shows.
3. Repeat with a light push against the arm by hand. Note those values too.
4. Pick each threshold between the calm maximum and the light-push value,
   with room on both sides.
5. Set them in galaxeo, `EFFORT_MOVE` (20) and `EFFORT_DESCEND` (12) in
   `galaxeo/arm/transport.py`, in a local clone installed with
   `pip install -e "../galaxeo-manipulators[arm]"`. One value applies to all
   joints.

Signs of a wrong threshold on the first real session: the backend's "deaf or
blocked" STOP trips before the guard (the guard is too loose), or `backoff`
fires during calm moves (too tight).

## 6. Messages

The driver connects to the relay at the path `/twin`. It reconnects every 2 s
if the relay is not there. `type` is always the first key: the relay reads it
with a regular expression. Positions are in the base frame, in metres.

| message | direction | fields | when |
|---|---|---|---|
| `registration` | relay to twin | `session`, `trusted` (the driver reads only these two) | when the registration changes; the relay also sends its last one when the twin connects |
| `target` | glasses to twin | `source` (`hand` or `object`), `pos` `[x,y,z]`, `session`, `ts`, `trusted`; for objects also `depthMeasured`, `age` (seconds since the detection was seen), `label` | on a pinch |
| `targetState` | twin to glasses | `state`, `reason`, `detail` (both optional), `pos` (the target), `tcp` (tool position now), `ts` (the target's, echoed), `mode` | for every target, at each state |
| `overlay` object `volume` | twin to glasses | `id: "reach"`, `pos` (centre), `size`, `color`, `alpha` 0.35, `label` | once on connect |
| `overlay` object `sphere` `tcp` | twin to glasses | `id: "tcp"`, `pos`, `size` `[0.02]`, `color` | on connect, then every 0.2 s |
| `overlay` object `sphere` `target` | twin to glasses | `id: "target"`, `pos` (hand point, or the point above the object), `size` `[0.025]`, green | on `accepted`; removed with `"remove": ["target"]` 2 s after the final state |

The glasses match a `targetState` to their target by `ts`. Other message types
are ignored. For the full relay protocol see the specs-lens README, section
"Messages".

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| The first target from home drives a long way, or gets `jump` | From home every box point is a 45 to 90 degree turn on some joint. With the default `--max-jump-deg` 150 this is a long drive, not a refusal. `jump` appears only with a lower `--max-jump-deg` or an unusual arm pose. Start with `--start-center`, or bring the arm near the box with the panel, disarm it, then pinch. |
| `untrusted` | No registration has reached the twin yet, or (trusted-only) it is not trusted. Look at the base markers until the glasses say "registration OK". Check the panel status line: `brak zaufanej rejestracji` means none. |
| `session` | The Lens restarted, so its session changed, and the twin still holds the old registration. Look at the markers again: `register.py` starts over for the new session and the twin picks it up. |
| `not armed` | `real` without `--i-know` or the consent box; or `sim` while the twin is connected to the real arm. |
| `busy` | The panel is armed, or a policy or wave holds the arm. Disarm the panel. |
| The driver keeps printing `Przekaznik ws://... niedostepny` | The rig machine cannot reach the relay. Use the Mac's IP address, not `127.0.0.1` (that is the rig machine itself). A `.local` name may not resolve on Linux; use the IP printed in the relay's `Lens serverUrl:` line. Allow incoming connections for `node` in the Mac's firewall. Both machines on the same network, not a guest network. |
| Connecting to the arm fails: no `0x052` frames | The arm has no power, the CAN cable or adapter is out, or another process owns the adapter (a panel, `jog_a1x.py`, the `python -m galaxeo.xcan_usb` check). Close the other process. On Linux bring `can0` up again after replugging. See docs/TWIN.md, Galaxea A1X. |
| The box drifts or sits in the wrong place | The registration is off. Registration error of 1 to 2 cm is expected. Look at the markers from about 40 to 50 cm, check that the plate and tags have not moved, and compare the TCP sphere with the gripper in `dry` before moving anything. |
| After a relay restart nothing is accepted | The twin reconnects by itself within 2 s, but the new relay has no registration yet. Look at the markers until "registration OK". |
| `hold` with `relay` | The relay connection dropped during a real move. Check the network, then pinch again. |
| `backoff` during calm moves | The effort thresholds are too tight. See [the effort log](#5-the-effort-log-in-dry-mode). |

## 8. Tests

```bash
.venv/bin/python -m pytest -q tests/test_twin_specsreach.py tests/test_twin_specsreach_a1x.py tests/test_twin_ui.py
pytest -q                     # the whole suite
pytest -q -m "not render"     # a machine without OpenGL
```

- `tests/test_twin_specsreach.py`: a fake relay (the `/twin` side of
  `server.js`) and the simulated A1X. Reaching a hand target, each refusal,
  object targets above and down with grasp, `dry`, retargeting, session trust,
  `--start-center` in stages, impacts on a wall from effort and from lag, STOP
  and panel preemption mid-move, cutting long legs into pieces, frozen
  feedback, `moved`, relay loss, mode change, a worker exception, scene
  obstacles, and the CLI (help, `--i-know`, a run in sim, a non-A1X
  workspace).
- `tests/test_twin_specsreach_a1x.py`: the real `a1x` backend on a fake CAN
  bus (`FakeA1XBus`). Consent, an effort spike and backoff, a stall at a wall
  tripping the guard before the backend fault, STOP, relay loss, lost or
  frozen feedback, and `--start-center` stalling near the end. Skipped
  without the `galaxeo` package.
- `tests/test_twin_ui.py`: the Spectacles folder refuses a non-A1X workspace;
  a mode change stops a move; **Rozlacz okulary** clears the consent box.

The Spectacles tests need `mujoco`, `websockets` and `galaxeo` (the `twin`
and `a1x` extras). None of them open a CAN interface or a camera.
