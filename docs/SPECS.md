# Spectacles driver

The Spectacles driver lets a person wearing Snap Spectacles point at a spot and
have the Galaxea A1X arm move there. This page covers the twin's side: the
`lerobot-twin specs` command, the Spectacles folder in the panel, and the code
in `src/lerobot_mp/twin/specsreach.py`. For the full two-machine setup with
the glasses, follow [RIG.md](RIG.md).

This driver moved the real A1X from the glasses on 2026-09-26. It is also
tested in simulation, with a fake relay and a fake CAN bus.

## 1. What it is

- The glasses send a point in the arm's base frame. The point is either a hand
  pinch inside the reach box or the centre of an object the glasses detected.
- The twin checks the point and, if every check passes, moves the arm there.
- The twin does all of this with its own code:
  - `twin/reach.py`: the reach box.
  - `twin/kinematics.py`: IK (inverse kinematics: joint angles for a tool
    position), computed by MuJoCo on the same model the twin simulates.
  - `twin/collision.py`: the collision checker.
  - `twin/impact.py`: the impact guard.
  - `twin/specsreach.py`: the checks, the ramp, the trim, the backoff, the
    speed caps, arm ownership, STOP, the three modes, consent for the real
    arm and the relay protocol.
  - `twin/ui/specs_panel.py`: the panel folder.
- The `galaxeo` package (repository galaxeo-manipulators) is used only by the
  `a1x` backend, for the CAN protocol and the CAN link (`galaxeo.protocol`,
  `galaxeo.bus`).
- The driver is written for the A1X. It also runs on the SO-101 in
  simulation (tests in `tests/test_twin_specsreach_so101.py`).

```
glasses (Lens) --/specs--> relay (server.js) --/twin--> twin driver --> Twin --> sim or A1X on CAN
```

The relay and the Lens live in the specs-lens repository
([README](https://github.com/machinekind/specs-lens/blob/reach-project/README.md),
sections "Robot frame" and "Reach box").

**Base frame.** Origin at the centre of the arm base on the joint 1 axis,
table top at z = 0, x forward, y to the arm's left, z up, metres.

**Reach box.** From `DEFAULTS` in `twin/reach.py`. For the A1X gripper it is
x 0.20 to 0.50 m, y −0.30 to 0.30 m, z 0.05 to 0.30 m. A `reach` key in
`workspace/twin.json` (`{"lo": [x, y, z], "hi": [x, y, z]}`) replaces it.

**What the collision check sees.** The checker builds its own MuJoCo scene
from the workspace: the floor, the table and the arm. It does not see task
objects in the twin scene, the placed cameras (they have no body in the
scene) or anything else on the table. Contacts that already exist in the home
pose (the base on the table top) are allowed. Every new contact is a
collision. The scene is rebuilt when the twin's scene changes.

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
| `--max-jump-deg DEG` | Refuse a target (`jump`) when any joint would turn more than this on one leg. Default 150. A leg is always driven as one move; it is never cut into pieces. |
| `--grasp` | Object targets: go down, close the gripper and come back up. Without it the arm only stops above the object. |
| `--workspace FILE` | The workspace file. Default `workspace/twin.json`. |
| `--seconds S` | Stop after S seconds (used by tests). |
| `--config FILE` | YAML config such as `configs/local.yaml` (A1X joint offsets and gripper settings). Also accepted before the command: `lerobot-twin --config FILE specs ...`. |

**Why `--start-center`.** The A1X home pose is far from the reach box. From
home, every point in the box needs more than 45 degrees on some joint. That
is below the default `--max-jump-deg` of 150, so the first target is not
refused. It is driven as one long move: a straight line in joint space at 20
degrees per second on the fastest joint, which takes several seconds. With
`--start-center` the arm first moves above the box centre in stages (joint
groups one after another, as the panel's Home does), at 10 degrees per second,
under the impact guard with tighter limits. Every later target then starts
close to its goal. Use `--start-center` for the first run in `real` mode.

- In `real` mode it asks for Enter first. Ctrl-C at that prompt ends the
  command and the arm holds its measured pose.
- On a real arm it is only allowed with `--mode real`. With `--mode dry` or
  `--mode sim` on the real arm the command refuses and exits with code 2.
- If the move fails, the command prints the reason and, on a real arm, exits
  with the arm holding its measured pose. An impact during this move only
  holds; there is no backoff.
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

`SPECS_RELAY` pre-fills the relay field; `SPECS_CONNECT=1` also presses
**Polacz z okularami** at start (mode `dry`, the arm stays put). `--port 8765` keeps the panel off
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
grasp. It does not use `--max-jump-deg` (it uses the default, 150). On trust
both behave alike: like the command without `--trusted-only`, the panel
accepts any registration the glasses use.

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
| `ik` | IK found no joint angles within 3 mm, or, for an object, the gripper would point more than about 15 degrees (0.26 rad) away from straight down |
| `jump` | a joint would turn more than `--max-jump-deg` on one leg |
| `collision` | the straight joint-space path from the current command to the goal hits the arm itself or the table |

`ik`, `jump` and `collision` carry a `detail` (IK error in mm, the jump in
degrees, the colliding pair and how far along the path).

**IK.** IK starts from the arm's current command, from home and from six
random poses, prefers the solution nearest the arm's current pose, and skips
solutions the collision check refuses. If every solution it finds collides,
the target ends as `collision`.

**Path check.** The path is sampled every 0.04 rad (about 2.3 degrees) of
the largest joint move, so a pose that is clear at both ends but crosses the
table on the way is refused.

**Legs.** A hand target is one leg: the tool goes to the point, in any
orientation. An object target goes to a point 8 cm above the object (at most
3 cm below the top of the box), with the gripper pointing down. With `--grasp`
two more legs follow: down to 1 cm above the centre at half speed, then close
the gripper and go back up along the same path. Each leg is checked from the
end of the one before it.

### After the checks

1. The driver sends `accepted` and shows a green `target` sphere in the
   glasses. In `dry` it then sends `hold` and stops there.
2. **Hold-off, `real` mode, hand targets only.** The driver waits 1 s. A new
   target in that second ends this one with `hold`, reason `retargeted`, and
   the new one runs. A mode change ends it with `hold`, reason `mode`. Losing
   the relay ends it with `hold`, reason `relay`. STOP pressed in that second
   is caught in the next step.
3. It takes the arm as owner `specs`. If it cannot, the target gets
   `rejected` with `busy` or `stop`.
4. **Re-check (`sim` and `real`).** If any joint of the arm's command changed
   by 0.5 degrees or more since the checks (someone moved a panel slider), the
   target ends with `hold`, reason `moved`, and `detail` gives the change.
   Nothing has moved. The target is not planned again: pinch again.
5. It sends `moving` and drives the legs. It sends `moving` again at the start
   of each later leg.

### The move

Each leg is the driver's own ramp. Every tick is `Twin.set_target` as owner
`specs`, so the twin's backend stays the only writer on the CAN bus, and STOP,
Home and the panel work as for any other owner.

- The ramp starts from the arm's current **command** and runs at 50 Hz. The
  command follows a smoothstep curve.
- Speed: 20 degrees per second on the fastest joint, 10 on the `--grasp`
  legs down and up. A leg takes at least 0.5 s. The ramp clock pauses while
  the twin's loop is frozen.
- While the arm moves, the twin's speed limit drops to 35 degrees per second
  (1.5 times the ramp speed, the smoothstep peak, plus 5). It drops to 15
  during a backoff. It never goes above the panel's value and goes back to it
  afterwards.
- Arrival: every joint within 2 degrees of the goal and slower than 10
  degrees per second. A joint that is not there 3 s after the ramp ends makes
  the arm hold its measured pose and ends the target with `hold`, reason
  `not arrived`; `detail` lists how far each joint is off.
- **Trim.** A servo under load settles 1 to 2 degrees off its command. After
  arrival the driver adds the error to the command, up to 4 times, 0.4 s
  apart, until every joint is within 0.2 degrees. The command stays within
  2.5 degrees of the IK goal. There is no trim on the `--grasp` legs down and
  up, and none when a joint's effort is above half its threshold (that is
  already pressing on something).
- A new target during the ramp interrupts it. The target ends with `hold`,
  reason `retargeted`, and the new one starts from the current command.

### The impact guard

It listens to every tick of the twin's loop during the move. Ticks where
nothing was sent or the feedback is not fresh are ignored. Two rules:

| rule | trips when |
|---|---|
| lag | the command keeps moving but the joint does not: command and joint more than 6 degrees apart for 0.15 s while the joint moves slower than 15 degrees per second. 3 degrees on the `--grasp` legs. After the ramp ends, while arriving: 2.5 degrees for 0.3 s. `--start-center` uses 4 degrees during the ramp and 2.5 degrees for 0.2 s while arriving. |
| effort | a joint's effort is above the threshold for 3 samples in a row. A1X: 20 on moves, 12 on the `--grasp` legs, in the raw units of the `0x052` feedback frame, the same for all six arm joints. Simulation: 0.8 (0.5 on the `--grasp` legs) times the joint's force range. SO-101 in simulation: no effort rule. |

A joint that is already above half its effort threshold at the start of a leg
gets only the lag rule for that leg (a warning goes to the log). A backend
that cannot report effort gets only the lag rule.

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
   0.4 s before the trip whose farthest joint is at least 3 degrees away.
   Without one, to the oldest good pose of the last 3 s. A good pose is a
   tick where no joint was lagging or over its effort threshold.
3. The target ends with `backoff`, reason `impact lag arm_joint2` or
   `impact effort arm_joint2` (the rule and the joint).
4. Targets that were waiting in the queue get `rejected` with `busy`, so the
   wearer points again.

The backoff has its own guard (6 degrees for 0.15 s, the normal effort
thresholds), armed after 0.3 s: a joint pressed on an obstacle still shows
high effort at first. A second trip during the backoff only holds: the target
ends with `hold` and the same `impact ...` reason. A backoff is not a STOP:
the next pinch works.

Other ways a backoff ends:

- No backoff pose, or the arm is no longer owned by `specs`: `hold`, reason
  `impact ...`.
- The arm is taken away, disconnects or does not arrive during the backoff:
  `hold`, reason `impact ...; stop`, `impact ...; preempted`,
  `impact ...; not connected` or `impact ...; not arrived`.
- A new target, a mode change, lost relay or STOP during the backoff: `hold`
  with the raw cause `retarget`, `mode`, `relay`, `preempt` or `stop`.

### Hold reasons

`hold` means the arm stopped where it is, holding its measured pose.

| reason | cause |
|---|---|
| (none) | `dry` mode: checks passed, nothing moved |
| `retargeted` | a new target arrived; the new one runs from where the arm is |
| `moved` | the command changed between the checks and taking the arm |
| `mode` | the mode or the consent changed in the panel during the move |
| `relay` | the relay connection dropped during a move in `real` mode |
| `not connected` | the arm disconnected during the move |
| `stop` | STOP, or the driver was stopped |
| `preempted` | something else took the arm: Home, the panel, another owner |
| `not arrived` | a joint did not reach the goal 3 s after the ramp ended |
| `impact ...` | a second trip during a backoff, or a backoff that could not run |
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

- Default, in the command and in the panel: the target's session must match
  the last registration of any kind. The target's own `trusted` flag is not
  looked at.
- Trusted-only (command with `--trusted-only`): the target must say
  `trusted: true`, and its session must match the last trusted registration.
  Once a session is trusted it stays trusted, even if a later registration in
  the same session is not. A registration for a new session (the Lens
  restarted) starts without trust.

In `real`, losing the relay during a move stops the move (`hold`, `relay`).
In `dry` and `sim` a move carries on.

## 5. The effort log in dry mode

In `dry` mode, while no target is running, the driver logs the arm's joint
efforts once a second. When the arm moves, it also logs them at the start of
each leg.

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
5. Set them in the twin: `SpecsReach.real_effort_abs` (20) and
   `SpecsReach.real_effort_abs_descend` (12) in
   `src/lerobot_mp/twin/specsreach.py`. One value applies to all joints.

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
| The first target from home drives a long way, or gets `jump` | From home every box point is more than 45 degrees on some joint. With the default `--max-jump-deg` 150 this is one long move at 20 degrees per second, not a refusal. `jump` appears only with a lower `--max-jump-deg` or an unusual arm pose. Start with `--start-center`, or bring the arm near the box with the panel, disarm it, then pinch. |
| `untrusted` | No registration has reached the twin yet, or (`--trusted-only`) it is not trusted. Look at the base markers until the glasses say "registration OK". The panel status line shows only trusted sessions: `brak zaufanej rejestracji` means no trusted one, even when an untrusted one is accepted. |
| `session` | The Lens restarted, so its session changed, and the twin still holds the old registration. Look at the markers again: `register.py` starts over for the new session and the twin picks it up. |
| `not armed` | `real` without `--i-know` or the consent box; or `sim` while the twin is connected to the real arm. |
| `busy` | The panel is armed, or a policy or wave holds the arm. Disarm the panel. |
| `collision` near an object | The checker does not know about objects on the table, so this is the arm or the table. Check `detail` for the pair. |
| The driver keeps printing `Przekaznik ws://... niedostepny` | The rig machine cannot reach the relay. Use the Mac's IP address, not `127.0.0.1` (that is the rig machine itself). A `.local` name may not resolve on Linux; use the IP printed in the relay's `Lens serverUrl:` line. Allow incoming connections for `node` in the Mac's firewall. Both machines on the same network, not a guest network. |
| Connecting to the arm fails: no `0x052` frames | The arm has no power, the CAN cable or adapter is out, or another process owns the adapter (a panel, `jog_a1x.py`, the `python -m galaxeo.xcan_usb` check). Close the other process. On Linux bring `can0` up again after replugging. See docs/TWIN.md, Galaxea A1X. |
| The box drifts or sits in the wrong place | The registration is off. Registration error of 1 to 2 cm is expected. Look at the markers from about 40 to 50 cm, check that the plate and tags have not moved, and compare the TCP sphere with the gripper in `dry` before moving anything. |
| After a relay restart nothing is accepted | The twin reconnects by itself within 2 s, but the new relay has no registration yet. Look at the markers until "registration OK". |
| `hold` with `relay` | The relay connection dropped during a real move. Check the network, then pinch again. |
| `backoff` during calm moves | The effort thresholds are too tight. See [the effort log](#5-the-effort-log-in-dry-mode). |

## 8. Tests

```bash
.venv/bin/python -m pytest -q tests/test_twin_specsreach.py tests/test_twin_specsreach_a1x.py \
    tests/test_twin_specsreach_so101.py tests/test_twin_reach.py tests/test_twin_impact.py tests/test_twin_ui.py
pytest -q                     # the whole suite
pytest -q -m "not render"     # a machine without OpenGL
```

- `tests/test_twin_specsreach.py`: a fake relay (the `/twin` side of
  `server.js`) and the simulated A1X. Reaching a hand target, the refusals
  `outside volume`, `session`, `no depth`, `stale` and `untrusted`, object
  targets above and down with grasp, `dry`, retargeting, session trust,
  `--start-center` in stages, impacts on a wall from effort and from lag, STOP
  mid-move, `moved`, relay loss, mode change, a worker exception, and the CLI
  (help, `--i-know`, a run in sim).
- `tests/test_twin_specsreach_a1x.py`: the real `a1x` backend on a fake CAN
  bus (`FakeA1XBus`). Consent (`not armed`), an effort spike and backoff, a
  stall at a wall tripping the guard before the backend fault, STOP, relay
  loss, and `--start-center` stalling near the end. Skipped without the
  `galaxeo` package.
- `tests/test_twin_specsreach_so101.py`: the SO-101 in simulation, moves
  without false trips and a wall tripping the lag rule.
- `tests/test_twin_reach.py` and `tests/test_twin_impact.py`: the reach box
  and the impact guard on their own.
- `tests/test_twin_ui.py`: a mode change stops a move; **Rozlacz okulary**
  clears the consent box.

The Spectacles tests need `mujoco` and `websockets` (the `twin` extra); the
A1X ones also need `galaxeo` (the `a1x` extra). None of them open a CAN
interface or a camera.
