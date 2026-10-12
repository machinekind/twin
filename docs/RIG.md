# Glasses and arm: the two-machine runbook

How to set up and run the whole chain: Snap Spectacles, the relay, the digital
twin and the Galaxea A1X arm. Follow the steps in order. What the twin does
with each target is in [SPECS.md](SPECS.md). The glasses side is in the
specs-lens
[README](https://github.com/machinekind/specs-lens/blob/reach-project/README.md).

The chain ran end to end on the glasses and the real arm on 2026-09-26 with
this driver: the twin's own reach box, IK, collision check, ramp and impact
guard. It is still new. Expect surprises and keep a hand on STOP.

## The machines

| machine | what runs there |
|---|---|
| **Glasses machine** (a Mac) | Lens Studio, which pushes the Lens to the glasses; the relay (`server.js`); the registration (`register.py`), which works out where the arm base is in the glasses' world from markers on the base |
| **Rig machine** | the A1X on CAN, the cameras, the digital twin (panel or `lerobot-twin specs`) |

Both machines and the glasses must be on the same Wi-Fi network. Not a guest
network: guest networks usually block traffic between devices.

The chain:

```
glasses --Wi-Fi--> relay on the glasses Mac :8080 <--Wi-Fi-- twin on the rig machine --CAN--> A1X
```

## One-time setup

### Glasses machine (Mac)

1. Install Lens Studio. Log in with the Snap account the Spectacles are paired
   with.
2. Install Node.js and Git LFS, then clone the Lens repository on today's
   branch:

   ```bash
   git lfs install
   git clone -b reach-project git@github.com:machinekind/specs-lens.git
   cd specs-lens/server && npm install
   ```

3. Python for the registration and the doctor (they need numpy, OpenCV and
   websockets). The tools use `~/git/machinekind/digital_twin_training/.venv`
   if it exists. Otherwise make one next to the relay:

   ```bash
   cd specs-lens/server
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

   Then start the tools below with `PYTHON="$PWD/server/.venv/bin/python"`
   in front (run from the `specs-lens` folder; the path must be absolute).
4. In Lens Studio open `lens/Assets/SpectaclesReach/SpectaclesReach.esproj`.
   Start the relay once (step 1 of the start order) and paste its
   `Lens serverUrl:` line into the `serverUrl` input of the `CamStream`
   component. Keep CamStream's **Smaller Dimension** at 480: other values gave
   no frames on the device.
5. Allow incoming connections for `node` when the macOS firewall asks. The rig
   machine must reach port 8080 on this Mac.

### Markers on the arm base

The registration looks for the markers described in
`specs-lens/server/robot.json`. Any one of them is enough; the best one wins.

- A 2 x 2 ArUco plate (dictionary `DICT_ARUCO_ORIGINAL`, 32 mm markers, ids
  305 to 308) flat on the table in front of the base, centred on the arm. Its
  middle is 114 mm forward of the base centre (45 mm out from the front edge
  of the base plate). The markers' x axis points to the arm's left.
- AprilTag 36h11 tags, 37 mm, on flat pieces on the base housing, printed top
  up: id 10 on the arm's left is in the file, measured through the glasses. Id 9
  on the arm's right is on the housing but left out of the file until it is
  measured again (its first measurement pointed the wrong way).

If you move a marker, its pose in `robot.json` is wrong. How to measure it
again: specs-lens README, "Registration targets".

### Rig machine

1. Clone the twin on today's branch and install it with the A1X extra. The
   `a1x` extra pulls the `galaxeo` package from galaxeo-manipulators `main`.
   The twin uses it only for the CAN protocol and the CAN link
   (`galaxeo.protocol`, `galaxeo.bus`).

   ```bash
   git clone -b a1x-twin git@github.com:machinekind/digital_twin_training.git
   cd digital_twin_training
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch==2.11.0
   pip install -e ".[twin,a1x,dev]" -c constraints.txt
   ```

   macOS (XCAN adapter): also `brew install libusb` and `pip install pyusb`.
   The `a1x` extra does not pull `pyusb`.

   If `.[a1x]` fails to install `galaxeo`, install the rest without it
   (`".[twin,dev]"`) and install `galaxeo` from a local clone, as in
   [TWIN.md](TWIN.md#galaxea-a1x), "Instalacja":
   `pip install -e ../galaxeo-manipulators` (macOS:
   `pip install -e "../galaxeo-manipulators[xcan]"`). Check it with
   `python -c "from galaxeo import protocol; protocol.selftest(); import lerobot_mp.robot.a1x"`.
2. Linux only: clone galaxeo-manipulators (`main`) next to the twin. Its
   `./can_up.sh` brings the CAN link up as `can0`. Run it again after every
   replug. On macOS nothing is needed: the port is `xcan`.

   ```bash
   git clone git@github.com:machinekind/galaxeo-manipulators.git
   ```
3. The workspace must be an A1X one. Check with `lerobot-twin workspace`: it
   must say `ramie: a1x`. For a new rig, `lerobot-twin demo --robot a1x`
   writes a simulated A1X workspace to start from.
4. Do the A1X first session in [TWIN.md](TWIN.md#galaxea-a1x) ("Pierwsza sesja
   krok po kroku") once with the panel alone: connect, a small slider move,
   Home. Read galaxeo-manipulators `docs/SAFETY.md` before the arm is powered
   the first time.

## Start order

Do these in order. Each step says what you should see when it works.

### 1. Relay and registration (glasses Mac)

```bash
cd specs-lens
tools/relay-restart.sh
```

It stops anything already on port 8080 and any old `register.py`, then starts
both. You should see:

- `Viewer:         http://localhost:8080/`
- `Twin overlay:   ws://127.0.0.1:8080/twin`
- `Lens serverUrl: ws://127.0.0.1:8080/specs ws://<Mac IP>:8080/specs ws://<name>.local:8080/specs`
- `[restart] relay pid ..., register pid ..., python ..., Ctrl-C stops both`

Write down the Mac IP from the `Lens serverUrl:` line. The rig machine needs
it. Leave this terminal open.

### 2. The Lens on the glasses

In Lens Studio, with `SpectaclesReach.esproj` open, press **Preview Lens**
(USB cable, or Wi-Fi with the glasses showing the system UI). A pushed Lens
keeps running after you unplug the cable. If the glasses drop off USB within
a second, run `tools/lens-studio-usb.sh` with Lens Studio open.

When it works, the relay terminal logs a `[specs] connected from ...` line.

### 3. Registration

Put the glasses on and look at the markers on the arm base from about 40 to
50 cm. Move your head slowly around them for a few seconds.

When it works, the glasses' status line says **registration OK**. Keep the
markers in place from now on.

### 4. The twin (rig machine)

Use **either** the panel **or** the command, never both: only one process may
own the arm.

**Panel:**

```bash
cd digital_twin_training && source .venv/bin/activate
SPECS_RELAY=ws://<Mac IP>:8080/twin lerobot-twin --config configs/local.yaml ui --port 8765
```

(Leave out `--config configs/local.yaml` if you have no local config.) Open
`http://localhost:8765`.

1. Tab **Ramie**: arm `a1x`, port `xcan` (macOS) or `can0` (Linux), then
   connect. The sliders show the measured pose.
2. Tab **Polityki**, folder **Spectacles**: check the relay address,
   **Tryb okularow** `dry`, then **Polacz z okularami**.

When it works, the status line says
`Okulary: polaczone, tryb dry, sesja: <id>`, and the glasses show the blue
reach box in front of the arm with a yellow sphere on the gripper.

**Command** (same result, first in `dry`; the arm is only read):

```bash
cd digital_twin_training && source .venv/bin/activate
lerobot-twin specs --relay ws://<Mac IP>:8080/twin --backend a1x --port xcan --mode dry
```

When it works, it prints
`Ramie: a1x (prawdziwe), tryb dry, przekaznik ws://<Mac IP>:8080/twin` and
logs `Polaczono z przekaznikiem ...`. If it keeps logging
`Przekaznik ... niedostepny`, see [When it does not work](#when-it-does-not-work).

### 5. The doctor (glasses Mac)

In a second terminal on the Mac:

```bash
cd specs-lens
tools/doctor.sh
```

It listens for 6 seconds and prints one line per link: relay, glasses,
frames, markers in view, registration, twin, reach box. When everything
works, the last line is
`All good. Look at the arm, the box should be there; pinch inside it to send a target.`
Otherwise fix the first `BAD` line first.

### 6. Check in dry mode

Still in `dry`: look at the yellow TCP sphere. It should sit on the gripper,
within 1 to 2 cm. Pinch inside the box and take both hands 5 cm out of the
box. The glasses send the point. The target should come back as `accepted`,
then `hold`, and the arm should not move. If the sphere is far off the
gripper, register again (step 3) before anything moves.

### 7. The first move

Stop the dry run first (Enter in the command's terminal, or
**Rozlacz okulary** in the panel). Then, with the command:

```bash
lerobot-twin specs --relay ws://<Mac IP>:8080/twin --backend a1x --port xcan --mode real --i-know --start-center
```

1. It prints a safety warning, then asks for Enter. Press Enter only when
   nothing is near the arm.
2. The arm moves in stages at 10 degrees per second to a pose above the centre
   of the reach box.
3. Pinch a point near the centre of the box. Take both hands out of the box:
   the point is sent only when both hands are 5 cm outside it.
4. After 1 s the arm moves there. The glasses show `moving`, then `reached`.

Without `--start-center` the first target from home is not refused: it is one
long move of more than 45 degrees on some joint at 20 degrees per second.
Use `--start-center` for the first run.

Keep your hands and objects away from the arm except at the pointed spot. One
target at a time. Use `--grasp` only after several clean `reached` above
objects.

With the panel instead: move the arm near the box with the panel, disarm it
(**UZBROJ** off, or every target gets `busy`), choose `real` and tick
**Okulary moga ruszac prawdziwym ramieniem**.

### STOP

- Terminal running `lerobot-twin specs`: **Enter** or **Ctrl-C**.
- Panel: the red **STOP** button at the top.
- The glasses **cannot** stop the arm.
- The power supply's emergency switch cuts power, and **the arm falls**: the
  A1X has no brakes. Use it only if the other STOPs fail.

## Stop order

1. Stop the twin driver. Command: **Enter** (STOP, then it exits). Panel:
   **Rozlacz okulary**. The arm holds its measured pose. The motors stay on:
   the twin never releases them.
2. Before cutting power, fold the arm and support it, because it has no brakes.
   Use the panel (after the command has exited, start the panel and connect
   the arm; after a panel STOP, press **Skasuj STOP** first). Arm the panel
   and set J2 to 3 degrees, J3 to −3 degrees, the rest to 0 (docs/TWIN.md, A1X
   first session, step 2). Then disconnect the arm in the panel and switch
   the power off while you support the arm.
3. Stop the relay and the registration: **Ctrl-C** in the `relay-restart.sh`
   terminal stops both.
4. Close the Lens on the glasses.

After the twin exits, the arm keeps holding the last pose it was sent, with
no stream. It stays there until the power goes off.

## When it does not work

1. Run `tools/doctor.sh` on the glasses Mac. Fix the first `BAD` line. The
   later ones often follow from it. If the relay runs elsewhere:
   `tools/doctor.sh --relay ws://<relay IP>:8080`.
2. A refusal or a `hold` from the twin: look up the reason in
   [SPECS.md, Troubleshooting](SPECS.md#7-troubleshooting) and
   [SPECS.md, What happens to a target](SPECS.md#3-what-happens-to-a-target).
3. The glasses side (no frames, USB, registration, the Lens): the specs-lens
   README, its "Troubleshooting" table first, then "Lens Studio and the
   glasses" and "Registration targets".
4. The arm on CAN (no `0x052` frames, link down): [TWIN.md](TWIN.md#galaxea-a1x),
   "Sprzęt i łącze CAN".

Common first problems:

- The rig machine cannot reach the relay. Use the Mac's IP address, not
  `127.0.0.1`. `.local` names may not resolve on Linux. Allow `node` in the
  Mac firewall. Check that neither machine is on a guest network.
- After `relay-restart.sh` the twin reconnects by itself within 2 s, but the
  registration starts over. Look at the markers again until
  **registration OK**.
- Targets get `session` after the Lens restarted: look at the markers again.
