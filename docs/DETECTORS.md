# Detector guide

How the twin runs detectors and tracks the objects they report, the LLM
detector and the seeded detector. [The workstation guide](TWIN.md) covers the
station itself.

## Detectors and objects

A detector is the fourth plugin kind, next to robots, backends and strategies.
It is a class in an installed package, registered in the
`machinekind_twin.detectors` entry point group. It reads camera frames and
reports what it sees **in the image** and where. The twin turns these image
positions into poses and tracks them. Strategies read the result as
`Observation.objects[key]`, an `ObjectPose` in the robot's base frame with
`source` and `age_s`. The key is the label, such as `red cube`; further objects
with that label are `red cube 2`, `red cube 3`, and so on. The contract lives in
[`machinekind_twin_api.detector`](../packages/machinekind_twin_api), and that
package's README has a minimal detector.

### The contract

The twin calls `detect(request)` on a dedicated thread, one call at a time. A
call starts at least `interval_s` after the previous one started. Nothing waits
for a detector, so a slow detector only makes its poses older.

`request.frames` maps camera names to frames. It offers the cameras that the
twin can place detections from at that moment. Such a camera is enabled and
fixed to the base, and its calibrated pose is trusted or the camera is
simulated. The twin builds the set again for every call, so a camera calibrated
during a run is offered from the next call on. A frame is fetched only when
read. Iterating and `len` list the offered cameras. `name in frames` and
`frames.get(name)` fetch the frame and tell whether the camera has one younger
than a second. `frames[name]` raises `KeyError` when it has none. Images are
raw RGB and never undistorted.

Each `Detection` carries a label, a camera and a pixel. The pixel follows the
OpenCV convention in the frame's own resolution. A detection may also carry a
box, a confidence, a boolean silhouette `mask`, `height_m`, `size_m` and its own
pixel error. `height_m` is the height above the table of the point the pixel
marks, and `size_m` is the object's size. Labels are compared after stripping
whitespace and lower-casing. Several objects may share a label: a detector
reports each in a detection of its own. The twin drops a detection that names an unknown
camera, has a pixel outside the frame or has a mask of the wrong shape. The
Detectors tab shows how many it dropped.

`start(info)` runs once before the first call and may raise to refuse the
start. `stop(reason)` runs once after a successful start. It may run while
`detect` is still running on the detector thread, and the twin discards the
late answer. When `detect` raises, the panel shows the error and the twin
retries after a wait. The wait doubles with each consecutive failure, up to
300 s.

### Where an object is

The twin first sorts each call's detections into objects (see
[Several blocks of one label](#several-blocks-of-one-label)): at most one
detection per camera for each object. It places each object in one of three
ways, and the Detectors tab shows the Polish name of the way it used:

| Path | When | Result |
|---|---|---|
| silhouette (`sylwetka`) | Every camera gave a mask of a full-size frame, and `size_m` is given | A box of `size_m` resting on the table or on another block, fitted to all masks at once. It gives the position, the yaw (`rotation[:, 0]` along `size_m[0]`) and the fit's IoU as confidence |
| triangulation (`triangulacja`) | Two or more cameras whose rays pass within the detectors' pixel error, scaled by depth | The point nearest to all rays. It must lie in front of every camera and between 3 cm below and 60 cm above the table |
| one ray (`promien`) | One camera, or rays that disagree | The most confident detection's ray meets the plane at `height_m`. Without `height_m` the plane is at half of `size_m`, and without both it is the table top |

The geometry uses the calibrated camera poses with lens distortion, also in
simulation. Simulated perception therefore tests the calibration. A placed
object's `t` is the capture time of the oldest frame used to place it.

Only the silhouette path gives yaw, so a detector that knows an object's size
should send it with a mask, whatever the object's shape. The fit takes
`size_m` as the box's x, y and z sides, z being its height above the table. The
yaw is folded by the footprint's symmetry: into [-45, 45) degrees for a square
footprint such as a cube, into [-90, 90) otherwise. A 50x25x25 mm block lies
within 0.5 mm and 1 degree of its true pose on synthetic masks from two cameras
at about 45 degrees, and within 1.3 mm and 1 degree in the simulated stand.

### Blocks on blocks

The fit first places the box where the block last rested: on the table, or on
the block it was last seen on. When that fit is not clearly good (IoU under
0.93, or one camera), it also searches the height of the surface under the box,
up to 15 cm, starting where the cameras' mask centroids meet. The searched fit
wins when two cameras see the box and its IoU is higher by 0.02, and, off the
table, at least 0.85. A height within 4 mm of the table is the table.

A box fitted off the table must rest on another block: its bottom within 6 mm
of that block's top and its centre over that block. The block under it is
placed in the same detector call, or tracked. Otherwise nothing is placed. One
label's masks from two different blocks can otherwise meet in a box in the air.

One camera cannot tell the height. A tracked block seen by one camera keeps the
height it last rested at. A new block seen by one camera is not placed while it
could rest on another block: its mask touches another object's mask, or the
camera's ray through it passes over another block. It is placed when two
cameras see it.

A block partly hidden by another one, such as the lower block of a stack, does
not cover its own mask in every camera. When its fit fails, the twin fits it
again with the other objects' masks in that camera treated like the arm: pixels
it may be hidden behind.

Blocks whose masks touch are then fitted again together, twice. Each box starts
where it is and is fitted to the touching masks taken together, with the other
boxes' silhouettes treated like the arm. The boxes end up covering the masks
between them, wherever a detector drew the line between two blocks. A box
resting on another one takes that block's top as its base. A box that would
move more than 1 cm keeps its first fit. A block whose own masks gave no fit
and that is not tracked yet, but whose masks touch a placed box, is fitted to
its masks joined with that box's, the box's silhouette treated like the arm. It
is tried resting on each such box, highest first, then on the table. So a block
of one colour lying across another is placed even when the detector's line
between them is poor.

The tracker knows stacks too. A closing jaw takes only the object nearest the
TCP, never the block under it. An object that leaves the jaws is put on the
highest other tracked block under it, else on the table.

In the simulated stand, cubes and cuboids stacked two and three high are
placed within 3 mm and 5 degrees of the physics (`tests/test_stacked_blocks.py`).

### Several blocks of one label

A camera that reports a label twice sees two objects. Two reports of one label
in one camera are one object reported twice, and the more confident one counts,
when their masks overlap by half (IoU 0.5), or their boxes do and the boxes'
centres lie within 0.15 of the shorter box side. A block resting on another of
its size has a box that overlaps the lower one's by more than half, but its
centre lies a block's height higher. The cameras' detections of a
label then pair by geometry: two detections are one object when their rays
meet within three times the pixel error's tolerance. The closest pairs go
first, and a detection that pairs with nothing is an object of its own. Each
object is placed from its own detections only, so the masks of two blocks never
go into one fit. Two objects of a label are one when their centres lie within
half the block's narrower side across and within half its height up, and the
one more cameras saw stays. A flat block on another of its label sits one
block's height higher and stays a block of its own. One detection per camera
without masks, with at most one such object tracked, is one object, as before.

Tracking keys each object. A measurement continues the track of its label that
lies nearest to it, within 6 cm, whatever order the detector listed it in. A
measurement left over takes a track of its label that no measurement of the
call is near, so a block lifted away by hand keeps its key. Otherwise it starts
a track under the lowest free key: `red cube`, then `red cube 2`. One track and
one measurement of a label always pair, however far apart. A key names one
block while it moves, rides in the gripper and is put down elsewhere.

In the simulated stand, two red cubes apart, touching, or one on the other, a
red cuboid flat on a red cuboid, along it or across it, a red cube touching a
red cuboid, and a block carried onto the other of its label, all followed by
the seeded detector, are placed within 3 mm and 5 degrees of the physics under
keys of their own (`tests/test_block_instances.py`).

### The example cube detector

The example detector in `examples/detectors/cube` replaces the Map tab's old
built-in cube search. Its package is `machinekind-twin-cube`, and `uv sync`
installs it. It thresholds one colour in HSV with the ranges and the opening
that the old `CubeDetector.mask` used. For each camera it reports the largest
blob's centroid, box and mask, and the cube's size. The twin runs the old
multi-view silhouette fit on those masks. The one change is the fit's input.
The masks come from the raw frames and are then warped to the table, where the
old pipeline thresholded the warped colour map. In simulation both pipelines
place the cube within 2 mm and 7 degrees of its true pose.

Its settings are `color` (red, green, blue or yellow), `size_m`, `label`,
`min_area_px` and `interval_s`. `label` defaults to `cube`, the key that
soarm-lab's policy reads. `interval_s` defaults to 0.33 s, the old Map tab's
rate.

### Tracking, `source` and `age_s`

Each object gets its own copy of the twin's cube tracker,
`perception.ObjectTracker`. The tracker also follows an object held in the
gripper. Each measurement is fed to it once, and only when its confidence is
above 0.4. A track is dropped after `hold_s` without a measurement. `hold_s` is
the larger of 6 s and three times the sum of the detector's interval and
latency, so a slow detector's object survives between answers.

`age_s` is the time since the capture of the frames the pose comes from. It is
0 while the object follows the jaws. `source` says what the pose is based on,
and it takes these Polish values:

- `kamery` (cameras) means the cameras saw the object recently. The value holds
  while the detector's latest answer contains the object and
  `age_s <= fresh_s`. `fresh_s` is twice the detector's interval, kept between
  0.5 and 1.0 s. The cube detector at 0.33 s keeps `kamery` for 0.66 s after
  capture.
- `ostatnie widziane` (last seen) means the pose is the last known one. The
  source switches to it when `age_s` exceeds `fresh_s` or when the detector's
  latest answer lacks the object. A measurement that is already older than
  `fresh_s` when it arrives gets this value at once. A detector that answers in
  tens of seconds, such as a language model, therefore never publishes
  `kamery` from a new answer. The LLM detector publishes `kamery` only when it
  repeats an answer for frames that did not change. "Freshness" under "LLM
  detector" explains why. A strategy reads `age_s` to decide whether a pose is
  too old.
- `w dloni` (in hand) means the gripper holds the object and the pose follows
  the TCP. On a connected arm, the closed jaw position comes from the
  hardware's limit, so an empty closed gripper does not count as a grasp.
- The tracker can annotate a value, as in `ostatnie widziane (upuszczona)`
  (last seen, dropped). Annotated values pass through unchanged.

The tracker's jaw thresholds, such as the 0.02 rad `still` test, apply per
update. Their effect therefore depends on how often the tracks are stepped.

### Threads and who steps the tracks

| Thread | Rate | Does |
|---|---|---|
| `detektor:<name>`, one per running detector | `interval_s`, longer after failures | Reads frames, calls `detect`, places the answer and feeds it to the tracks |
| `detektor-start:<name>`, short-lived | Once per start | Creates the detector, calls its `start` and starts its thread |
| strategy runner | The strategy's `hz` | Reads `Observation.objects`, which steps the tracks |
| panel loop | 5 Hz | Steps the tracks when nothing else did for 0.15 s, then draws them |

The tracks have no thread of their own. Each read of the objects steps them, at
most once per 10 ms. A running strategy steps them at its `hz`. Without one,
the panel steps them at 5 Hz. The Detectors tab, the 3D markers and `w dloni`
therefore keep working while the operator moves the arm from the panel. Reading
the objects never waits for a detector.

### Running detectors

Start a detector in the **Detektory** (Detectors) tab. The config section with
the detector's name prefills its form. Starting a running detector again
restarts it with the new settings. Several detectors can run at once. When two
of them report the same object, the measurement with the newer capture time
wins. Detectors never move the arm, so STOP leaves them running. Closing the
panel stops them.

Starting shows `Startuje...` immediately. During each detection call, the panel
shows `Wywolanie w toku` with elapsed seconds beside Start and in the detector's
row. Previous results stay visible while the next answer is pending.

`twin ui --detector cube` starts the named detector when the panel starts. It
uses the settings from the detector's config section, like the **Uruchom
detektor** (Start detector) button. Repeat the flag to start several
detectors. When a detector fails to start, the panel logs the error, shows it
in the detector's row and keeps running.

Loader messages and panel labels are in Polish, like the rest of the panel. The
panel shows a plugin's exception text verbatim after a Polish label such as
`blad:` (error:).

### Limits

- Detectors only get cameras fixed to the base. A camera on the arm is never
  offered.
- An object placed by triangulation or by one ray has no orientation, so its
  rotation is the identity. Only the silhouette fit gives yaw; it needs a mask
  and `size_m` from every camera that saw the label.
- The box is fitted resting on the table or on a block in the pose `size_m`
  describes: `size_m[2]` is its height. A cuboid stood on its end, or a
  cylinder on its end, with the lying size listed is gated out (IoU below
  0.75) and the label shows `ostatnie widziane`. List the size as the block
  rests.
- A cylinder is fitted as its bounding box. Lying on its side, it fits worse
  than a box (IoU 0.74-0.82 at the true pose from 45-degree cameras) and may
  fail the 0.75 gate at some yaws.
- A cuboid lying across two blocks, as a bridge, is not placed: a box off the
  table needs its centre over one block.
- The seeded detector shares a blob of one colour out by the model's points.
  For a flat block lying on another of its colour those points lie only a few
  pixels apart, so a model that misplaces them by that much splits the blob
  poorly. The twin then places the pair only once it can fit one of the two.
- A block that reappears far from where it was seen, while another block of
  its label is out of view, may take that block's key.
- A block resting on a block that no detector places, such as bare wood on a
  wooden table, gets no pose.

## LLM detector

[`machinekind-twin-llm`](../packages/machinekind_twin_llm), installed by
`uv sync`, registers the detector `llm`. Each call sends the cameras' frames to
a vision model and reads back what is where in each frame. The model sits
behind a CLI installed on the machine, `codex` (Codex CLI) or `claude` (Claude
Code), and answers on the subscription that CLI is logged in with. The package
holds no API key, uses no model SDK and opens no network connection itself:
every question is one headless run of the CLI as a subprocess. Its README has
the exact command lines and a smoke check to run after a CLI update.

The panel never starts it by itself. Start it in the **Detektory** tab or with
`twin ui --detector llm`, after filling an `llm:` section in the config
(`configs/default.yaml` has a commented example).

### Subscription and login check

`start` finds the CLI (`executable`, else `cli` on PATH) and asks it how it is
logged in (`claude auth status`, `codex login status`), with a 15 s timeout. It
refuses a CLI that is not logged in and one that uses an API key. Any other
login passes and is named in the log: a claude.ai login, a `claude setup-token`
token (`CLAUDE_CODE_OAUTH_TOKEN`), a ChatGPT login for codex. The refusal shows
in the detector's row and the detector does not start. The check runs on the
start thread, so the panel never waits for it.

Every question runs the CLI with API keys, base-URL overrides and the variables
of a calling Claude Code or cmux session removed from its environment, so it
cannot fall back to a key or route through a proxy. When `cli` on PATH is a
terminal wrapper (a cmux shim), point `executable` at the real binary, e.g.
`~/.local/bin/codex`.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `cli` | `codex` | `codex` or `claude` |
| `model` | `gpt-6-astra` | the model as the CLI names it (`sonnet`, `claude-opus-5-5`, ...); `""` = the CLI's default. A `gpt-` model needs codex, a Claude model needs claude |
| `effort` | `low` | `low`, `medium`, `high`, `xhigh` or `max`; `""` = the CLI's default |
| `objects` | `""` | labels to look for, comma-separated (`red pen, cube`); `""` = every movable object, labelled by the model in short lowercase English |
| `cameras` | `""` | cameras to send, comma-separated; `""` = every camera offered in each call |
| `prompt` | `""` | extra instructions appended to the built-in prompt |
| `interval_s` | 10 | shortest time between the starts of two questions [s], at least 2 |
| `timeout_s` | 120 | the CLI's process group is killed after this [s], 5 to 900 |
| `max_side_px` | 1280 | long edge of the images sent [px]; answers are scaled back to each frame |
| `pixel_error_px` | 25 | how far off the model's pixels may be, in the sent image's pixels |
| `reuse_if_still` | true | repeat the last answer instead of asking while no camera's image changes. "Cost and rate" says what counts as a change |
| `reuse_max_age_s` | 60 | longest time after a question that its answer may be repeated [s], 0 to 3600 |
| `max_calls_per_hour` | 0 | cap on questions per hour; 0 = no cap |
| `min_confidence` | 0.3 | answers the model is less sure of are dropped |
| `executable` | `""` | the CLI binary; `""` = `cli` on PATH |
| `claude_images` | `stream` | claude only: `stream` sends images inline, `read` has the model open temp files with its Read tool (slower) |

A value the detector cannot work with refuses it when it is created, with a
message naming the setting. With `objects` set, an answer is kept only when its
label is one of them or ends with one ("the red pen" counts as `red pen`), so
label drift does not create stray tracks.

The prompt asks for one entry per object per image, with a box each, also when
objects of one label touch or stand on each other. A number the model appends
to a label (`red cube 2`) is dropped; the twin numbers the objects itself.

### Cost and rate

Every question counts against the CLI subscription's usage limits. At the
default 10 s, a scene that keeps changing costs up to 360 questions an hour. A
still scene costs 60. Five things bound it:

- one question in flight at a time;
- `interval_s`, at least 2 s;
- `reuse_if_still`, on by default. While no camera's image changes, a call
  repeats the last answer instead of asking. The check turns each new frame and
  the frame last sent into grey thumbnails 256 px long, blurred 3x3. It matches
  their mean brightness, so auto exposure does not count. An image has changed
  when more than 15 thumbnail pixels differ by 16 grey levels or more. A 3 cm
  cube at 1 m through a 60 degree lens covers about 45 thumbnail pixels. The
  check therefore sees the cube leave or move by a centimetre. Sensor noise, a
  blinking LED and an exposure step stay under the limit. A move of less than
  4.5 mm at 1 m can pass. The model's own error is larger. Once the last answer
  is `reuse_max_age_s` old, the next call asks again;
- `max_calls_per_hour`: a call over the cap fails with "hourly cap of N
  questions reached";
- backoff: a failed question (a timeout, a non-zero exit, a subscription limit,
  an answer that is not the JSON asked for) is retried after a wait that
  doubles with each consecutive failure, up to 300 s.

A subscription limit shows in the row as
`blad: CliError: subscription limit: ...`, with the time of the next try.

### Latency and accuracy

In the smoke run (two 640x480 images, effort `low`) one question took 4 s with
claude and 9 s with codex. More cameras, larger images and higher effort take
longer. Strategies never wait for an answer; the poses only get older.

The answer has a pixel and a box per camera, but no mask, size or height. The
twin therefore places an object by triangulation when two cameras see it and
on the table top otherwise, with no orientation. Each detection's pixel error
is `pixel_error_px` scaled from the sent image to its frame, which is what lets
two cameras' rays pair. Expect pixel errors of tens of pixels: at 0.5 m with a
focal length of 600 to 900 px that is 1 to 3 cm. The positions are good for
"which object and roughly where", not for grasping a pen.

### Freshness: `ostatnie widziane`

An answer arrives seconds after its frames were captured, more than the 1 s
`fresh_s` allows. The LLM's objects therefore publish `ostatnie widziane`, not
`kamery`, with `age_s` counting from the capture. A track survives between
answers, because `hold_s` is three times the interval plus the latency: 57 s
at a 10 s interval and a 9 s answer. soarm-lab's policy, which waits for
`kamery`, never takes such a pose for a live one. A strategy that wants the
LLM's objects reads `age_s` and decides how old is too old. `w dloni` still
works: the tracker follows an object the gripper closes on.

A repeated answer from `reuse_if_still` is the exception. It returns at once.
The twin stamps it with the capture time of the frames just read. The object
therefore publishes `kamery` for `fresh_s` after each repeat. The frames match
the ones the model answered on, so the object stands where the model saw it.
The pose is only as good as the model's answer. `reuse_max_age_s` limits how
long a wrong answer repeats. Turn `reuse_if_still` off when a strategy must
act only on a pose the model has just confirmed.

### Privacy

The frames leave the machine. They go to OpenAI (codex) or Anthropic (claude)
under the terms of the subscription the CLI is logged in with, as JPEGs of
at most `max_side_px`. Each question's files live in a temporary directory
that is removed afterwards. With `MACHINEKIND_TWIN_LLM_KEEP=<dir>` set, a copy
of the images, the prompt and the CLI's output goes to `<dir>` first.

The tests never ask a model. One real question runs only on request:

```bash
MACHINEKIND_TWIN_LLM=codex uv run pytest -q packages/machinekind_twin_llm/tests/test_llm_real.py
```

## Seeded detector

[`machinekind-twin-seeded`](../packages/machinekind_twin_seeded), installed by
`uv sync`, registers the detector `seeded`. It follows blocks by colour at the
twin's call rate after the LLM detector has said which blob is which block. A
vision model behind the `claude` or `codex` CLI names the blocks every
`seed_interval_s`, 10 s by default. Between its answers the detector grows
each block from the model's pixel by colour and follows the blob. For every
block it sees, it returns the block's mask and size, so the silhouette fit
places the block with its yaw five times a second.

The panel never starts it by itself. Start it in the **Detektory** tab or with
`twin ui --detector seeded`, after filling a `seeded:` section in the config
(`configs/default.yaml` has a commented example). The package README has the
details.

### Threads

| Thread | Rate | Does |
|---|---|---|
| `detektor:seeded`, the runner | `interval_s`, 0.2 s | Takes the model's latest answer, hands the worker a copy of the frames when a question is due, grows every track in the current frames |
| `detektor-seed:<cli>`, the worker | at most every `seed_interval_s` | Asks the LLM detector about the copied frames, one question at a time |

A label is `<colour> <shape>` as the model names a block (`red cube`,
`wooden cuboid`): the shape word gives its size, the colour word its colour.

A track is one block in one camera, and several blocks may share a label. A
fresh seed takes the blob of the label's colour whose box lies within the
seed's pixel error. From then on the track follows the blob nearest its last
centroid, within `track_error_px`. The model's seeds of a label in a camera go
to that label's tracks by distance, closest first, whatever order the model
lists them in. A seed near a followed track only confirms it; a seed that no
track is near starts a new one. One seed and one track of a label always pair.
A track the model leaves out of three answers in a row is dropped.

A blob that several tracks reach holds several blocks: blocks of one colour
that touch, or one on another. Its pixels are shared out, each to the nearest
track's centre, and the twin then fits the blocks' boxes to the masks together
(see [Blocks on blocks](#blocks-on-blocks)). The centres settle on the blocks'
pixels, unless that moves one more than half way to its neighbour: the cut then
follows the blob's outline, which cuts a flat block lying on another along its
length, so the centres stay where they are. While tracks share a blob their
centres move with the blob. A blob with fewer than three quarters of the pixels
its tracks had cannot hold them all: a block left, and its track reached the
neighbour. The track nearest the blob keeps it. A blob, or a share of one,
longer than `max_blob_ratio` times the model's box is not one block and is not
followed: a block the colour of the table, or two blocks of one colour while
the model has named only one of them.

The model answers about frames some seconds old. A new seed that would share a
blob a seen track follows is checked against the frame the model saw: when its
box now holds less than three quarters of the colour it held then, its block
has left since, and the seed is dropped. The block is seeded again from the
next answer.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `cli` | `codex` | `codex` or `claude`, as for the LLM detector |
| `model` | `gpt-6-astra` | the model as the CLI names it; `""` = the CLI's default |
| `effort` | `low` | reasoning effort passed to the CLI |
| `labels` | `""` | the blocks to follow, comma-separated; `""` = every block the model names |
| `labels_sizes` | `""` | size overrides by label or shape word, `name=LxWxH` [m] |
| `labels_hsv` | `""` | colour overrides by label or colour word, OpenCV HSV ranges |
| `cameras` | `""` | cameras to track and to send, comma-separated; `""` = every camera offered |
| `prompt` | `""` | extra instructions appended to the LLM's prompt |
| `interval_s` | 0.2 | shortest time between two colour-tracking calls [s], at least 0.1 |
| `seed_interval_s` | 10 | shortest time between the starts of two LLM questions [s], at least 2 |
| `seed_error_px` | 25 | how far off the model's pixel may be, in the sent image's pixels [px] |
| `track_error_px` | 40 | how far a followed blob may move between two calls, in frame pixels [px], 1 to 500 |
| `lost_after_s` | 3 | a track without a blob for this long is dropped [s] |
| `min_area_px` | 30 | smallest patch of the colour that counts as a blob [px] |
| `max_blob_ratio` | 2 | a blob longer than this times the model's box is not one block; at least 1 |
| `timeout_s` | 120 | the CLI's process group is killed after this [s] |
| `max_side_px` | 1280 | long edge of the images sent [px] |
| `max_calls_per_hour` | 0 | cap on questions per hour; 0 = no cap |
| `min_confidence` | 0.3 | seeds the model is less sure of are dropped |
| `executable` | `""` | the CLI binary; `""` = `cli` on PATH |
| `claude_images` | `stream` | claude only: `stream` or `read` |

The settings the LLM detector also has go to it unchanged. `seed_interval_s`
becomes its `interval_s`, `seed_error_px` its `pixel_error_px`, and `labels`
its `objects`. A line naming the colours and shapes goes before `prompt`.

### The blocks

| Shape | Size `LxWxH` [m] | Rests |
|---|---|---|
| `cube` | `0.029x0.029x0.029` | on a face |
| `cuboid` | `0.060x0.029x0.014` | flat |
| `cylinder` | `0.060x0.029x0.029` | on its side, as its bounding box |

Colours, OpenCV HSV (H 0..180, S and V 0..255): `red`, `yellow`, `green`,
`light blue` (or `blue`) and `wooden` (or `wood`, `natural`, `unpainted`); the
package README lists the ranges. `labels_sizes` (`name=LxWxH`) and
`labels_hsv` (`name=H0-H1,S0-S1,V0-V1`, `|` between ranges) override them,
`;` between entries; a key is a whole label or a shape or colour word.

Bare wood is about the colour of a wooden table. On such a table a wooden
block's blob grows into the table, is longer than `max_blob_ratio` times the
model's box, and is not followed: the label gets no pose rather than a wrong
one. On a table of another colour bare wood is followed like paint.

A block stood on its end does not fit its lying box and its label shows
`ostatnie widziane` until it is laid down. Blocks the model names alike share
one label, and the twin keys them `red cube`, `red cube 2`, and so on. A label
with a trailing number takes its size and colour from the words before it.

### What it returns

For every followed block, per camera, a `Detection` with its share of the
blob as `mask`, its `box`, its centroid as `pixel`, the label's `size_m`,
confidence 1 and `pixel_error_px` 6. The twin pairs each block's detections
across cameras and fits a box of `size_m` to their masks, which gives the
position, the yaw and the fit's IoU as confidence.

`pixel` is the colour blob's centroid: with `size_m` the twin reads it as the
centre at half the object's height (`height_m`'s rule), which the silhouette
centroid approximates; the silhouette path never reads it. The model's own
pixel, the footprint's centre, only anchors a fresh seed.

### Freshness

`fresh_s` is 0.5 s at `interval_s` 0.2. A reader sees the pose aged by the
frame's age at the call (the panel's frame cache refreshes every 0.2 s), the
time since the call (up to `interval_s`) and the fit (about 10 to 40 ms per
label). That comes to about 0.55 s with four labels in the panel, so the
source may read `ostatnie widziane` for the tail of a cycle. A strategy treats
`ostatnie widziane` under about 1 s as the current pose. Headless perception
reads its frames live, so its poses stay under about 0.25 s old.

### Lost blobs and failed questions

A label whose blob is not found is left out of that call: the twin shows the
last pose as `ostatnie widziane` and drops it after `hold_s`, 6 s. The
detector drops a track without a blob for `lost_after_s` and waits for the
next seed. For a seed that never found its blob, the time counts from the
answer's arrival.

A failed question (a timeout, a non-zero exit, a subscription limit) is
logged, and the next one goes out after `seed_interval_s`. While any block is
followed, the detector keeps returning it and raises nothing. With nothing to
follow, the call after a failed answer raises `seeded: <the error>`. The
panel shows it and calls again after a wait that doubles with each failure,
which also paces the questions.

`stop` returns at once. It cancels a running question and wakes the worker,
which ends by itself: within 0.2 s when the CLI obeys SIGTERM, and after the
CLI's 2 s grace and SIGKILL otherwise. Only the runner's own join waits.

Do not run `llm` and `seeded` on the same labels at once. Two detectors that
report one label compete for it, and the newer capture time wins: every LLM
answer would replace the followed pose with a point without yaw.

### Privacy

The frames leave the machine every `seed_interval_s`, as for the LLM detector:
JPEGs of at most `max_side_px` go to OpenAI (codex) or Anthropic (claude)
under the terms of the subscription the CLI is logged in with. The colour
tracking runs on the machine.

### Calibrating the colours on the stand

The default ranges cover the simulation and plain paint. Painted blocks under
the stand's lamp may need their own ranges, or every blob is missed and no
label appears. Take a
still from each camera, find a block's pixel (u, v) and read the HSV values
around it:

```python
import cv2
hsv = cv2.cvtColor(cv2.imread("still.png"), cv2.COLOR_BGR2HSV)
print(hsv[v - 3:v + 4, u - 3:u + 4].reshape(-1, 3))
```

Take H about 8 either side of what you see, and S and V from about two thirds
of the lowest values up to 255. Check the shaded side of the block too. Red
lies near both H = 0 and H = 180 and needs one range on each side.
