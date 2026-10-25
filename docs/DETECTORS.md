# Detector guide

How the twin runs detectors and tracks the objects they report, and the LLM
detector. [The workstation guide](TWIN.md) covers the station itself.

## Detectors and objects

A detector is the fourth plugin kind, next to robots, backends and strategies.
It is a class in an installed package, registered in the
`machinekind_twin.detectors` entry point group. It reads camera frames and
reports what it sees **in the image** and where. The twin turns these image
positions into poses and tracks them. Strategies read the result as
`Observation.objects[label]`, an `ObjectPose` in the robot's base frame with
`source` and `age_s`. The contract lives in
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
whitespace and lower-casing. The twin drops a detection that names an unknown
camera, has a pixel outside the frame or has a mask of the wrong shape. The
Detectors tab shows how many it dropped.

`start(info)` runs once before the first call and may raise to refuse the
start. `stop(reason)` runs once after a successful start. It may run while
`detect` is still running on the detector thread, and the twin discards the
late answer. When `detect` raises, the panel shows the error and the twin
retries after a wait. The wait doubles with each consecutive failure, up to
300 s.

### Where an object is

For each label, the twin keeps the most confident detection from each camera.
It places the object in one of three ways, and the Detectors tab shows the
Polish name of the way it used:

| Path | When | Result |
|---|---|---|
| silhouette (`sylwetka`) | Every camera gave a mask of a full-size frame, and `size_m` is a cube | The cube silhouette fit across all masks. It gives the position, the yaw and the fit's IoU as confidence |
| triangulation (`triangulacja`) | Two or more cameras whose rays pass within the detectors' pixel error, scaled by depth | The point nearest to all rays. It must lie in front of every camera and between 3 cm below and 60 cm above the table |
| one ray (`promien`) | One camera, or rays that disagree | The most confident detection's ray meets the plane at `height_m`. Without `height_m` the plane is at half of `size_m`, and without both it is the table top |

The geometry uses the calibrated camera poses with lens distortion, also in
simulation. Simulated perception therefore tests the calibration. A placed
object's `t` is the capture time of the oldest frame used to place it.

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

Each label gets its own copy of the twin's cube tracker,
`perception.ObjectTracker`. The tracker also follows an object held in the
gripper. Each measurement is fed to it once, and only when its confidence is
above 0.4. A track is dropped after `hold_s` without a measurement. `hold_s` is
the larger of 6 s and three times the sum of the detector's interval and
latency, so a slow detector's object survives between answers.

`age_s` is the time since the capture of the frames the pose comes from. It is
0 while the object follows the jaws. `source` says what the pose is based on,
and it takes these Polish values:

- `kamery` (cameras) means the cameras saw the object recently. The value holds
  while the detector's latest answer contains the label and
  `age_s <= fresh_s`. `fresh_s` is twice the detector's interval, kept between
  0.5 and 1.0 s. The cube detector at 0.33 s keeps `kamery` for 0.66 s after
  capture.
- `ostatnie widziane` (last seen) means the pose is the last known one. The
  source switches to it when `age_s` exceeds `fresh_s` or when the detector's
  latest answer lacks the label. A measurement that is already older than
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
of them report the same label, the measurement with the newer capture time
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
  rotation is the identity. Only the cube silhouette fit gives yaw.
- Each label holds one object. Two red pens need two labels.

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
