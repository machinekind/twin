# machinekind-twin-seeded

An object detector for the digital twin that follows blocks by colour, five times a second, after
a vision model has said which blob is which block. The model is the
[LLM detector](../machinekind_twin_llm), asked every `seed_interval_s` (10 s by default) through
the `claude` or `codex` CLI on your subscription. Between its answers the detector grows each
block from the model's pixel by colour and follows the blob. For every block it sees, it hands the
twin the block's silhouette mask and size, so the twin's box fit gives the position and the yaw at
the twin's call rate.

The twin finds it under the entry point name `seeded`. Its config section is `seeded:`.

The images leave the machine every `seed_interval_s`. They go to Anthropic or OpenAI under the
terms of the subscription the CLI is logged in with, as the LLM detector sends them.

## How it works

Two threads share the work:

| Thread | Rate | Does |
|---|---|---|
| `detektor:seeded`, the twin's runner | `interval_s` (0.2 s) | Takes the model's latest answer, hands the worker a copy of the frames when a question is due, grows every track in the current frames |
| `detektor-seed:<cli>`, the worker | at most every `seed_interval_s` | Asks the LLM detector about the copied frames, one question at a time |

A track is one label in one camera:

1. A fresh seed is the model's pixel for that label in that camera. The track takes the blob of
   the label's colour whose box lies within the seed's pixel error (`seed_error_px`, scaled from
   the sent image to the frame). A blob whose box's long side exceeds `max_blob_ratio` times the
   long side of the model's box is two touching blocks and is not taken.
2. From then on the track follows the blob nearest its last centroid, within `track_error_px`.
3. A track that is being followed outranks a new seed: the model's next answer only confirms it.
   A track the model leaves out of three answers in a row is dropped.
4. A track without a blob for `lost_after_s` is dropped and waits for the next seed. For a seed
   that never found its blob, the time counts from the answer's arrival.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `cli` | `codex` | `codex` or `claude`, as for the LLM detector |
| `model` | `gpt-6-astra` | the model as the CLI names it; `""` = the CLI's default |
| `effort` | `low` | reasoning effort passed to the CLI |
| `labels_sizes` | the four blocks | per label the block's box as it rests on the table, `LxWxH` [m] |
| `labels_hsv` | the four blocks | per label the OpenCV HSV ranges of its colour |
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

The settings from `cli` to `claude_images` that the LLM detector also has are handed to it
unchanged; `seed_interval_s` becomes its `interval_s`, `seed_error_px` its `pixel_error_px`, and the
labels its `objects`. Its own checks run when the detector is built, and their `llm: ...` messages
pass through.

## The label tables

Both tables name the same labels, one entry per label, entries separated by `;`. Labels are
compared as the twin compares them: stripped, lower-cased, inner spaces collapsed.

- `labels_sizes`: `label=LxWxH` in metres, the block's box as it rests on the table, `H` its
  height above the table. A lying cylinder is listed as its bounding box, length x diameter x
  diameter.
- `labels_hsv`: `label=H0-H1,S0-S1,V0-V1` in OpenCV's HSV (H 0..180, S and V 0..255), both ends
  included. Join several ranges of one label with `|`: red wraps around H = 0 and needs two.

The defaults are the four Classic World blocks with so-arm101's placeholder sizes (a 2.5 cm unit,
to be measured) and the colours the simulation paints them:

```
labels_sizes: cube=0.025x0.025x0.025;cuboid5=0.05x0.025x0.025;cuboid85=0.085x0.025x0.025;cylinder=0.05x0.025x0.025
labels_hsv: cube=0-12,120-255,70-255|170-180,120-255,70-255;cuboid5=20-35,100-255,90-255;cuboid85=40-85,80-255,60-255;cylinder=135-165,60-255,60-255
```

The size table lists each block as it rests on the table (`LxWxH`, H above the table); a block
stood on end does not fit and its label shows `ostatnie widziane` until it is laid down or its size
is listed that way.

The labels go to the model as the objects to look for. Labels like `cuboid5` tell it little: say
which is which in `prompt`, e.g. `cube is the red cube; cuboid5 the short yellow block; cuboid85 the
long green block; cylinder the purple cylinder`.

## What `detect` returns

For every followed blob, per camera, a `Detection` with:

- `mask`: the blob's pixels, a bool image of the frame's shape;
- `box`: the blob's first and last pixel column and row;
- `pixel`: the blob's centroid;
- `size_m`: the label's size from `labels_sizes`;
- `confidence` 1.0 and `pixel_error_px` 6.

`pixel` is the colour blob's centroid, which marks the centre at half the object's height as the
twin reads a `size_m` pixel; the twin's box fit uses the mask and never the pixel. The model's own
pixel, the footprint's centre, only anchors a fresh seed.

A label is left out of a call when its blob is not found, and before the first answer arrives.
The twin then shows the last pose as `ostatnie widziane` and drops it after its hold time (6 s at
0.2 s calls).

## Failures

- A question that fails (a timeout, a non-zero exit, a subscription limit, an unusable answer)
  is logged, and the next one goes out after `seed_interval_s`. While any track is followed,
  `detect` keeps returning it and raises nothing.
- With nothing to follow, the call after a failed answer raises `seeded: <the error>`. The twin
  shows it and calls again after a wait that doubles with each failure, which also paces the
  questions.
- No frame at all from the cameras raises `seeded: no fresh frame from cameras ...`.
- `stop` returns at once. It cancels a running question (the CLI's process group gets SIGTERM
  within 0.2 s, and SIGKILL after up to 2 s) and wakes the worker, which then ends by itself.

## Calibrating the colours on the stand

The default ranges match the simulation's colours. Painted blocks under the stand's lamp need
their own ranges, or every blob is missed and the labels never appear. Take a still from each
camera, find a block's pixel (u, v), and read the HSV values around it:

```python
import cv2
hsv = cv2.cvtColor(cv2.imread("still.png"), cv2.COLOR_BGR2HSV)
print(hsv[v - 3:v + 4, u - 3:u + 4].reshape(-1, 3))
```

Take H about 8 either side of what you see, and S and V from about two thirds of the lowest
values up to 255. Check the shaded side of the block too. For red, near H = 0 or 180, give one
range on each side of the wrap.

## Configuration

`configs/local.yaml`:

```yaml
seeded:
  cli: codex
  model: gpt-6-astra
  labels_sizes: cube=0.025x0.025x0.025;cuboid5=0.05x0.025x0.025
  labels_hsv: cube=0-12,120-255,70-255|170-180,120-255,70-255;cuboid5=20-35,100-255,90-255
  prompt: cube is the red cube; cuboid5 the short yellow block
  interval_s: 0.2
  seed_interval_s: 10.0
```

Start it in the panel's **Detektory** tab or with `twin ui --detector seeded`. Do not run `llm` and
`seeded` on the same labels at once: two detectors reporting one label compete for it.

## Dependencies

`machinekind-twin-api` (the contracts), `machinekind-twin-llm` (the questions), numpy and OpenCV;
never the twin.

## Tests

```bash
uv run pytest -q packages/machinekind_twin_seeded/tests
```

They run against the LLM package's fake CLI (`packages/machinekind_twin_llm/tests/fake_cli.py`):
no model, no network.
