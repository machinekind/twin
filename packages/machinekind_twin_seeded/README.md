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

A label is `<colour> <shape>`, as the model names a block: `red cube`, `light blue cuboid`,
`wooden cylinder`. The shape word gives the block's size and the colour word its colour (see
[The blocks](#the-blocks)). A label with no known shape or colour, and no override for it, is not
followed.

A track is one label in one camera:

1. A fresh seed is the model's pixel for that label in that camera. The track takes the blob of
   the label's colour whose box lies within the seed's pixel error (`seed_error_px`, scaled from
   the sent image to the frame). A blob whose box's long side exceeds `max_blob_ratio` times the
   long side of the model's box is not one block and is not taken: two touching blocks of one
   colour, or a block the colour of the table. Without a box from the model, the seed's search
   diameter stands in for it.
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
| `labels` | `""` | the blocks to follow, comma-separated, e.g. `red cube, wooden cuboid`; `""` = every block the model names |
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

The settings from `cli` to `claude_images` that the LLM detector also has are handed to it
unchanged; `seed_interval_s` becomes its `interval_s`, `seed_error_px` its `pixel_error_px`, and
`labels` its `objects`. A line telling the model the colours and shapes and how to name the blocks
goes before `prompt`. Its own checks run when the detector is built, and their `llm: ...` messages
pass through.

## The blocks

The shape word, the last word of a label naming a shape, gives the block's box as it rests [m]:

| Shape | Size `LxWxH` | Rests |
|---|---|---|
| `cube` | `0.029x0.029x0.029` | on a face |
| `cuboid` (or `brick`) | `0.060x0.029x0.014` | flat |
| `cylinder` | `0.060x0.029x0.029` | on its side, as its bounding box |

The words before it name the colour, in OpenCV's HSV (H 0..180, S and V 0..255):

| Colour | Also | HSV ranges |
|---|---|---|
| `red` | | `0-10,100-255,60-255\|170-180,100-255,60-255` |
| `yellow` | | `18-35,110-255,90-255` |
| `green` | | `40-85,70-255,40-255` |
| `light blue` | `blue` | `86-125,50-255,70-255` |
| `wooden` | `wood`, `natural`, `unpainted` | `5-25,25-105,120-255` |

A phrase not in the table is read by its last colour word: `natural wood` is `wooden`, `pale blue`
is `light blue`.

Bare wood is pale and warm, about the colour of a wooden table. Paint never passes the `wooden`
range, but such a table does. A wooden block on it grows into the table's blob, which is longer
than `max_blob_ratio` times the model's box, so the detector does not follow it there and logs
that once. Its label shows no pose rather than a wrong one. On a table of another colour, bare
wood is followed like paint. Set `labels_hsv` for the stand's wood and table if they differ.

Two overrides take precedence, entries separated by `;`. A key is a whole label, which wins, or a
shape or colour word:

- `labels_sizes`: `name=LxWxH` in metres, the block's box as it rests, `H` its height. A lying
  cylinder is listed as its bounding box.
- `labels_hsv`: `name=H0-H1,S0-S1,V0-V1`, both ends included. Join several ranges of one name with
  `|`: red wraps around H = 0 and needs two.

```
labels_sizes: cube=0.03x0.03x0.03;red cube=0.031x0.031x0.031
labels_hsv: wooden=10-20,30-90,150-255
```

A block stood on end does not fit its lying box and its label shows `ostatnie widziane` until it
is laid down. Several blocks the model names alike (two `red cube`s) share one label: the twin
keeps one object per label, and each camera's surest seed.

## What `detect` returns

For every followed blob, per camera, a `Detection` with:

- `mask`: the blob's pixels, a bool image of the frame's shape;
- `box`: the blob's first and last pixel column and row;
- `pixel`: the blob's centroid;
- `size_m`: the block's size;
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

The default ranges cover the simulation's colours and plain paint. Painted blocks under the stand's lamp may need
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
  labels: red cube, light blue cuboid, wooden cylinder   # "" = every block the model names
  labels_hsv: wooden=10-20,30-90,150-255                  # the stand's bare wood, if it needs it
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
