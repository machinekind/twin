# cube

An example object detector, packaged the way a detector's own repository would package it. It
finds a cube of one colour in each camera frame with an HSV threshold. `pyproject.toml` registers
it through an entry point:

```toml
[project.entry-points."machinekind_twin.detectors"]
cube = "cube_detector:ColourCube"
```

The detector reports what it sees in each image and where. For each camera it returns one
`Detection` with the label, the blob's centre in pixels, its box, the silhouette mask and the
cube's size. The twin fits a cube of `size_m` to the masks from every camera with a trusted pose.
The fit gives the position, the yaw and the confidence that strategies see. The detector's own
confidence is always 1.0. The twin tracks the result, and strategies read it as
`obs.objects["cube"]`, or under the configured `label`.

The detector declares its settings as a dataclass in `ColourCube.settings`. The config file's
`cube:` section prefills the panel's form. The twin builds the dataclass from the form and creates
the detector as `ColourCube(settings)`. An unknown key, a missing key or a wrong type is an
error.

```yaml
# configs/local.yaml: prefills the panel's form
cube:
  color: green        # red, green, blue or yellow
  size_m: 0.03        # edge length [m]
  label: cube         # the key strategies read it under
  min_area_px: 20     # smallest blob that counts [px]
  interval_s: 0.33    # seconds between calls
```

Its only twin dependency is `machinekind-twin-api`, the contract package in
`packages/machinekind_twin_api` of the twin's repository. `ColourCube` subclasses the `Detector`
protocol, which supplies the `start` and `stop` that it leaves out:

```python
from machinekind_twin_api import Detector


class ColourCube(Detector[CubeSettings]):
    settings = CubeSettings
    pixel_error_px = 4.0

    def detect(self, request):
        ...
```

`uv sync` in a clone of the twin installs it in editable mode. The twin's tests load it without
installing it, through the `cube_example` fixture in `tests/conftest.py`.
