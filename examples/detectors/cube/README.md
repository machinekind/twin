# cube

An object detector packaged the way a detector's own repository packages it: it finds a cube of
one colour in each camera's frame by an HSV threshold. `pyproject.toml` registers it through an
entry point:

```toml
[project.entry-points."machinekind_twin.detectors"]
cube = "cube_detector:ColourCube"
```

The detector only says what it sees where in the image. Per camera it answers with a
`Detection`: the label, the blob's centre in pixels, its box, the silhouette mask and the cube's
size. The twin does the geometry: it fits a cube of `size_m` to the masks of every camera with a
trusted pose, which gives the position, the yaw and the final confidence, and tracks the result.
Strategies read it as `obs.objects["cube"]` (or under `label`). The detector's own confidence is
always 1.0; the twin's fit gives the one strategies see.

The detector declares its settings as a dataclass in `ColourCube.settings`. The twin builds that
dataclass from the panel's form, which the config file's `cube:` section prefills, strictly (an
unknown key, a missing one or a wrong type is an error), and creates the detector as
`ColourCube(settings)`.

```yaml
# configs/local.yaml: prefills the panel's form
cube:
  color: green        # red, green, blue or yellow
  size_m: 0.03        # edge length [m]
  label: cube         # the key strategies read it under
  min_area_px: 20     # smallest blob that counts [px]
  interval_s: 0.33    # seconds between calls
```

The package depends on `machinekind-twin-api`, the contracts (`packages/machinekind_twin_api` in
the twin's repository), never on the twin. `ColourCube` subclasses the `Detector` protocol, which
supplies the members it leaves out (`start`, `stop`):

```python
from machinekind_twin_api import Detector


class ColourCube(Detector[CubeSettings]):
    settings = CubeSettings
    pixel_error_px = 4.0

    def detect(self, request):
        ...
```

`uv sync` in the twin's clone installs it, editable. The twin's tests load it without
`pip install` (`tests/conftest.py`, fixture `cube_example`).
