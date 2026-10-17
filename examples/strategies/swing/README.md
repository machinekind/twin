# swing

A control strategy packaged the way a strategy's own repository packages it: it swings one joint
around where it starts, then waits for the joint to come back. `pyproject.toml` registers it
through an entry point:

```toml
[project.entry-points."machinekind_twin.strategies"]
swing = "swing:Swing"
```

The strategy declares its settings as a dataclass in `Swing.settings`. The twin builds that
dataclass from the panel's form, which the config file's `swing:` section prefills, strictly (an
unknown key, a missing one or a wrong type is an error), and creates the strategy as
`Swing(settings)`. Settings fields are `str`, `int`, `float` or `bool`; the panel draws a text
field, a number or a switch for each.

```yaml
# configs/local.yaml: prefills the panel's form
swing:
  joint: shoulder_pan
  amplitude: 15.0
```

The package depends on `machinekind-twin-api`, the contracts (`packages/machinekind_twin_api` in the twin's
repository), never on the twin. `Swing` subclasses the `Strategy` protocol, which supplies the
members it leaves out (`stop`) and lets a type checker check `settings`, `start` and `act`:

```python
from machinekind_twin_api import Strategy


class Swing(Strategy[SwingSettings]):
    hz = 20.0
    settings = SwingSettings
```

`act` reads an `Observation` and returns an `Action`; an object with `Action`'s attributes or a
plain mapping of joint targets would do as well.

Try it in the twin's environment, which already has `machinekind-twin-api`:

```bash
pip install -e examples/strategies/swing
twin ui      # panel: Strategie (Strategies) > swing, joint: shoulder_pan, Uruchom (Run)
```

The twin's tests load it without `pip install` (`tests/conftest.py`, fixture `swing`).
