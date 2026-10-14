# swing

A control strategy packaged the way a strategy's own repository packages it: it swings one joint
around where it starts, then waits for the joint to come back. `pyproject.toml` registers it
through an entry point:

```toml
[project.entry-points."lerobot_mp.strategies"]
swing = "swing:Swing"
```

The strategy declares its settings as a dataclass in `Swing.settings`. The twin builds that
dataclass from the config file's `swing:` section and the panel's form, strictly (an unknown key,
a missing one or a wrong type is an error), and creates the strategy as `Swing(settings)`.
Settings fields are `str`, `int`, `float` or `bool`; the panel draws a text field, a number or a
switch for each.

```yaml
# configs/local.yaml: prefills the panel's form
swing:
  joint: shoulder_pan
  amplitude: 15.0
```

The package depends on `lerobot-mp-api`, the contracts (`packages/lerobot_mp_api` in the twin's
repository), never on the twin. `act` reads an `Observation` and returns an `Action`; an object
with `Action`'s attributes or a plain mapping of joint targets would do as well. The `Strategy`
protocol lets a type checker check `settings` and `act`:

```python
from lerobot_mp_api import Action, Observation, RunInfo, Strategy

_: type[Strategy[SwingSettings]] = Swing
```

Try it in the twin's environment, which already has `lerobot-mp-api`:

```bash
pip install -e examples/strategies/swing
lerobot-twin ui      # Strategie > swing, joint: shoulder_pan, Uruchom
```

The twin's tests load it without installing it (`tests/conftest.py`, fixture `swing`).
