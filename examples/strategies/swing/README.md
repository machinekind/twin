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

Nothing in the package imports the twin at runtime. `act` returns the package's own `Move`, an
object with the attributes of the twin's `Action`. The twin's `Strategy` protocol is imported
under `TYPE_CHECKING` only, so a type checker checks `settings` and `act`:

```python
if TYPE_CHECKING:
    _: type[Strategy[SwingSettings]] = Swing
```

Try it in the twin's environment:

```bash
pip install -e examples/strategies/swing
lerobot-twin ui      # Strategie > swing, joint: shoulder_pan, Uruchom
```

The twin's tests load it without installing it (`tests/conftest.py`, fixture `swing`).
