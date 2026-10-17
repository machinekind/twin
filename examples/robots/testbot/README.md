# testbot

The digital twin's test robot, packaged the way a robot's own repository packages its arm: a robot
folder (`testbot/robot/`: `robot.yaml`, with a wrist camera on the hand, and the MJCF model) and a
backend plugin (`testbot/backend.py`, an arm in memory). `pyproject.toml` registers both through
entry points:

```toml
[project.entry-points."machinekind_twin.robots"]
testbot = "testbot:ROBOT_DIR"

[project.entry-points."machinekind_twin.backends"]
testbot = "testbot.backend:create"
```

The package depends on `machinekind-twin-api`, the contracts (`packages/machinekind_twin_api` in the twin's
repository), never on the twin. The backend is any object with `connect`, `disconnect`,
`read_joints`, `send_joints` and `is_connected` (`machinekind_twin_api.BackendPlugin` lists the
optional methods); `create` receives a `machinekind_twin_api.BackendContext`: joints, limits, port and
the backend's own config section.

Try it in the twin's environment, which already has `machinekind-twin-api`:

```bash
pip install -e examples/robots/testbot
twin ui      # workspace robot: testbot, then "Robot testowy (w pamieci)"
```

The twin's tests load it without installing it (`tests/conftest.py`, fixture `testbot`).
