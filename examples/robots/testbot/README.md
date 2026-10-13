# testbot

The digital twin's test robot, packaged the way a robot's own repository packages its arm: a robot
folder (`testbot/robot/`: `robot.yaml`, with a wrist camera on the hand, and the MJCF model) and a
backend plugin (`testbot/backend.py`, an arm in memory). `pyproject.toml` registers both through
entry points:

```toml
[project.entry-points."lerobot_mp.robots"]
testbot = "testbot:ROBOT_DIR"

[project.entry-points."lerobot_mp.backends"]
testbot = "testbot.backend:create"
```

The package depends on `lerobot-mp-api`, the contracts (`packages/lerobot_mp_api` in the twin's
repository), never on the twin. The backend is any object with `connect`, `disconnect`,
`read_joints`, `send_joints` and `is_connected` (`lerobot_mp_api.BackendPlugin` lists the
optional methods); `create` receives a `lerobot_mp_api.BackendContext`: joints, limits, port and
the backend's own config section.

Try it in the twin's environment, which already has `lerobot-mp-api`:

```bash
pip install -e examples/robots/testbot
lerobot-twin ui      # workspace robot: testbot, then "Robot testowy (w pamieci)"
```

The twin's tests load it without installing it (`tests/conftest.py`, fixture `testbot`).
