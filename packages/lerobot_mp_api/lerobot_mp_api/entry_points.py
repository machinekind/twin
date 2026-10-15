"""The entry point groups through which the twin finds plugin packages.

A package registers what it brings in its `pyproject.toml`, each entry named like what it
registers (the twin compares names in lower case and refuses two packages registering one name):

    [project.entry-points."lerobot_mp.robots"]
    myarm = "myarm:ROBOT_DIR"

    [project.entry-points."lerobot_mp.backends"]
    myarm = "myarm.backend:create"

    [project.entry-points."lerobot_mp.strategies"]
    pi05 = "vla_strategies.pi05:Pi05"

Entry points live in the installed package's metadata: a package registers nothing until it is
installed (`pip install`), and a change to them needs a reinstall.
"""

__all__ = ["BACKENDS", "ROBOTS", "STRATEGIES"]

#: A robot: the entry point resolves to the path (`str` or `os.PathLike`) of the robot's folder,
#: which holds `robot.yaml` and the model it names. `robot.yaml`'s `name` equals the entry's name.
ROBOTS = "lerobot_mp.robots"

#: A backend: the entry point resolves to a factory `create(ctx: BackendContext)` that returns a
#: `RobotBackend` or any object following `BackendPlugin`. The twin's config section named like the
#: backend reaches it raw, as `ctx.settings`.
BACKENDS = "lerobot_mp.backends"

#: A strategy: the entry point resolves to a class following `Strategy`. The twin builds its
#: `settings` dataclass from the panel's form, which the config section named like the strategy
#: prefills, and creates the strategy as `cls(settings)`, or `cls()` when the class has no `settings`.
STRATEGIES = "lerobot_mp.strategies"
