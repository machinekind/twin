"""Wybor backendu robota na podstawie konfiguracji."""

from __future__ import annotations

import importlib.util
import logging

from ..twin.config import TwinConfig
from .base import RobotBackend

logger = logging.getLogger(__name__)


def lerobot_available() -> bool:
    return importlib.util.find_spec("lerobot") is not None


def create_backend(cfg: TwinConfig) -> RobotBackend:
    """Create a real-arm backend: `lerobot`, `feetech`, `a1x`, or `auto` (needs a port).

    The simulated arm is `twin.runtime.SceneBackend`: it needs the twin's scene, so `Twin.connect`
    builds it for `sim` and never calls this function for it.
    """
    backend = cfg.robot.backend.lower()

    if backend == "sim":
        raise ValueError(
            "Backend `sim` to fizyka blizniaka - polacz przez Twin.connect('sim') "
            "(panel: Ramie = symulacja); create_backend tworzy tylko prawdziwe ramiona."
        )

    if backend == "lerobot":
        from .lerobot_backend import LeRobotArm

        return LeRobotArm(cfg)

    if backend == "feetech":
        from .feetech import FeetechArm

        return FeetechArm(cfg)

    if backend == "a1x":
        # Galaxea A1X po CAN (pakiet galaxeo, importowany leniwie) - tylko dla stawow A1X.
        from .a1x import A1XArm

        return A1XArm(cfg)

    if backend == "auto":
        # A request that reaches this function wants hardware: never fall back to a simulation.
        if not cfg.robot.port:
            raise ValueError(
                "Backend `auto` bez portu: podaj port prawdziwego ramienia (robot.port) "
                "albo wybierz backend `sim` (symulacja w blizniaku)."
            )
        if lerobot_available():
            from .lerobot_backend import LeRobotArm

            return LeRobotArm(cfg)
        # Bez LeRobota zostaje rozmowa wprost z serwami - nie ciagnie za soba torcha.
        logger.info("Tryb auto: brak biblioteki lerobot, uzywam backendu `feetech`.")
        from .feetech import FeetechArm

        return FeetechArm(cfg)

    raise ValueError(
        f"Nieznany backend robota: {cfg.robot.backend!r} (sim|lerobot|feetech|a1x|auto)"
    )
