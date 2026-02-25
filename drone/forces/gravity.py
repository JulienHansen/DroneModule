"""
Gravitational force model.
"""

from __future__ import annotations

import torch

from .base import ForceModel


class Gravity(ForceModel):
    """
    Constant gravitational force along the world -Z axis.

        F = -mass · g · ẑ    (world frame)
        τ = 0

    Parameters
    ----------
    mass : float
        Drone mass [kg].
    g : float
        Gravitational acceleration [m/s²]. Default: 9.81.
    device : str
        PyTorch device.

    Example
    -------
    >>> cfg = load_config("configs/crazyflie.yaml")
    >>> grav = Gravity.from_drone_config(cfg)
    """

    def __init__(
        self,
        mass: float,
        g: float = 9.81,
        device: str = "cpu",
    ) -> None:
        self.mass = float(mass)
        self.g = float(g)
        self.device = torch.device(device)

    @classmethod
    def from_drone_config(cls, config, device: str = "cpu") -> "Gravity":
        """Build from a :class:`~drone.config.loader.DroneConfig`."""
        return cls(mass=config.physics.mass, device=device)

    def to(self, device: str) -> "Gravity":
        self.device = torch.device(device)
        return self

    def compute(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = state.shape[0]
        force = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        force[:, 2] = -self.mass * self.g
        torque = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        return force, torque
