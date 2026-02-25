"""
Aerodynamic drag force model.
"""

from __future__ import annotations

from typing import Union

import torch

from .base import ForceModel


class BodyDrag(ForceModel):
    """
    Linear aerodynamic drag proportional to velocity.

        F = -k ⊙ v    (world frame)
        τ = 0

    For isotropic drag pass a scalar ``k``.  For axis-dependent drag
    (e.g., higher resistance face-on) pass ``k`` as ``[kx, ky, kz]``.

    Parameters
    ----------
    coefficients : float or list[float]
        Drag coefficient(s) [N·s/m].  Scalar → same for all axes.
    device : str
        PyTorch device.

    Example
    -------
    >>> drag = BodyDrag(coefficients=0.1)
    >>> drag = BodyDrag(coefficients=[0.1, 0.1, 0.2])   # more drag vertically

    Config-based
    ------------
    Add an optional ``drone.drag_coefficients`` entry to your YAML:

    .. code-block:: yaml

        drone:
          drag_coefficients: [0.1, 0.1, 0.15]   # [N·s/m]

    If the field is absent, drag defaults to zero.
    """

    def __init__(
        self,
        coefficients: Union[float, list[float]],
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if isinstance(coefficients, (int, float)):
            coefficients = [float(coefficients)] * 3
        self._k = torch.tensor(
            [float(c) for c in coefficients],
            dtype=torch.float32,
            device=self.device,
        )  # [3]

    @classmethod
    def from_drone_config(cls, config, device: str = "cpu") -> "BodyDrag":
        """
        Build from a :class:`~drone.config.loader.DroneConfig`.

        Reads ``drone.drag_coefficients`` from the config if present;
        otherwise returns a zero-drag model.
        """
        drag = getattr(config.physics, "drag_coefficients", None)
        if drag is None:
            coeff: Union[float, list[float]] = 0.0
        elif isinstance(drag, list):
            coeff = drag
        else:
            coeff = [float(drag)] * 3
        return cls(coefficients=coeff, device=device)

    def to(self, device: str) -> "BodyDrag":
        self.device = torch.device(device)
        self._k = self._k.to(device)
        return self

    def compute(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vel = state[:, 7:10].to(device=self.device, dtype=torch.float32)
        force = -self._k * vel          # [N, 3]
        torque = torch.zeros_like(force)
        return force, torque
