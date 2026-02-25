"""
Abstract base class for all force/torque models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class ForceModel(ABC):
    """
    Base class for external forces and torques acting on the drone body.

    Every concrete subclass must implement :meth:`compute`.

    Convention
    ----------
    - **Force** is returned in the **world frame** [N].
    - **Torque** is returned in the **body frame** [N·m].
    - Both tensors have shape ``[N, 3]`` where N is the batch size.
    """

    @abstractmethod
    def compute(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute force and torque contributions for the current state.

        Parameters
        ----------
        state : torch.Tensor, shape ``[N, 13]``
            Drone state: ``[pos(3) | quat(4) | lin_vel(3) | ang_vel(3)]``.
            Quaternion convention: ``[w, x, y, z]`` (scalar-first).

        Returns
        -------
        force  : ``[N, 3]``  force in world frame [N]
        torque : ``[N, 3]``  torque in body frame [N·m]
        """

    def to(self, device: str) -> "ForceModel":
        """Move internal tensors to *device* and return ``self``."""
        return self
