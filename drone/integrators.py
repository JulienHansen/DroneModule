"""
Numerical integrators for rigid-body dynamics.

All integrators share the same interface:

    state_next = integrator.step(state, derivatives_fn, dt)

where ``derivatives_fn(state) -> state_dot`` computes the full state
derivative ``[N, 13]`` at a given state.

Adding a new integrator
-----------------------
Subclass :class:`Integrator` and implement :meth:`step`.  The helper
:func:`_apply` handles quaternion renormalisation automatically.

    class MyIntegrator(Integrator):
        def step(self, state, derivatives_fn, dt):
            state_dot = derivatives_fn(state)
            return _apply(state, state_dot, dt)
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(state: torch.Tensor, state_dot: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Apply one Euler-style increment and renormalise the quaternion.

    Parameters
    ----------
    state     : ``[N, 13]``
    state_dot : ``[N, 13]``  (same layout as state)
    dt        : float

    Returns
    -------
    ``[N, 13]``  new state with normalised quaternion.
    """
    new_state = state + dt * state_dot
    q = new_state[:, 3:7]
    new_state[:, 3:7] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return new_state


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Integrator(ABC):
    """
    Abstract numerical integrator for rigid-body state ``[N, 13]``.

    The state layout is:
        ``[ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]``

    ``derivatives_fn(state) -> state_dot`` must return the time derivative
    of every element of the state (same shape ``[N, 13]``).
    """

    @abstractmethod
    def step(
        self,
        state: torch.Tensor,
        derivatives_fn,
        dt: float,
    ) -> torch.Tensor:
        """
        Advance *state* by one timestep *dt*.

        Parameters
        ----------
        state          : ``[N, 13]``
        derivatives_fn : callable ``(state [N,13]) -> state_dot [N,13]``
        dt             : float  timestep [s]

        Returns
        -------
        state_next : ``[N, 13]``
        """


# ---------------------------------------------------------------------------
# Euler (explicit / forward Euler)
# ---------------------------------------------------------------------------

class EulerIntegrator(Integrator):
    """
    First-order explicit Euler integration.

        x_{k+1} = x_k + dt · ẋ(x_k)

    Fast and sufficient for small timesteps (dt ≤ 2 ms).
    One ``derivatives_fn`` call per step.
    """

    def step(
        self,
        state: torch.Tensor,
        derivatives_fn,
        dt: float,
    ) -> torch.Tensor:
        state_dot = derivatives_fn(state)
        return _apply(state, state_dot, dt)


# ---------------------------------------------------------------------------
# RK4 placeholder  (ready to fill in when needed)
# ---------------------------------------------------------------------------

class RK4Integrator(Integrator):
    """
    Classical 4th-order Runge-Kutta integration.

        k1 = ẋ(x_k)
        k2 = ẋ(x_k + dt/2 · k1)
        k3 = ẋ(x_k + dt/2 · k2)
        k4 = ẋ(x_k + dt   · k3)
        x_{k+1} = x_k + dt/6 · (k1 + 2k2 + 2k3 + k4)

    4× more accurate than Euler at the same dt, at 4× the cost.
    Useful for larger timesteps or high-fidelity validation.

    Note: the control input is held constant across all 4 sub-steps
    (Zero-Order Hold), so the controller is called only once per step
    regardless of the integrator chosen.
    """

    def step(
        self,
        state: torch.Tensor,
        derivatives_fn,
        dt: float,
    ) -> torch.Tensor:
        k1 = derivatives_fn(state)
        k2 = derivatives_fn(_apply(state, k1, dt / 2.0))
        k3 = derivatives_fn(_apply(state, k2, dt / 2.0))
        k4 = derivatives_fn(_apply(state, k3, dt))

        state_dot = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return _apply(state, state_dot, dt)
