import torch
import torch.nn as nn


class PID_Vectorized(nn.Module):
    """
    Discrete-time PID controller with derivative filtering and anti-windup,
    vectorized over a batch of N environments.

    Each instance controls a single axis (e.g. roll rate, vx) across N envs.

    Derivative term uses a bilinear-transform (Tustin) first-order low-pass
    with time constant ``tau``. Anti-windup is applied whenever the output
    saturates against ``[limit_down, limit_up]``.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        kp: float,
        ki: float,
        kd: float,
        limit_up: float,
        limit_down: float,
        tau: float,
    ):
        super().__init__()

        self.kp = torch.tensor(kp, dtype=torch.float, device=device)
        self.ki = torch.tensor(ki, dtype=torch.float, device=device)
        self.kd = torch.tensor(kd, dtype=torch.float, device=device)
        self.limit_up   = torch.tensor(limit_up,   dtype=torch.float, device=device)
        self.limit_down = torch.tensor(limit_down, dtype=torch.float, device=device)
        self.tau = torch.tensor(tau, dtype=torch.float, device=device)

        self.register_buffer("integrator",    torch.zeros(num_envs, dtype=torch.float, device=device))
        self.register_buffer("differentiator", torch.zeros(num_envs, dtype=torch.float, device=device))
        self.register_buffer("error_d1",       torch.zeros(num_envs, dtype=torch.float, device=device))
        self.register_buffer("u",              torch.zeros(num_envs, dtype=torch.float, device=device))
        self.register_buffer("u_unsat",        torch.zeros(num_envs, dtype=torch.float, device=device))

        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids=None):
        """Reset integrator, differentiator and stored values for selected envs (or all)."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.integrator[env_ids]    = 0.0
        self.differentiator[env_ids] = 0.0
        self.error_d1[env_ids]      = 0.0
        self.u[env_ids]             = 0.0
        self.u_unsat[env_ids]       = 0.0

    def forward(self, error: torch.Tensor, Ts: float) -> torch.Tensor:
        """
        One step of the PID loop.

        Args:
            error: Current tracking error [num_envs].
            Ts:    Timestep [s].

        Returns:
            Saturated control output [num_envs].
        """
        # Integrator (trapezoidal rule), only when ki != 0
        ki_active = (self.ki != 0.0)
        integrator_update = self.integrator + Ts / 2.0 * (error + self.error_d1)
        self.integrator = torch.where(ki_active, integrator_update, self.integrator)

        # Derivative (bilinear low-pass)
        den = 2.0 * self.tau + Ts
        self.differentiator = (
            (2.0 * self.tau - Ts) / den * self.differentiator
            + 2.0 / den * (error - self.error_d1)
        )

        self.error_d1 = error

        u = self.kp * error + self.ki * self.integrator + self.kd * self.differentiator
        self.u_unsat = u

        u_clamped = torch.clamp(u, self.limit_down, self.limit_up)

        # Anti-windup: correct integrator by saturation error
        sat_error = u_clamped - self.u_unsat
        ki_safe = torch.where(ki_active, self.ki, torch.tensor(1.0, device=self._device))
        self.integrator = torch.where(
            ki_active,
            self.integrator + sat_error / ki_safe,
            self.integrator,
        )

        self.u = u_clamped
        return u_clamped

    def apply_external_saturation_and_antiwindup(self, u_saturated: torch.Tensor):
        """
        Apply anti-windup correction from an *external* saturation stage.
        Used by PosController when roll/pitch magnitude is clamped directionally.

        Args:
            u_saturated: Externally saturated output [num_envs].
        """
        if self.ki.item() != 0.0:
            correction = (u_saturated - self.u_unsat) / self.ki
            self.integrator = self.integrator + correction
        self.u = u_saturated
