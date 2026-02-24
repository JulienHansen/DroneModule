"""
Cascade PID controllers for quadrotor drones.

Architecture
────────────
Position loop (outer)  →  Velocity loop (inner)  →  [roll/pitch ref, thrust]
Angle loop    (outer)  →  Rate loop     (inner)  →  torques

Both controllers are vectorized over N parallel environments and load their
gains from a DroneConfig object (see drone_control.config.loader).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .pid import PID_Vectorized
from ..config.loader import (
    DroneConfig,
    DronePhysicsConfig,
    AttitudeControllerConfig,
    PositionControllerConfig,
    PIDConfig,
)

G = 9.81  # m/s²


def _cross(omega: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Batched cross product omega × vector along last dim."""
    return torch.linalg.cross(omega, vector, dim=-1)


def _make_pid(num_envs: int, device: torch.device, cfg: PIDConfig) -> PID_Vectorized:
    return PID_Vectorized(
        num_envs, device,
        kp=cfg.kp, ki=cfg.ki, kd=cfg.kd,
        limit_up=cfg.limit, limit_down=-cfg.limit,
        tau=cfg.tau,
    )


# ---------------------------------------------------------------------------
# Attitude controller
# ---------------------------------------------------------------------------

class AttController_Vectorized(nn.Module):
    """
    Cascade attitude controller for N quadrotors in parallel.

    Outer loop  (run_angle): angle error [rad]   → body rate setpoints [rad/s]
    Inner loop  (run_rate):  rate error  [rad/s]  → torques [N·m]

    The torque computation uses the full Euler equation:
        τ = J α_ref + ω × (J ω)
    which accounts for gyroscopic effects .
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        config: AttitudeControllerConfig,
    ):
        super().__init__()

        self.num_envs = num_envs
        self.device = device
        self.dt_rate  = 1.0 / config.freq_rate_hz
        self.dt_angle = 1.0 / config.freq_angle_hz

        # Inner (rate) loop
        self.pid_rollrate  = _make_pid(num_envs, device, config.rate.roll)
        self.pid_pitchrate = _make_pid(num_envs, device, config.rate.pitch)
        self.pid_yawrate   = _make_pid(num_envs, device, config.rate.yaw)

        # Outer (angle) loop
        self.pid_roll  = _make_pid(num_envs, device, config.angle.roll)
        self.pid_pitch = _make_pid(num_envs, device, config.angle.pitch)
        self.pid_yaw   = _make_pid(num_envs, device, config.angle.yaw)

    @classmethod
    def from_drone_config(
        cls,
        drone_config: DroneConfig,
        num_envs: int,
        device: torch.device,
    ) -> "AttController_Vectorized":
        """Instantiate directly from a loaded DroneConfig."""
        return cls(num_envs, device, drone_config.attitude)

    def reset(self, env_ids=None):
        """Reset all PID states for selected environments (or all if None)."""
        for pid in (
            self.pid_rollrate, self.pid_pitchrate, self.pid_yawrate,
            self.pid_roll, self.pid_pitch, self.pid_yaw,
        ):
            pid.reset(env_ids)

    def run_rate(
        self,
        ref_omegab: torch.Tensor,   # [N, 3] desired body rates [rad/s]
        meas_omegab: torch.Tensor,  # [N, 3] measured body rates [rad/s]
        J: torch.Tensor,            # [N, 3, 3] inertia tensors [kg·m²]
    ) -> torch.Tensor:              # [N, 3] torques [N·m]
        """Inner rate loop: rate error → body torques via Euler equation."""
        err = ref_omegab - meas_omegab

        alpha_ref = torch.stack([
            self.pid_rollrate.forward(err[:, 0],  self.dt_rate),
            self.pid_pitchrate.forward(err[:, 1], self.dt_rate),
            self.pid_yawrate.forward(err[:, 2],   self.dt_rate),
        ], dim=-1)  # [N, 3]  angular acceleration setpoint [rad/s²]

        # τ = J α + ω × (J ω)  — full Euler equation
        Jw   = torch.matmul(J, meas_omegab.unsqueeze(-1)).squeeze(-1)
        tau  = torch.matmul(J, alpha_ref.unsqueeze(-1)).squeeze(-1) + _cross(meas_omegab, Jw)
        return tau

    def run_angle(
        self,
        ref_rpy: torch.Tensor,   # [N, 3] desired roll-pitch-yaw [rad]
        meas_rpy: torch.Tensor,  # [N, 3] measured roll-pitch-yaw [rad]
    ) -> torch.Tensor:           # [N, 3] body rate setpoints [rad/s]
        """Outer angle loop: angle error → body rate setpoints."""
        err = ref_rpy - meas_rpy

        omega_roll  = self.pid_roll.forward(err[:, 0],  self.dt_angle)
        omega_pitch = self.pid_pitch.forward(err[:, 1], self.dt_angle)

        # Wrap yaw error to [-π, π]
        ey = err[:, 2]
        ey = torch.where(ey >  math.pi, ey - 2.0 * math.pi, ey)
        ey = torch.where(ey < -math.pi, ey + 2.0 * math.pi, ey)
        omega_yaw = self.pid_yaw.forward(ey, self.dt_angle)

        return torch.stack([omega_roll, omega_pitch, omega_yaw], dim=-1)


# ---------------------------------------------------------------------------
# Position controller
# ---------------------------------------------------------------------------

class PosController_Vectorized(nn.Module):
    """
    Cascade position controller for N quadrotors in parallel.

    Outer loop (run_pos): position error [m]    → velocity setpoints [m/s]
    Inner loop (run_vel): velocity error [m/s]  → (roll/pitch ref [rad], thrust [N])

    The velocity→attitude conversion accounts for the current yaw heading so
    that vx/vy commands are always given in the world (Earth) frame.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        config: PositionControllerConfig,
        physics: DronePhysicsConfig,
    ):
        super().__init__()

        self.num_envs = num_envs
        self.device = device
        self.dt_vel = 1.0 / config.freq_vel_hz
        self.dt_pos = 1.0 / config.freq_pos_hz

        # Derived limits
        self.max_thrust  = config.max_thrust_scale * physics.max_thrust  # [N]
        self.max_angle_v = config.max_horizontal_angle_deg * math.pi / 180.0  # [rad]

        # Inner (velocity) loop
        self.pid_vx = _make_pid(num_envs, device, config.velocity.vx)
        self.pid_vy = _make_pid(num_envs, device, config.velocity.vy)
        self.pid_vz = _make_pid(num_envs, device, config.velocity.vz)

        # Outer (position) loop
        self.pid_x = _make_pid(num_envs, device, config.position.x)
        self.pid_y = _make_pid(num_envs, device, config.position.y)
        self.pid_z = _make_pid(num_envs, device, config.position.z)

    @classmethod
    def from_drone_config(
        cls,
        drone_config: DroneConfig,
        num_envs: int,
        device: torch.device,
    ) -> "PosController_Vectorized":
        """Instantiate directly from a loaded DroneConfig."""
        return cls(num_envs, device, drone_config.position, drone_config.physics)

    def reset(self, env_ids=None):
        """Reset all PID states for selected environments (or all if None)."""
        for pid in (self.pid_vx, self.pid_vy, self.pid_vz,
                    self.pid_x, self.pid_y, self.pid_z):
            pid.reset(env_ids)

    def run_pos(
        self,
        ref_pos: torch.Tensor,   # [N, 3]  desired xyz [m]
        meas_pos: torch.Tensor,  # [N, 3]  measured xyz [m]
    ) -> torch.Tensor:           # [N, 3]  velocity setpoints [m/s]
        """Outer position loop: position error → velocity setpoints."""
        err = ref_pos - meas_pos
        return torch.stack([
            self.pid_x.forward(err[:, 0], self.dt_pos),
            self.pid_y.forward(err[:, 1], self.dt_pos),
            self.pid_z.forward(err[:, 2], self.dt_pos),
        ], dim=-1)

    def run_vel(
        self,
        ref_ve: torch.Tensor,    # [N, 3]  desired vx vy vz [m/s] in world frame
        meas_ve: torch.Tensor,   # [N, 3]  measured vx vy vz [m/s] in world frame
        meas_yaw: torch.Tensor,  # [N]     measured yaw angle [rad]
        mass: torch.Tensor,      # [N]     drone mass [kg] (allows heterogeneous envs)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inner velocity loop: velocity error → roll/pitch setpoints and thrust.

        Returns
        -------
        rp_ref_clamped : [N, 2]  roll and pitch setpoints [rad]
        thrust_ref     : [N]     total thrust command [N]
        """
        err = ref_ve - meas_ve

        # --- Horizontal: vx/vy → roll/pitch via yaw rotation ---
        T1 = self.pid_vx.forward(err[:, 0], self.dt_vel)  # [N] acceleration demand x
        T2 = self.pid_vy.forward(err[:, 1], self.dt_vel)  # [N] acceleration demand y

        # Rotate from world to body-horizontal frame using current yaw
        sin_y, cos_y = torch.sin(meas_yaw), torch.cos(meas_yaw)
        R = torch.stack([
            torch.stack([ sin_y, -cos_y], dim=-1),
            torch.stack([ cos_y,  sin_y], dim=-1),
        ], dim=-2)  # [N, 2, 2]

        T12 = torch.stack([T1, T2], dim=-1).unsqueeze(-1)     # [N, 2, 1]
        rp_ref_vec = torch.matmul(R, T12).squeeze(-1) / G     # [N, 2]  roll/pitch [rad]

        # Directional saturation: clamp magnitude, preserve direction
        mag = torch.linalg.norm(rp_ref_vec, dim=-1)           # [N]
        scale = torch.clamp(self.max_angle_v / (mag + 1e-6), max=1.0)
        scale = torch.where(mag > self.max_angle_v, scale, torch.ones_like(scale))
        rp_ref_clamped = rp_ref_vec * scale.unsqueeze(-1)     # [N, 2]

        # Anti-windup for vx/vy: back-project clamped rp to T1/T2
        T12_sat = torch.matmul(R.transpose(-1, -2),
                               (G * rp_ref_clamped).unsqueeze(-1)).squeeze(-1)  # [N, 2]
        for pid, T_sat in [(self.pid_vx, T12_sat[:, 0]),
                           (self.pid_vy, T12_sat[:, 1])]:
            ki_nz    = (pid.ki != 0.0)
            ki_safe  = torch.where(ki_nz, pid.ki, torch.tensor(1.0, device=self.device))
            pid.integrator = torch.where(
                ki_nz,
                pid.integrator + (T_sat - pid.u_unsat) / ki_safe,
                pid.integrator,
            )

        # --- Vertical: vz → thrust ---
        T3 = self.pid_vz.forward(err[:, 2], self.dt_vel)  # [N]  vertical accel demand
        thrust_ref = torch.clamp(
            mass * G + mass * T3,
            min=0.8 * G * mass,
            max=torch.full_like(mass, self.max_thrust),
        )

        # Anti-windup for vz
        T3_sat = (thrust_ref - mass * G) / mass
        ki_nz_vz   = (self.pid_vz.ki != 0.0)
        ki_safe_vz = torch.where(ki_nz_vz, self.pid_vz.ki, torch.tensor(1.0, device=self.device))
        self.pid_vz.integrator = torch.where(
            ki_nz_vz,
            self.pid_vz.integrator + (T3_sat - self.pid_vz.u_unsat) / ki_safe_vz,
            self.pid_vz.integrator,
        )

        return rp_ref_clamped, thrust_ref
