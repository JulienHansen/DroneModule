"""
LeePositionController — geometric position controller (Lee et al. 2010).

Reference
---------
T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a
quadrotor UAV on SE(3)", CDC 2010.

Usage
-----
>>> from drone_control import load_config, LeePositionController
>>> cfg  = load_config("configs/crazyflie.yaml")
>>> ctrl = LeePositionController.from_drone_config(cfg, num_envs=4, device="cpu")
>>> thrust, moment = ctrl(root_state, target_pos=ref_pos)

``root_state`` has shape ``[N, 13]`` = ``[pos(3), quat(4), lin_vel(3), ang_vel(3)]``.
Quaternion convention: **[w, x, y, z]** (scalar first).

The controller is **stateless** (no integrators). ``reset()`` is a no-op.
Outputs are ``(thrust [N, 1], moment [N, 3])`` in SI units, matching the
interface of ``CrazyfliePIDController``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..utils.math_utils import (
    euler_xyz_from_quat,
    expand_to,
    matrix_from_quat,
    normalize,
    quat_apply_inverse,
)

G = 9.81  # m/s²

# Default gains for a Crazyflie-class micro-UAV (mass ≈ 27 g).
# Override via YAML ``controllers.lee`` or constructor kwargs.
_DEFAULT_GAINS: dict = {
    "position_gain":     [0.50,  0.50,  0.70],
    "velocity_gain":     [0.20,  0.20,  0.30],
    "attitude_gain":     [0.06,  0.06,  0.030],
    "angular_rate_gain": [0.002, 0.002, 0.001],
    "max_acceleration":  float("inf"),
}


class LeePositionController:
    """
    Geometric position controller following Lee et al. (2010).

    Computes desired collective thrust [N] and body moments [N·m] to track a
    position / velocity / yaw trajectory using SO(3) attitude error.

    The controller is **stateless** (no integrators). ``reset()`` is a no-op.

    Loop structure
    --------------
    position  →  force_des  →  attitude  →  moment
                 (k_pos, k_vel)              (k_att, k_rate)

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    device : str
        PyTorch device string.
    mass : float
        Drone mass [kg].
    inertia : list of float
        Principal moments of inertia ``[Ixx, Iyy, Izz]`` [kg·m²].
    position_gain : list of float, optional
        ``k_pos`` [3] — position error [m] → force correction [N/m].
    velocity_gain : list of float, optional
        ``k_vel`` [3] — velocity error [m/s] → force correction [N·s/m].
    attitude_gain : list of float, optional
        ``k_att`` [3] — SO(3) attitude error → moment [N·m/rad].
    angular_rate_gain : list of float, optional
        ``k_rate`` [3] — body-rate error [rad/s] → moment [N·m·s/rad].
    max_acceleration : float, optional
        Clip the commanded acceleration magnitude to this value [m/s²].
        ``inf`` (default) = no clipping.
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        *,
        mass: float = 1.0,
        inertia: list[float] | None = None,
        position_gain:     list[float] | None = None,
        velocity_gain:     list[float] | None = None,
        attitude_gain:     list[float] | None = None,
        angular_rate_gain: list[float] | None = None,
        max_acceleration:  float = float("inf"),
    ) -> None:
        self.num_envs = num_envs
        self.device   = torch.device(device)

        def _t(v):
            return torch.as_tensor(v, device=self.device, dtype=torch.float32)

        self.mass    = _t(mass)
        self.g_vec   = _t([0.0, 0.0, 1.0])
        self.inertia = _t(inertia if inertia is not None else [1.0, 1.0, 1.0])

        def _g(key, provided):
            return _t(provided if provided is not None else _DEFAULT_GAINS[key])

        self.k_pos  = _g("position_gain",     position_gain)
        self.k_vel  = _g("velocity_gain",     velocity_gain)
        self.k_att  = _g("attitude_gain",     attitude_gain)
        self.k_rate = _g("angular_rate_gain", angular_rate_gain)
        self.max_acc = _t(max_acceleration)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_drone_config(
        cls,
        drone_config,       # DroneConfig (avoids circular import)
        num_envs: int,
        device: str = "cpu",
    ) -> "LeePositionController":
        """
        Build from a :class:`~drone_control.config.loader.DroneConfig`.

        If the config contains a ``lee`` section (YAML key
        ``controllers.lee``), those gains override the defaults.

        Example
        -------
        >>> cfg  = load_config("configs/crazyflie.yaml")
        >>> ctrl = LeePositionController.from_drone_config(cfg, num_envs=4)
        """
        phys = drone_config.physics
        lee  = getattr(drone_config, "lee", None)

        kwargs: dict = dict(
            num_envs=num_envs,
            device=device,
            mass=phys.mass,
            inertia=[phys.inertia.ixx, phys.inertia.iyy, phys.inertia.izz],
        )
        if lee is not None:
            kwargs.update(
                position_gain=lee.position_gain,
                velocity_gain=lee.velocity_gain,
                attitude_gain=lee.attitude_gain,
                angular_rate_gain=lee.angular_rate_gain,
                max_acceleration=lee.max_acceleration,
            )
        return cls(**kwargs)

    # ── Device management ────────────────────────────────────────────────────

    def to(self, device: str) -> "LeePositionController":
        """Move all tensors to *device* in-place and return ``self``."""
        dev = torch.device(device)
        for attr in ("mass", "g_vec", "inertia", "k_pos", "k_vel",
                     "k_att", "k_rate", "max_acc"):
            setattr(self, attr, getattr(self, attr).to(dev))
        self.device = dev
        return self

    # ── Reset (no-op — controller is stateless) ──────────────────────────────

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """No-op: the Lee controller carries no integrator state."""

    # ── Main call ────────────────────────────────────────────────────────────

    def __call__(
        self,
        root_state: torch.Tensor,
        target_pos:      Optional[torch.Tensor] = None,
        target_vel:      Optional[torch.Tensor] = None,
        target_acc:      Optional[torch.Tensor] = None,
        target_yaw:      Optional[torch.Tensor] = None,
        target_yaw_rate: Optional[torch.Tensor] = None,
        *,
        body_rates_in_body_frame: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one control step.

        Parameters
        ----------
        root_state : [N, 13]
            Current state: pos(3) | quat(4) | lin_vel(3) | ang_vel(3).
        target_pos : [N, 3], optional
            Desired position [m].  Defaults to current position (hover).
        target_vel : [N, 3], optional
            Desired velocity feed-forward [m/s].  Defaults to zero.
        target_acc : [N, 3], optional
            Desired acceleration feed-forward [m/s²].  Defaults to zero.
        target_yaw : [N] or [N, 1], optional
            Desired yaw angle [rad].  Defaults to current yaw.
        target_yaw_rate : [N] or [N, 1], optional
            Desired yaw rate [rad/s] — used as the z-component of ω_des.
        body_rates_in_body_frame : bool
            If True, ``ang_vel`` in ``root_state`` is already in the body
            frame.  If False (default), it is in the world frame and will
            be rotated by the inverse quaternion.

        Returns
        -------
        thrust : [N, 1]
            Collective thrust [N].
        moment : [N, 3]
            Body moments [N·m].
        """
        if root_state.dim() == 1:
            root_state = root_state.unsqueeze(0)

        root_state = root_state.to(device=self.device, dtype=torch.float32)
        pos, quat, lin_vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)

        if not body_rates_in_body_frame:
            ang_vel = quat_apply_inverse(quat, ang_vel)

        # ── Reference defaults ────────────────────────────────────────────────
        if target_pos is None:
            target_pos = pos.clone()
        if target_vel is None:
            target_vel = torch.zeros_like(lin_vel)
        if target_acc is None:
            target_acc = torch.zeros_like(lin_vel)

        f32 = pos.dtype
        target_pos = expand_to(target_pos.to(device=self.device, dtype=f32), pos)
        target_vel = expand_to(target_vel.to(device=self.device, dtype=f32), lin_vel)
        target_acc = expand_to(target_acc.to(device=self.device, dtype=f32), lin_vel)

        # ── Yaw ───────────────────────────────────────────────────────────────
        _, _, yaw_actual = euler_xyz_from_quat(quat)          # [N]
        yaw_actual = yaw_actual.unsqueeze(-1)                  # [N, 1]

        if target_yaw is None:
            yaw_des = yaw_actual
        else:
            yaw_des = target_yaw.to(device=self.device, dtype=f32)
            if yaw_des.dim() < 2:
                yaw_des = yaw_des.unsqueeze(-1)
            yaw_des = expand_to(yaw_des, yaw_actual)

        if target_yaw_rate is None:
            yaw_rate_des = torch.zeros_like(yaw_des)
        else:
            yaw_rate_des = target_yaw_rate.to(device=self.device, dtype=f32)
            if yaw_rate_des.dim() < 2:
                yaw_rate_des = yaw_rate_des.unsqueeze(-1)
            yaw_rate_des = expand_to(yaw_rate_des, yaw_des)

        # ── Position controller → desired force ───────────────────────────────
        # F_des = k_pos*(pos - pos_des) + k_vel*(vel - vel_des) - m*g*ẑ - m*a_ff
        # Note sign convention: errors are (actual - desired).
        # b3_des = -normalize(F_des) points along the thrust axis.
        pos_err = pos - target_pos
        vel_err = lin_vel - target_vel

        force_des = (
              self.k_pos * pos_err
            + self.k_vel * vel_err
            - self.mass * G * self.g_vec
            - self.mass * target_acc
        )

        if torch.isfinite(self.max_acc):
            acc_cmd = force_des / self.mass
            norm    = torch.linalg.norm(acc_cmd, dim=-1, keepdim=True).clamp_min(1e-9)
            acc_cmd = acc_cmd * (torch.minimum(norm, self.max_acc) / norm)
            force_des = acc_cmd * self.mass

        # ── Desired rotation matrix from force + yaw ──────────────────────────
        R = matrix_from_quat(quat)                             # [N, 3, 3]

        b3_des = -normalize(force_des)                         # [N, 3]
        b1_c   = torch.cat([
            torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)
        ], dim=-1)                                             # candidate b1
        b2_des = normalize(torch.linalg.cross(b3_des, b1_c, dim=-1))
        b1_des = torch.linalg.cross(b2_des, b3_des, dim=-1)
        R_des  = torch.stack((b1_des, b2_des, b3_des), dim=-1)  # [N, 3, 3]

        # ── SO(3) attitude error (vee map of skew-symmetric part) ─────────────
        att_err_mat = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        e_R = torch.stack([
            att_err_mat[..., 2, 1],
            att_err_mat[..., 0, 2],
            att_err_mat[..., 1, 0],
        ], dim=-1)                                             # [N, 3]

        # ── Body-rate error ───────────────────────────────────────────────────
        omega_des = torch.zeros_like(ang_vel)
        omega_des[..., 2] = yaw_rate_des.squeeze(-1)
        e_omega = ang_vel - omega_des

        # ── Moment  τ = -k_att·e_R - k_rate·e_Ω + Ω × (J·Ω) ─────────────────
        coriolis = torch.linalg.cross(ang_vel, self.inertia * ang_vel, dim=-1)
        moment   = -self.k_att * e_R - self.k_rate * e_omega + coriolis  # [N, 3]

        # ── Thrust  T = -F_des · b3_actual ────────────────────────────────────
        thrust = -(force_des * R[..., 2]).sum(dim=-1, keepdim=True)        # [N, 1]

        return thrust, moment
