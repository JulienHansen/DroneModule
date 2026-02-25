"""
QuadMixer — maps collective thrust + body moments to rotor speeds.

Theory
------
Each motor i produces:
    thrust  F_i = k_thrust · ω_i²     [N]
    drag torque  τ_i = k_drag  · ω_i²  [N·m]  (reaction opposes spin direction)

The total wrench on the body is:

    T  = Σ F_i
    Mx = Σ  y_i · F_i          (roll  — moment arm along y)
    My = Σ -x_i · F_i          (pitch — moment arm along x, right-hand-rule)
    Mz = Σ  s_i · τ_i          (yaw   — s_i = +1 CW motor, −1 CCW motor)

Written as a linear system:
    wrench = B · omega_sq
where omega_sq = [ω_0², …, ω_3²] and B is the [4×4] allocation matrix.

Inverting B gives motor speeds from the desired wrench:
    omega_sq = B⁻¹ · wrench

Frame conventions
-----------------
FLU (Forward-Left-Up) body frame. Motor numbering (looking from above):

  X-config:                  +-config:
    M0(CCW)  M1(CW)           M0(CCW)
       \\      /              /
        center         M3(CW)  center  M1(CW)
       /      \\              \\
    M2(CCW)  M3(CW)           M2(CCW)

  M0: front-left  (+d, +d)   M0: front  (+L, 0)
  M1: front-right (+d, -d)   M1: left   (0, +L)
  M2: back-right  (-d, -d)   M2: back   (-L, 0)
  M3: back-left   (-d, +d)   M3: right  (0, -L)

  d = arm_length / √2        L = arm_length

Sign conventions:
  Mx > 0  →  left side up   (motors 0,3 with y>0 contribute positively)
  My > 0  →  nose up        (motors 2,3 with x<0 push harder)
  Mz > 0  →  yaw CCW        (CW motors 1,3 contribute +Mz as their reaction is CCW)

Usage
-----
>>> from drone_control import load_config, QuadMixer
>>> cfg   = load_config("configs/crazyflie.yaml")
>>> mixer = QuadMixer.from_drone_config(cfg, device="cpu")
>>> omega = mixer(thrust, moment)   # [N,1], [N,3] → [N,4] rad/s
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class QuadMixer:
    """
    Quadrotor control allocation: ``(thrust, moment) → motor speeds``.

    Parameters
    ----------
    arm_length : float
        Distance from the drone center to each motor [m].
    k_thrust : float
        Motor thrust coefficient — ``F = k_thrust · ω²`` [N·s²].
    k_drag : float
        Motor drag coefficient — ``τ = k_drag · ω²`` [N·m·s²].
    layout : str
        Frame configuration: ``'x'`` (default) or ``'+'``.
    speed_min : float
        Minimum motor speed [rad/s].  Commands below this are clamped.
    speed_max : float
        Maximum motor speed [rad/s].  Commands above this are clamped.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        arm_length: float,
        k_thrust:   float,
        k_drag:     float,
        layout:     str   = "x",
        speed_min:  float = 0.0,
        speed_max:  float = float("inf"),
        device:     str   = "cpu",
    ) -> None:
        self.device    = torch.device(device)
        self.k_thrust  = float(k_thrust)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        layout = layout.lower()
        if layout not in ("x", "+"):
            raise ValueError(f"layout must be 'x' or '+', got '{layout}'")

        d = arm_length / math.sqrt(2) if layout == "x" else float(arm_length)
        c = k_drag / k_thrust  # drag-to-thrust ratio

        if layout == "x":
            # Motor positions (FLU): M0=(+d,+d) M1=(+d,-d) M2=(-d,-d) M3=(-d,+d)
            # Spin: M0=CCW M1=CW M2=CCW M3=CW
            #
            #   T  row: all +k_t
            #   Mx row: y_i * k_t  → (+d, -d, -d, +d)
            #   My row: -x_i * k_t → (-d, -d, +d, +d)
            #   Mz row: s_i * k_d  → CW=+k_d, CCW=-k_d → (-c, +c, -c, +c) * k_t
            B = torch.tensor([
                [ 1.0,  1.0,  1.0,  1.0],   # T
                [ d,   -d,   -d,    d  ],    # Mx
                [-d,   -d,    d,    d  ],    # My
                [-c,    c,   -c,    c  ],    # Mz
            ], dtype=torch.float64) * k_thrust
        else:
            # Motor positions (FLU): M0=(+L,0) M1=(0,+L) M2=(-L,0) M3=(0,-L)
            # Spin: M0=CCW M1=CW M2=CCW M3=CW
            B = torch.tensor([
                [ 1.0,  1.0,  1.0,  1.0],   # T
                [ 0.0,   d,   0.0,  -d  ],   # Mx
                [-d,    0.0,   d,   0.0 ],   # My
                [-c,    c,    -c,    c  ],   # Mz
            ], dtype=torch.float64) * k_thrust

        self._B_inv = torch.linalg.inv(B).to(dtype=torch.float32,
                                              device=self.device)  # [4, 4]

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_drone_config(
        cls,
        drone_config,
        device: str = "cpu",
    ) -> "QuadMixer":
        """
        Build from a :class:`~drone_control.config.loader.DroneConfig`.

        Requires ``drone.motor`` to be present in the YAML.

        Example
        -------
        >>> cfg   = load_config("configs/crazyflie.yaml")
        >>> mixer = QuadMixer.from_drone_config(cfg)
        """
        motor = drone_config.physics.motor
        if motor is None:
            raise ValueError(
                "DroneConfig has no motor section. "
                "Add a 'drone.motor' block to your YAML config."
            )
        return cls(
            arm_length=motor.arm_length,
            k_thrust=motor.k_thrust,
            k_drag=motor.k_drag,
            layout=motor.layout,
            speed_min=motor.speed_min,
            speed_max=motor.speed_max,
            device=device,
        )

    # ── Device management ────────────────────────────────────────────────────

    def to(self, device: str) -> "QuadMixer":
        """Move internal tensors to *device* in-place and return ``self``."""
        self._B_inv = self._B_inv.to(device)
        self.device = torch.device(device)
        return self

    # ── Main call ────────────────────────────────────────────────────────────

    def __call__(
        self,
        thrust: torch.Tensor,
        moment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert desired collective thrust and body moments to rotor speeds.

        Parameters
        ----------
        thrust : [N, 1]
            Collective thrust [N].
        moment : [N, 3]
            Body moments [N·m] — (Mx, My, Mz).

        Returns
        -------
        omega : [N, 4]
            Motor angular speeds [rad/s] in motor order
            (M0=front-left/front, M1=front-right/left,
             M2=back-right/back, M3=back-left/right).
        """
        thrust = thrust.to(device=self.device, dtype=torch.float32)
        moment = moment.to(device=self.device, dtype=torch.float32)

        wrench = torch.cat([thrust, moment], dim=-1)   # [N, 4]

        # omega_sq_i = (B⁻¹ · wrench)_i
        omega_sq = wrench @ self._B_inv.t()            # [N, 4]

        # Clamp to valid range before sqrt (negative values → motor at min)
        omega_sq = omega_sq.clamp(
            min=self.speed_min ** 2,
            max=self.speed_max ** 2,
        )
        return torch.sqrt(omega_sq)                    # [N, 4]  rad/s
