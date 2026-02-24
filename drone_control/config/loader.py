"""
YAML configuration loader for drone_control.

Config files follow the schema defined by the dataclasses below.
Limits on PID outputs are assumed symmetric (±limit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# PID gain block
# ---------------------------------------------------------------------------

@dataclass
class PIDConfig:
    """Gains and output saturation for a single PID axis."""
    kp: float
    ki: float
    kd: float
    tau: float   # Derivative low-pass time constant [s]
    limit: float  # Symmetric output saturation: output ∈ [-limit, +limit]


# ---------------------------------------------------------------------------
# Attitude controller config
# ---------------------------------------------------------------------------

@dataclass
class AttitudeRateConfig:
    """Rate (inner) loop — error in [rad/s], output in [rad/s²]."""
    roll:  PIDConfig
    pitch: PIDConfig
    yaw:   PIDConfig


@dataclass
class AttitudeAngleConfig:
    """Angle (outer) loop — error in [rad], output in [rad/s]."""
    roll:  PIDConfig
    pitch: PIDConfig
    yaw:   PIDConfig


@dataclass
class AttitudeControllerConfig:
    freq_rate_hz:  float  # Inner loop frequency [Hz]
    freq_angle_hz: float  # Outer loop frequency [Hz]
    rate:  AttitudeRateConfig
    angle: AttitudeAngleConfig


# ---------------------------------------------------------------------------
# Position controller config
# ---------------------------------------------------------------------------

@dataclass
class PositionVelConfig:
    """Velocity (inner) loop — error in [m/s], output in [m/s²] (acceleration)."""
    vx: PIDConfig
    vy: PIDConfig
    vz: PIDConfig


@dataclass
class PositionPosConfig:
    """Position (outer) loop — error in [m], output in [m/s]."""
    x: PIDConfig
    y: PIDConfig
    z: PIDConfig


@dataclass
class PositionControllerConfig:
    freq_vel_hz: float  # Inner loop frequency [Hz]
    freq_pos_hz: float  # Outer loop frequency [Hz]
    max_horizontal_angle_deg: float  # Roll/pitch saturation when chasing velocity [deg]
    max_thrust_scale: float          # Safety factor on drone max thrust (0–1)
    velocity: PositionVelConfig
    position: PositionPosConfig


# ---------------------------------------------------------------------------
# Drone physics
# ---------------------------------------------------------------------------

@dataclass
class InertiaConfig:
    """Principal moments of inertia [kg·m²], assuming diagonal inertia tensor."""
    ixx: float
    iyy: float
    izz: float


@dataclass
class DronePhysicsConfig:
    name: str
    mass: float         # [kg]
    inertia: InertiaConfig
    max_thrust: float   # Total maximum thrust (all motors combined) [N]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class DroneConfig:
    physics:  DronePhysicsConfig
    attitude: AttitudeControllerConfig
    position: PositionControllerConfig


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _pid(d: dict) -> PIDConfig:
    return PIDConfig(
        kp=float(d["kp"]),
        ki=float(d["ki"]),
        kd=float(d["kd"]),
        tau=float(d["tau"]),
        limit=float(d["limit"]),
    )


def load_config(path: str | Path) -> DroneConfig:
    """
    Parse a drone YAML config file and return a DroneConfig dataclass.

    Example
    -------
    >>> cfg = load_config("configs/crazyflie.yaml")
    >>> att_ctrl = AttController_Vectorized.from_drone_config(cfg, num_envs=4, device="cpu")
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    # --- Physics ---
    d = raw["drone"]
    physics = DronePhysicsConfig(
        name=str(d["name"]),
        mass=float(d["mass"]),
        inertia=InertiaConfig(
            ixx=float(d["inertia"]["ixx"]),
            iyy=float(d["inertia"]["iyy"]),
            izz=float(d["inertia"]["izz"]),
        ),
        max_thrust=float(d["max_thrust"]),
    )

    # --- Attitude controller ---
    att = raw["controllers"]["attitude"]
    attitude = AttitudeControllerConfig(
        freq_rate_hz=float(att["freq_rate_hz"]),
        freq_angle_hz=float(att["freq_angle_hz"]),
        rate=AttitudeRateConfig(
            roll=_pid(att["rate"]["roll"]),
            pitch=_pid(att["rate"]["pitch"]),
            yaw=_pid(att["rate"]["yaw"]),
        ),
        angle=AttitudeAngleConfig(
            roll=_pid(att["angle"]["roll"]),
            pitch=_pid(att["angle"]["pitch"]),
            yaw=_pid(att["angle"]["yaw"]),
        ),
    )

    # --- Position controller ---
    pos = raw["controllers"]["position"]
    position = PositionControllerConfig(
        freq_vel_hz=float(pos["freq_vel_hz"]),
        freq_pos_hz=float(pos["freq_pos_hz"]),
        max_horizontal_angle_deg=float(pos["max_horizontal_angle_deg"]),
        max_thrust_scale=float(pos["max_thrust_scale"]),
        velocity=PositionVelConfig(
            vx=_pid(pos["velocity"]["vx"]),
            vy=_pid(pos["velocity"]["vy"]),
            vz=_pid(pos["velocity"]["vz"]),
        ),
        position=PositionPosConfig(
            x=_pid(pos["position"]["x"]),
            y=_pid(pos["position"]["y"]),
            z=_pid(pos["position"]["z"]),
        ),
    )

    return DroneConfig(physics=physics, attitude=attitude, position=position)
