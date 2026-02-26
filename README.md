---
<div align="center">
  <img src="https://github.com/JulienHansen/DroneModule/blob/main/docs/assets/banner.png"
       alt="Pearl's banner"
       width="1200"
       height="800" />
</div>

---

# DroneModule - Pytorch tools for Drone Simulation

A modular Python library for quadrotor cascade PID control, designed for
simulation and reinforcement learning research.
Controllers are vectorized over **N parallel environments** using PyTorch,
and every parameter (gains, limits, drone physics) is loaded from a **YAML
config file** — no code changes needed when switching drones.

---

## Installation

```bash
pip install -e .
```

Dependencies: `torch >= 2.0`, `pyyaml >= 6.0`.

---

## Package structure

```
drone_control/
├── controllers/
│   ├── pid.py          # PID_Vectorized  — single-axis PID, batched over N envs
│   └── cascade.py      # AttController_Vectorized, PosController_Vectorized
└── config/
    └── loader.py       # load_config() + typed dataclasses

configs/
├── crazyflie.yaml
└── generic_quad_250mm.yaml
```

---

## Quick start

```python
import torch
from drone_control import load_config, AttController_Vectorized, PosController_Vectorized

# 1. Load a drone config
cfg = load_config("configs/crazyflie.yaml")

# 2. Instantiate controllers
N      = 4                        # number of parallel environments
device = torch.device("cuda")

att = AttController_Vectorized.from_drone_config(cfg, num_envs=N, device=device)
pos = PosController_Vectorized.from_drone_config(cfg, num_envs=N, device=device)

# 3. Build the inertia tensor from config (diagonal)
Ixx, Iyy, Izz = cfg.physics.inertia.ixx, cfg.physics.inertia.iyy, cfg.physics.inertia.izz
J = torch.diag(torch.tensor([Ixx, Iyy, Izz], device=device)).unsqueeze(0).expand(N, -1, -1)
mass = torch.full((N,), cfg.physics.mass, device=device)

# 4. Control loop (one step)
ref_pos  = torch.zeros(N, 3, device=device)
meas_pos = torch.zeros(N, 3, device=device)
meas_vel = torch.zeros(N, 3, device=device)
meas_rpy = torch.zeros(N, 3, device=device)
meas_omega = torch.zeros(N, 3, device=device)

vel_ref              = pos.run_pos(ref_pos, meas_pos)           # position → velocity setpoints
rp_ref, thrust_ref   = pos.run_vel(vel_ref, meas_vel,           # velocity → attitude + thrust
                                   meas_rpy[:, 2], mass)

ref_rpy        = torch.cat([rp_ref, torch.zeros(N, 1, device=device)], dim=-1)
omega_ref      = att.run_angle(ref_rpy, meas_rpy)              # angle → rate setpoints
tau_ref        = att.run_rate(omega_ref, meas_omega, J)         # rate  → torques [N·m]
```

---

## Control architecture

The library implements a **two-level cascade** for both attitude and position:

```
Position reference
      │
      ▼
┌─────────────┐   vel_ref [m/s]   ┌─────────────┐   rp_ref [rad]
│  run_pos()  │ ────────────────► │  run_vel()  │ ──────────────►  to attitude loop
│  (outer)    │                   │  (inner)    │   thrust [N]   ──────────────►  to mixer
└─────────────┘                   └─────────────┘

Attitude reference
      │
      ▼
┌─────────────┐  omega_ref [rad/s] ┌─────────────┐
│ run_angle() │ ──────────────────► │  run_rate() │ ──► torques [N·m]
│  (outer)    │                    │  (inner)     │
└─────────────┘                    └─────────────┘
```

### Attitude controller (`AttController_Vectorized`)

| Method | Input | Output |
|---|---|---|
| `run_angle(ref_rpy, meas_rpy)` | angle error [rad] | body-rate setpoints [rad/s] |
| `run_rate(ref_omega, meas_omega, J)` | rate error [rad/s] | torques [N·m] |

The torque computation uses the full **Euler rigid-body equation**:

```
τ = J · α_ref + ω × (J · ω)
```

This accounts for gyroscopic coupling between axes, which simpler
implementations (e.g. Betaflight) omit.

### Position controller (`PosController_Vectorized`)

| Method | Input | Output |
|---|---|---|
| `run_pos(ref_pos, meas_pos)` | position error [m] | velocity setpoints [m/s] |
| `run_vel(ref_ve, meas_ve, yaw, mass)` | velocity error [m/s] | roll/pitch ref [rad] + thrust [N] |

The velocity-to-attitude conversion rotates the horizontal acceleration
demand into the body frame using the current yaw angle, so **velocity
commands are always expressed in the world frame**.
A directional saturation stage clamps the roll/pitch magnitude to
`max_horizontal_angle_deg` while preserving the commanded direction.

---

## Resetting environments

Both controllers expose a `reset(env_ids)` method that clears all PID
integrators and derivative states for the specified environments:

```python
# Reset all environments at episode start
att.reset()
pos.reset()

# Reset only environments 0 and 2 (e.g. after a crash)
done_envs = torch.tensor([0, 2], device=device)
att.reset(done_envs)
pos.reset(done_envs)
```

---

## YAML configuration

### File structure

```yaml
drone:
  name: "Crazyflie 2.1"
  mass: 0.027            # [kg]
  inertia:
    ixx: 1.657e-5        # [kg·m²]
    iyy: 1.657e-5
    izz: 2.900e-5
  max_thrust: 0.638      # [N]  total (all motors)

controllers:
  attitude:
    freq_rate_hz:  100   # inner loop [Hz]
    freq_angle_hz: 100   # outer loop [Hz]
    rate:
      roll:  { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
      pitch: { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
      yaw:   { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
    angle:
      roll:  { kp: 4.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }
      pitch: { kp: 4.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }
      yaw:   { kp: 3.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }

  position:
    freq_vel_hz: 100
    freq_pos_hz: 100
    max_thrust_scale: 0.8           # effective max = scale × drone.max_thrust
    max_horizontal_angle_deg: 30.0  # roll/pitch saturation from velocity loop
    velocity:
      vx: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
      vy: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
      vz: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
    position:
      x: { kp: 5.0, ki: 0.0, kd: 1.0, tau: 0.10, limit: 10.0 }
      y: { kp: 5.0, ki: 0.0, kd: 1.0, tau: 0.10, limit: 10.0 }
      z: { kp: 5.0, ki: 0.0, kd: 3.5, tau: 0.10, limit:  5.0 }
```

### PID parameters

| Field | Unit | Description |
|---|---|---|
| `kp` | — | Proportional gain |
| `ki` | — | Integral gain (set to `0.0` to disable) |
| `kd` | — | Derivative gain |
| `tau` | s | Derivative low-pass time constant (bilinear Tustin filter) |
| `limit` | *see below* | Symmetric output saturation: output ∈ `[-limit, +limit]` |

**Output units by loop:**

| Loop | Error unit | Output unit | Typical `limit` |
|---|---|---|---|
| Rate (inner att.) | rad/s | rad/s² | ~50 (≈ 8 rev/s²) |
| Angle (outer att.) | rad | rad/s | ~15.7 (≈ 2.5 rev/s) |
| Velocity (inner pos.) | m/s | m/s² | 9999 (handled geometrically for xy) |
| Position (outer pos.) | m | m/s | 5–10 |

### Adding a new drone

Copy one of the existing configs and adjust the values — no Python code to modify:

```bash
cp configs/crazyflie.yaml configs/my_drone.yaml
# edit configs/my_drone.yaml
```

```python
cfg = load_config("configs/my_drone.yaml")
att = AttController_Vectorized.from_drone_config(cfg, num_envs=N, device=device)
```

---

## Accessing config values directly

`load_config` returns a plain Python dataclass, so you can read any field
programmatically:

```python
cfg = load_config("configs/crazyflie.yaml")

print(cfg.physics.mass)                   # 0.027
print(cfg.physics.inertia.ixx)            # 1.657e-05
print(cfg.attitude.rate.roll.kp)          # 50.0
print(cfg.position.max_horizontal_angle_deg)  # 30.0
```

---

## Provided configs

| File | Drone | Mass | Max thrust |
|---|---|---|---|
| `configs/crazyflie.yaml` | Crazyflie 2.1 | 27 g | 0.638 N |
| `configs/generic_quad_250mm.yaml` | Generic 5" quad | 250 g | 12 N |
