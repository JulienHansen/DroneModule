# Getting Started

## Installation

```bash
git clone https://github.com/your-org/DroneStandalone
cd DroneStandalone
pip install -e .
```

Dependencies: `torch >= 2.0`, `pyyaml >= 6.0`.

To preview this documentation locally:

```bash
pip install mkdocs-material
mkdocs serve
```

---

## Basic usage

### 1 — Load a config

```python
from drone_control import load_config

cfg = load_config("configs/crazyflie.yaml")

print(cfg.physics.mass)          # 0.027 kg
print(cfg.physics.max_thrust)    # 0.638 N
```

### 2 — Create the controller

```python
from drone_control import CrazyfliePIDController

ctrl = CrazyfliePIDController.from_drone_config(
    cfg,
    num_envs=4,    # parallel environments
    dt=0.002,      # simulation timestep [s]  (= 500 Hz)
    device="cpu",
)
```

### 3 — Build the root state

The controller expects a `[N, 13]` tensor:

```
root_state = [ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]
```

Quaternion convention: **[w, x, y, z]** (scalar first).

```python
import torch

N = 4
root_state = torch.zeros(N, 13)
root_state[:, 3] = 1.0      # identity quaternion (w = 1)
```

### 4 — Run a control step

```python
target_pos = torch.tensor([[1.0, 0.0, 1.0]]).repeat(N, 1)

thrust, moment = ctrl(
    root_state,
    target_pos=target_pos,
    command_level="position",
)
# thrust : [N, 1]  total thrust [N]
# moment : [N, 3]  body moments [N·m]
```

---

## Command levels

| Level | Required inputs | Description |
|---|---|---|
| `"position"` | `target_pos` | Full cascade: pos → vel → att → rate |
| `"velocity"` | `target_vel` | Enter at velocity loop |
| `"attitude"` | `target_attitude`, `thrust_cmd` | Enter at attitude loop |
| `"body_rate"` | `target_body_rates`, `thrust_cmd` | Inner rate loop only |

All levels accept optional `target_yaw` or `target_yaw_rate`.

---

## Resetting environments

```python
# Reset all environments (e.g. at episode start)
ctrl.reset()

# Reset only environments 0 and 2 (e.g. after collision)
ctrl.reset(env_ids=torch.tensor([0, 2]))
```

---

## Per-environment rate gains

Useful for domain randomisation:

```python
# Same gains for all envs
ctrl.set_rate_gains(rate_kp=torch.tensor([300., 300., 150.]))

# Different gains for envs 0 and 1 only
ctrl.set_rate_gains(
    rate_kp=torch.tensor([250., 250., 120.]),
    env_ids=torch.tensor([0, 1]),
)
```

---

---

## Motor speeds with QuadMixer

After getting `thrust` and `moment` from any controller, convert them to
per-motor speeds with `QuadMixer`:

```python
from drone_control import QuadMixer

mixer = QuadMixer.from_drone_config(cfg)   # requires drone.motor in YAML

omega = mixer(thrust, moment)   # [N, 4]  rad/s
# omega[:, 0] → M0 front-left (CCW)
# omega[:, 1] → M1 front-right (CW)
# omega[:, 2] → M2 back-right (CCW)
# omega[:, 3] → M3 back-left (CW)
```

---

## Lee geometric controller

For singularity-free trajectory tracking use `LeePositionController`:

```python
from drone_control import LeePositionController

lee = LeePositionController.from_drone_config(cfg, num_envs=4)

thrust, moment = lee(
    root_state,
    target_pos=torch.tensor([[0., 0., 1.]]).repeat(4, 1),
)
```

The Lee controller is **stateless** — no `reset()` needed between episodes.

---

## RC rate profiles (Acro mode)

Convert a normalised stick input [-1, 1] to a body-rate setpoint [rad/s]:

```python
from drone_control import betaflight_rate_profile

stick    = torch.zeros(4, 3)
stick[:, 0] = 0.5             # 50 % roll stick

omega_sp = betaflight_rate_profile(stick)   # [4, 3] rad/s

thrust, moment = ctrl(
    root_state,
    target_body_rates=omega_sp,
    thrust_cmd=torch.full((4, 1), 0.3),   # normalised throttle [N]
    command_level="body_rate",
)
```

---

## Provided configurations

| File | Drone | Mass | Max thrust |
|---|---|---|---|
| `configs/crazyflie.yaml` | Crazyflie 2.1 | 27 g | 0.638 N |
| `configs/generic_quad_250mm.yaml` | Generic 250 mm racing quad | 250 g | 12 N |
