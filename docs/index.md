# drone-control

**Vectorized PID controllers for quadrotor drones**, designed for parallel
simulation of N environments simultaneously using PyTorch.

```python
from drone_control import load_config, CrazyfliePIDController

cfg  = load_config("configs/crazyflie.yaml")
ctrl = CrazyfliePIDController.from_drone_config(cfg, num_envs=4, dt=0.002)

thrust, moment = ctrl(root_state, target_pos=ref, command_level="position")
```

---

## Features

| Feature | Details |
|---|---|
| **Multi-environment** | Batched over N parallel envs via PyTorch |
| **Multi-rate loops** | Position/velocity @ 100 Hz, attitude/rate @ 500 Hz |
| **4 command levels** | `position`, `velocity`, `attitude`, `body_rate` |
| **Derivative on measurement** | No derivative kick on setpoint steps |
| **Full Euler equation** | Gyroscopic term ω × (J·ω) included |
| **YAML configuration** | Gains, limits and physics from config files |
| **Partial reset** | Reset only a subset of environments |

---

## Quick links

- [Getting Started](getting-started.md) — install and first example
- [Architecture](architecture.md) — how the cascade loops work
- [API Reference](api/controller.md) — all parameters and methods
- [YAML Config Format](config/yaml-format.md) — full config schema
