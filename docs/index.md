# drone-control

**Vectorized PID controllers for quadrotor drones**, designed for parallel
simulation of N environments simultaneously using PyTorch.

```python
from drone import load_config, CascadePIDController

cfg  = load_config("configs/crazyflie.yaml")
ctrl = CascadePIDController.from_drone_config(cfg, num_envs=4, dt=0.002)

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
| **Geometric controller** | Lee SO(3) position + attitude controller |
| **Control allocation** | Physics-based `QuadMixer` (X and + layouts) |
| **RC rate profiles** | Betaflight, RaceFlight, Actual, KISS curves |
| **Pole-placement tuning** | `tune_from_physics` derives gains from bandwidth |
| **Digital twin** | `Drone` — rigid-body integrator with modular forces |
| **Modular forces** | `Gravity`, `BodyDrag`, custom `ForceModel` subclasses |
| **Modular integrators** | `EulerIntegrator` (default), `RK4Integrator` |
| **YAML configuration** | Gains, limits and physics from config files |
| **Partial reset** | Reset only a subset of environments |

---

## Quick links

- [Getting Started](getting-started.md) — install and first example
- [Architecture](architecture.md) — how the cascade loops work
- [Firmware Architecture](firmware-architecture.md) — sensor-to-motor pipeline explained
- [API: CascadePIDController](api/controller.md)
- [API: LeePositionController](api/lee_controller.md)
- [API: QuadMixer](api/mixer.md)
- [API: Rate Profiles](api/rate_profiles.md)
- [API: PID_Vectorized](api/pid.md)
- [API: Configuration Loader](api/loader.md)
- [YAML Config Format](config/yaml-format.md) — full config schema
- [Gain Tuning](tuning.md) — pole-placement guide
- [Examples: IsaacLab Integration](examples/isaaclab.md) — all 5 command levels
