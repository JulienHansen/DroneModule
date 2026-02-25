# LeePositionController

Geometric position + attitude controller based on Lee et al., CDC 2010.
Operates directly on SO(3) — no Euler-angle singularities.

```python
from drone import load_config, LeePositionController

cfg  = load_config("configs/crazyflie.yaml")
ctrl = LeePositionController.from_drone_config(cfg, num_envs=4, device="cpu")

thrust, moment = ctrl(root_state, target_pos, target_vel, target_acc,
                      target_yaw, target_yaw_rate)
```

---

## Constructor

```python
LeePositionController(
    mass,
    inertia,          # [3] or [3, 3]
    position_gain,    # [3]
    velocity_gain,    # [3]
    attitude_gain,    # [3]
    angular_rate_gain,# [3]
    num_envs=1,
    max_acceleration=inf,
    device="cpu",
)
```

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `mass` | `float` | kg | Drone mass |
| `inertia` | `list[float]` | kg·m² | `[Ixx, Iyy, Izz]` (diagonal) or full `[3,3]` |
| `position_gain` | `list[float]` | N/m | `k_pos` — position error → force |
| `velocity_gain` | `list[float]` | N·s/m | `k_vel` — velocity error → force |
| `attitude_gain` | `list[float]` | N·m/rad | `k_R` — SO(3) attitude error → moment |
| `angular_rate_gain` | `list[float]` | N·m·s/rad | `k_Ω` — angular-rate error → moment |
| `num_envs` | `int` | — | Number of parallel environments |
| `max_acceleration` | `float` | m/s² | Cap on the desired acceleration norm |
| `device` | `str` | — | PyTorch device |

---

## Class method: `from_drone_config`

```python
ctrl = LeePositionController.from_drone_config(
    drone_config,    # DroneConfig from load_config()
    num_envs=1,
    device="cpu",
)
```

Reads gains from the `controllers.lee` section of the YAML config.
Requires that section to be present; raises `ValueError` otherwise.

---

## `__call__` — main control step

```python
thrust, moment = ctrl(
    root_state,         # [N, 13]
    target_pos,         # [N, 3]  or broadcastable
    target_vel=None,    # [N, 3]  default zeros
    target_acc=None,    # [N, 3]  default zeros  (feed-forward)
    target_yaw=None,    # [N, 1]  default 0
    target_yaw_rate=None, # [N, 1]  default 0
    *,
    body_rates_in_body_frame=False,
)
```

### `root_state` — `[N, 13]`

```
[ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]
```

Quaternion: **[w, x, y, z]** (scalar first).

### Returns

| Output | Shape | Unit |
|---|---|---|
| `thrust` | `[N, 1]` | Newtons [N] |
| `moment` | `[N, 3]` | Newton-metres [N·m] |

---

## Control law (summary)

**Force** (world frame):

```
F_des = −k_pos·(p − p_des) − k_vel·(v − v_des) + mass·(g·ẑ + a_des)
```

**Desired attitude**: align body-z with `−F_des / ‖F_des‖`, then rotate
around body-z to match `target_yaw`.

**Attitude error** (on SO(3), avoids gimbal lock):

```
e_R = ½ · vee( R_desᵀ·R − Rᵀ·R_des )
```

**Moment** (body frame):

```
M = −k_R·e_R − k_Ω·e_Ω + ω × (J·ω)
```

where `e_Ω = ω − Rᵀ·R_des·ω_des` is the angular-rate error.

---

## `reset`

```python
ctrl.reset(env_ids=None)
```

The Lee controller is **stateless** (no integrators). `reset()` is a no-op
provided for API compatibility with `CrazyfliePIDController`.

---

## Stateless vs CrazyfliePIDController

| Property | `LeePositionController` | `CrazyfliePIDController` |
|---|---|---|
| Integrators | None | 4 × PID integral states |
| Attitude representation | SO(3) quaternion | Euler RPY |
| Singularity-free | Yes | No (gimbal lock at ±90° pitch) |
| Requires reset on episode end | No | Yes |
| Thrust/moment output | Yes | Yes |
| Full cascade | Yes (1 step) | Yes (4 nested loops) |

---

## Example with `QuadMixer`

```python
from drone import load_config, LeePositionController, QuadMixer
import torch

cfg   = load_config("configs/crazyflie.yaml")
ctrl  = LeePositionController.from_drone_config(cfg, num_envs=N)
mixer = QuadMixer.from_drone_config(cfg)

# Simulation loop
thrust, moment = ctrl(root_state, target_pos)
omega = mixer(thrust, moment)   # [N, 4] motor speeds [rad/s]
```
