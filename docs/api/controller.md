# CascadePIDController

Generic 4-level cascaded PID controller for multirotor drones.
Default gains match the Crazyflie 2.x firmware; fully configurable via YAML.

```python
from drone import load_config, CascadePIDController

cfg  = load_config("configs/crazyflie.yaml")
ctrl = CascadePIDController.from_drone_config(cfg, num_envs=4, dt=0.002)
```

---

## Constructor

```python
CascadePIDController(
    dt,
    num_envs,
    device="cpu",
    params=None,
    mass=1.0,
    inertia=None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `dt` | `float` | Simulation timestep [s] |
| `num_envs` | `int` | Number of parallel environments |
| `device` | `str` | PyTorch device (`"cpu"`, `"cuda"`) |
| `params` | `dict` | Override any gain or limit (see [YAML format](../config/yaml-format.md)) |
| `mass` | `float` | Drone mass [kg] |
| `inertia` | `list[float]` | `[Ixx, Iyy, Izz]` principal moments [kg·m²] |

---

## Class method: `from_drone_config`

```python
ctrl = CascadePIDController.from_drone_config(
    drone_config,   # DroneConfig from load_config()
    num_envs,
    dt,
    device="cpu",
)
```

Builds the controller from a loaded YAML config. If the config contains a
`controllers.cascade_pid` section, those gains override the defaults.
`thrust_cmd_scale` is derived automatically from `drone.max_thrust`.

---

## `__call__` — main control step

```python
thrust, moment = ctrl(
    root_state,
    *,
    command_level,
    target_pos=None,
    target_vel=None,
    target_attitude=None,
    target_body_rates=None,
    target_yaw=None,
    target_yaw_rate=None,
    thrust_cmd=None,
    body_rates_in_body_frame=False,
)
```

### `root_state` — `[N, 13]`

```
[ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]
```

Quaternion: **[w, x, y, z]** (scalar first). Angular velocity is expected
in the **world frame** by default; set `body_rates_in_body_frame=True` if
it is already in the body frame.

### `command_level`

| Value | Required inputs | Loop entry point |
|---|---|---|
| `"position"` | `target_pos [N, 3]` | Full cascade |
| `"velocity"` | `target_vel [N, 3]` | Velocity loop |
| `"attitude"` | `target_attitude [N, 3]`, `thrust_cmd [N, 1]` | Attitude loop |
| `"body_rate"` | `target_body_rates [N, 3]`, `thrust_cmd [N, 1]` | Rate loop only |

### Yaw inputs (all command levels)

| Input | Shape | Effect |
|---|---|---|
| `target_yaw` | `[N, 1]` | Set absolute yaw setpoint [rad] |
| `target_yaw_rate` | `[N, 1]` | Integrate yaw rate [rad/s] |

If both are `None`, the yaw setpoint holds its previous value.

### Returns

| Output | Shape | Unit |
|---|---|---|
| `thrust` | `[N, 1]` | Newtons [N] |
| `moment` | `[N, 3]` | Newton-metres [N·m] |

---

## `reset`

```python
ctrl.reset()                                  # reset all envs
ctrl.reset(env_ids=torch.tensor([0, 2]))      # reset envs 0 and 2
```

Clears all integrators, derivative states, and internal setpoint buffers.
`_step_count` is reset to 0 so decimation restarts.

---

## `set_physical_params`

```python
ctrl.set_physical_params(
    mass=torch.tensor(0.027),
    inertia_tensor=J,          # [N, 3, 3]
)
```

Update mass and inertia at runtime (e.g. for domain randomisation).

---

## `set_rate_gains`

```python
ctrl.set_rate_gains(
    rate_kp=torch.tensor([250., 250., 120.]),
    rate_ki=torch.tensor([500., 500.,  16.7]),
    rate_kd=torch.tensor([  2.5,  2.5,   0.]),
    env_ids=None,    # None → all envs; tensor → subset
)
```

Update rate-loop gains per environment. The gains start as shared `[3]`
tensors; calling this with `env_ids` promotes them to `[N, 3]`.

---

## Internal state attributes

These are exposed for testing and logging but should not be written directly.

| Attribute | Shape | Description |
|---|---|---|
| `_vel_sp` | `[N, 3]` | Current velocity setpoint [m/s] |
| `_att_sp` | `[N, 3]` | Current attitude setpoint [rad] |
| `_rate_sp` | `[N, 3]` | Current body-rate setpoint [rad/s] |
| `_yaw_sp` | `[N, 1]` | Current yaw setpoint [rad] |
| `_step_count` | `int` | Step counter used for decimation |
| `pos_pid` | `PID_Vectorized` | Position loop PID |
| `vel_pid` | `PID_Vectorized` | Velocity loop PID |
| `att_pid` | `PID_Vectorized` | Attitude loop PID |
| `rate_kp/ki/kd` | `[3]` or `[N, 3]` | Rate loop gains |

---

## Decimation properties

| Property | Description |
|---|---|
| `posvel_decimation` | How many sim steps per pos/vel update |
| `att_decimation` | How many sim steps per att/rate update |
| `posvel_dt` | Effective dt for pos/vel PIDs [s] |
| `att_dt` | Effective dt for att/rate PIDs [s] |
