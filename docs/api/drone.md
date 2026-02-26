# Drone

Rigid-body digital twin — connects a controller, a list of force models,
and a numerical integrator into a single simulation object.

```python
from drone import load_config, Drone, Gravity, BodyDrag
from drone.controllers import LeePositionController

cfg   = load_config("configs/crazyflie.yaml")
ctrl  = LeePositionController.from_drone_config(cfg, num_envs=4)
drone = Drone.from_drone_config(cfg, controller=ctrl,
                                forces=[Gravity.from_drone_config(cfg),
                                        BodyDrag(coefficients=0.05)])

state = drone.hover_state(num_envs=4)
for _ in range(500):   # 500 × 2 ms = 1 s
    state = drone.step(state, dt=0.002,
                       target_pos=sp, target_vel=None,
                       target_acc=None, target_yaw=None,
                       target_yaw_rate=None)
```

---

## Simulation loop (Zero-Order Hold)

Each call to `step()` follows a **ZOH** scheme:

```
1. thrust, moment  ←  controller(state, **ctrl_kwargs)   [computed once]
2. state_dot       ←  _derivatives(state, thrust, moment)
3. state_next      ←  integrator.step(state, deriv_fn, dt)
```

The control input is **fixed for the entire step**, even when using RK4.
This avoids calling a stateful PID controller multiple times per step.

---

## Constructor

```python
Drone(
    config,
    controller,
    forces=None,
    integrator=None,
    device="cpu",
)
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `DroneConfig` | Loaded YAML config |
| `controller` | callable | Any `(state, **kwargs) → (thrust [N,1], moment [N,3])`. Compatible with `CascadePIDController` and `LeePositionController` |
| `forces` | `list[ForceModel] \| None` | External force/torque models. Default: `[Gravity(mass)]` |
| `integrator` | `Integrator \| None` | Numerical integrator. Default: `EulerIntegrator()` |
| `device` | `str` | PyTorch device |

---

## `from_drone_config`

```python
drone = Drone.from_drone_config(
    config,
    controller,
    forces=None,
    integrator=None,
    device="cpu",
)
```

Convenience factory — same as the constructor.

---

## `hover_state`

```python
state = Drone.hover_state(num_envs=1, device="cpu")
```

Returns a `[num_envs, 13]` state tensor at the origin, level and at rest
(identity quaternion, zero velocity). Useful as simulation starting point.

---

## `step`

```python
state_next = drone.step(state, dt, **ctrl_kwargs)
```

| Parameter | Description |
|---|---|
| `state` | `[N, 13]` current state |
| `dt` | float — timestep [s] |
| `**ctrl_kwargs` | forwarded to the controller (e.g. `target_pos`, `command_level`, …) |

Returns `state_next : [N, 13]`.

The state layout is:
```
[ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]
```

---

## `motor_speeds`

```python
omega = drone.motor_speeds(thrust, moment)  # [N, 4] rad/s
```

Converts controller output to per-motor angular speeds via `QuadMixer`.
Requires `drone.motor` to be present in the YAML config.

---

## `to`

```python
drone = drone.to("cuda")
```

Moves all internal tensors (inertia, mixer, forces) to a different device.
Returns `self`.

---

## Swapping the integrator

```python
from drone import Drone, RK4Integrator

drone = Drone.from_drone_config(cfg, controller=ctrl,
                                integrator=RK4Integrator())
```

No other change is required. The integrator is the only line that differs
between Euler and RK4.

---

## Adding custom forces

Any callable subclassing `ForceModel` can be passed:

```python
from drone.forces import ForceModel
import torch

class ConstantWind(ForceModel):
    def __init__(self, wind_vec, device="cpu"):
        self._f = torch.tensor(wind_vec, device=device, dtype=torch.float32)

    def compute(self, state):
        N = state.shape[0]
        return self._f.expand(N, 3), torch.zeros(N, 3, device=self._f.device)

wind  = ConstantWind([0.5, 0.0, 0.0])    # 0.5 N in world-x
drone = Drone.from_drone_config(cfg, ctrl,
                                forces=[Gravity.from_drone_config(cfg), wind])
```

---

## Internal attributes

| Attribute | Description |
|---|---|
| `config` | `DroneConfig` |
| `controller` | The feedback controller |
| `forces` | List of active `ForceModel` instances |
| `integrator` | Active `Integrator` instance |
| `mixer` | `QuadMixer` (or `None` if no motor config) |
| `_J` | Inertia tensor `[1, 3, 3]` |
| `_J_inv` | Inverse inertia tensor `[1, 3, 3]` |
