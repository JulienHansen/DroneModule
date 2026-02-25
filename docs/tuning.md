# Gain Tuning via Pole Placement

```python
from drone_control import tune_from_physics, TuningResult
```

---

## What is pole placement?

A **pole** of a closed-loop system is a root of its characteristic polynomial.
The pole's real part sets the settling speed; the imaginary part sets the
oscillation frequency.  By choosing where to place the poles we pick the
bandwidth ω₀ and damping ratio ζ, and then solve for the PID gains algebraically.

For a PI controller `C(s) = kp + ki/s` applied to the plant `G(s) = 1/s`
(a pure integrator):

```
closed-loop char. poly  =  s² + kp·s + ki
desired poly            =  s² + 2ζω₀·s + ω₀²
```

Matching coefficients:

| Gain | Formula |
|---|---|
| `kp` | `2 · ζ · ω₀` |
| `ki` | `ω₀²` |
| `kd` | `kp · τ_d` (optional derivative time) |

---

## Loop hierarchy

The `CrazyfliePIDController` has four nested loops.  They must be tuned
from the **inside out** and each outer loop must be **at least 5× slower**
than the loop it wraps, otherwise they interact and the system can become
unstable.

```
rate  →  att  →  vel  →  pos
  ω₀      ω₀/5    ω₀/25   ω₀/125
```

If you only provide `bandwidth_rate`, the other three are derived automatically
using this 5× rule.

### Loop summary

| Loop | Input error | Output | Plant `G(s)` |
|---|---|---|---|
| Rate | body rate [rad/s] | angular accel [rad/s²] | `1/s` |
| Attitude | angle [rad] | rate setpoint [rad/s] | `1/s` |
| Velocity x/y | horizontal vel [m/s] | roll/pitch [deg] | `g/s` |
| Velocity z | vertical vel [m/s] | thrust delta [PWM] | `K_z/s` |
| Position | position [m] | vel setpoint [m/s] | `1/s` (pure P) |

`K_z = max_thrust · vel_thrust_scale / (thrust_cmd_max · mass)` — converts
PWM output to m/s² acceleration.

---

## `tune_from_physics`

```python
result = tune_from_physics(
    mass,
    inertia,
    bandwidth_rate,
    bandwidth_att  = None,
    bandwidth_vel  = None,
    bandwidth_pos  = None,
    damping        = 0.7,
    derivative_time = 0.0,
    max_thrust      = 0.638,
    thrust_cmd_max  = 65535.0,
    vel_thrust_scale = 1000.0,
    sim_dt          = None,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `mass` | `float` | Drone mass [kg] |
| `inertia` | `list[float]` | `[Ixx, Iyy, Izz]` [kg·m²] |
| `bandwidth_rate` | `float` | Rate loop bandwidth ω₀ [rad/s].  Typical: 20–60 rad/s |
| `bandwidth_att` | `float \| None` | Attitude loop bandwidth [rad/s].  Default: `bandwidth_rate / 5` |
| `bandwidth_vel` | `float \| None` | Velocity loop bandwidth [rad/s].  Default: `bandwidth_att / 5` |
| `bandwidth_pos` | `float \| None` | Position loop bandwidth [rad/s].  Default: `bandwidth_vel / 5` |
| `damping` | `float` | Damping ratio ζ.  0.7 ≈ critical (≈5 % overshoot), 1.0 = no overshoot |
| `derivative_time` | `float` | τ_d [s]: `kd = kp · τ_d`.  0 → pure PI |
| `max_thrust` | `float` | Total max thrust [N] — used to scale z-velocity gains |
| `thrust_cmd_max` | `float` | Full-scale PWM (default 65535) |
| `vel_thrust_scale` | `float` | PWM scale in the firmware (default 1000) |
| `sim_dt` | `float \| None` | If given, emits warnings when a bandwidth exceeds 10 % of Nyquist |

### Returns

A `TuningResult` dataclass.

---

## `TuningResult`

```python
@dataclass
class TuningResult:
    bandwidth_rate: float
    bandwidth_att:  float
    bandwidth_vel:  float
    bandwidth_pos:  float
    damping:        float

    rate_kp: list[float]   # [roll, pitch, yaw]
    rate_ki: list[float]
    rate_kd: list[float]

    att_kp:  list[float]
    att_ki:  list[float]
    att_kd:  list[float]

    vel_kp:  list[float]   # x/y in deg/(m/s), z in PWM/(m/s)
    vel_ki:  list[float]
    vel_kd:  list[float]

    pos_kp:  list[float]
    pos_ki:  list[float]
    pos_kd:  list[float]

    warnings: list[str]
```

### `.to_params() → dict`

Returns a dict compatible with `CrazyfliePIDController(params=...)`.
Feed-forward terms and saturation limits keep their controller defaults.

### `.summary() → str`

Prints a human-readable table of all gains, bandwidths, and any warnings.

---

## Example

```python
from drone_control import load_config, CrazyfliePIDController
from drone_control import tune_from_physics

cfg = load_config("configs/crazyflie.yaml")

result = tune_from_physics(
    mass=cfg.physics.mass,
    inertia=[cfg.physics.inertia.ixx,
             cfg.physics.inertia.iyy,
             cfg.physics.inertia.izz],
    bandwidth_rate=30.0,   # [rad/s]
    damping=0.7,
    max_thrust=cfg.physics.max_thrust,
    sim_dt=0.002,
)
print(result)
```

Output:

```
── Pole-placement tuning result ─────────────────────────────
  Bandwidths  rate=30.0  att=6.0  vel=1.20  pos=0.240  rad/s
  Damping ζ = 0.7

  Rate loop
    kp = [42.0, 42.0, 42.0]
    ki = [900.0, 900.0, 900.0]
    kd = [0.0, 0.0, 0.0]

  Attitude loop
    kp = [8.4, 8.4, 8.4]
    ki = [36.0, 36.0, 36.0]
    kd = [0.0, 0.0, 0.0]

  Velocity loop  (x/y in deg/(m/s), z in PWM/(m/s))
    kp = [3.929, 3.929, 26.56]
    ki = [3.347, 3.347, 22.63]
    kd = [0.0, 0.0, 0.0]

  Position loop
    kp = [0.24, 0.24, 0.24]
    ki = [0.0, 0.0, 0.0]
    kd = [0.0, 0.0, 0.0]
─────────────────────────────────────────────────────────────
```

Apply only the rate gains while keeping the rest at defaults:

```python
ctrl = CrazyfliePIDController.from_drone_config(cfg, num_envs=1, dt=0.002)
ctrl.set_rate_gains(
    rate_kp=result.to_params()["rate_kp"],
    rate_ki=result.to_params()["rate_ki"],
)
```

Or pass all gains at construction time:

```python
ctrl = CrazyfliePIDController(
    num_envs=1, dt=0.002,
    mass=cfg.physics.mass,
    max_thrust=cfg.physics.max_thrust,
    params=result.to_params(),
)
```

---

## Choosing the bandwidth

| `bandwidth_rate` | Effect |
|---|---|
| < 15 rad/s | Sluggish; drones feel unresponsive |
| 20–40 rad/s | Good starting range for small quadrotors |
| 40–60 rad/s | Aggressive; requires low-noise sensors |
| > 60 rad/s | Usually impossible without hardware-level filtering |

!!! tip "Rule of thumb"
    Start with `bandwidth_rate = 25.0` and `damping = 0.7`.
    Increase bandwidth until oscillation appears, then back off 20 %.

!!! warning "Simulation timestep"
    Pass `sim_dt` to get an automatic Nyquist sanity check.
    The rate bandwidth should stay below `0.1 / sim_dt` rad/s
    (e.g., ≤ 50 rad/s at dt = 2 ms).
