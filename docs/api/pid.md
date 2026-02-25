# PID_Vectorized

Discrete-time PID controller, vectorized over N environments and
optionally over multiple axes simultaneously.

```python
from drone_control import PID_Vectorized
```

---

## Constructor

```python
PID_Vectorized(
    num_envs=None,
    device="cpu",
    *,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    kff=0.0,
    tau=0.01,
    limit_up=inf,
    limit_down=-inf,
    integral_limit=None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `num_envs` | `int \| None` | Pre-allocate buffers for N envs. `None` → lazy init on first call |
| `device` | `str` | PyTorch device |
| `kp, ki, kd` | `float \| list \| Tensor` | PID gains — scalars for single-axis, `[axes]` for multi-axis |
| `kff` | `float \| list \| Tensor` | Feed-forward gain |
| `tau` | `float` | Derivative low-pass time constant [s]. `0` → raw backward difference |
| `limit_up, limit_down` | `float` | Asymmetric output saturation bounds |
| `integral_limit` | `float \| list \| Tensor \| None` | Symmetric cap on the integral term before output anti-windup |

### Single-axis mode (`num_envs` given)

Each instance controls one axis across N environments.
State shape: `[N]`. Used by `cascade.py`.

```python
pid = PID_Vectorized(num_envs=4, device="cpu",
                     kp=6.0, ki=3.0, kd=0.0, tau=0.01,
                     limit_up=15.708, limit_down=-15.708)
out = pid(error, Ts=0.002)   # error: [4]  → out: [4]
```

### Multi-axis mode (lazy init)

One instance handles all 3 axes. State shape: `[N, 3]`.
Used by `CrazyfliePIDController`.

```python
pid = PID_Vectorized(kp=[6., 6., 6.], ki=[3., 3., 1.], kd=0.,
                     tau=0.0, device="cpu",
                     integral_limit=[0.349, 0.349, 6.283])
out = pid(error, Ts=0.002)   # error: [N, 3]  → out: [N, 3]
```

---

## `forward` (callable)

```python
output = pid(error, Ts, feedforward=None, measurement_dot=None)
```

| Parameter | Description |
|---|---|
| `error` | Tracking error, shape `[N]` or `[N, axes]` |
| `Ts` | Timestep [s] |
| `feedforward` | Optional signal; added as `kff * feedforward` |
| `measurement_dot` | Time derivative of the **measurement** (not the error). Enables derivative-on-measurement: `d_term = −kd · ẋ` |

### Derivative modes

| Condition | Derivative computation |
|---|---|
| `measurement_dot` provided | `d = −kd · measurement_dot` (no setpoint kick) |
| `tau > 0` | Tustin bilinear low-pass: `d = kd · differentiator` |
| `tau = 0` | Raw backward difference: `d = kd · (e − e_prev) / Ts` |

### Anti-windup

Two-stage:

1. **Integral clamping** (if `integral_limit` set): `ki·∫e` is clamped and
   the integral state is back-calculated to stay consistent.
2. **Output back-calculation**: after output saturation, the integrator is
   corrected by `(u_sat − u_unsat) / ki`.

---

## `reset`

```python
pid.reset()                                # reset all
pid.reset(env_ids=torch.tensor([0, 2]))   # reset subset
```

Zeroes: `integrator`, `differentiator`, `error_d1`, `u`, `u_unsat`.

---

## State attributes

| Attribute | Description |
|---|---|
| `integrator` | Integral accumulator |
| `differentiator` | Tustin-filtered derivative signal |
| `error_d1` | Previous error (for trapezoidal integration and backward diff) |
| `u` | Last saturated output |
| `u_unsat` | Last unsaturated output (before clamping) |
