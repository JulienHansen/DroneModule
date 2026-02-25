# Integrators

Numerical integrators for rigid-body dynamics.

```python
from drone import EulerIntegrator, RK4Integrator
```

---

## Interface

All integrators share the same signature:

```python
state_next = integrator.step(state, derivatives_fn, dt)
```

| Parameter | Description |
|---|---|
| `state` | `[N, 13]` current state |
| `derivatives_fn` | `callable (state [N,13]) → state_dot [N,13]` |
| `dt` | float — timestep [s] |

The quaternion is **automatically renormalised** after every step.

---

## `EulerIntegrator` (default)

First-order explicit Euler:

```
x_{k+1} = x_k + dt · ẋ(x_k)
```

One `derivatives_fn` call per step. Fast and sufficient for `dt ≤ 2 ms`.

```python
from drone import EulerIntegrator
integrator = EulerIntegrator()
```

---

## `RK4Integrator`

Classical 4th-order Runge-Kutta:

```
k1 = ẋ(x_k)
k2 = ẋ(x_k + dt/2 · k1)
k3 = ẋ(x_k + dt/2 · k2)
k4 = ẋ(x_k + dt   · k3)
x_{k+1} = x_k + dt/6 · (k1 + 2k2 + 2k3 + k4)
```

Four `derivatives_fn` calls per step. More accurate at larger timesteps
(`dt > 5 ms`) or for high-fidelity validation.

```python
from drone import RK4Integrator
integrator = RK4Integrator()
```

!!! note "ZOH — control is computed once"
    Regardless of the integrator, the controller is called **once per
    `Drone.step()`** and the resulting `(thrust, moment)` is held constant
    across all sub-steps.  This matches the Zero-Order Hold assumption used
    in real flight controllers and avoids calling a stateful PID integrator
    multiple times per step.

---

## Swapping integrators

The integrator is the only thing that changes — everything else stays
identical:

```python
from drone import Drone, EulerIntegrator, RK4Integrator

drone_euler = Drone.from_drone_config(cfg, ctrl)                          # default
drone_rk4   = Drone.from_drone_config(cfg, ctrl, integrator=RK4Integrator())
```

---

## Writing a custom integrator

Subclass `Integrator` and implement `step`:

```python
from drone.integrators import Integrator, _apply
import torch

class HeunIntegrator(Integrator):
    """Heun's method (explicit trapezoidal, 2nd order)."""

    def step(self, state, derivatives_fn, dt):
        k1      = derivatives_fn(state)
        state_p = _apply(state, k1, dt)          # predictor (Euler step)
        k2      = derivatives_fn(state_p)
        avg     = (k1 + k2) / 2.0               # corrector
        return _apply(state, avg, dt)
```

`_apply(state, state_dot, dt)` applies one Euler increment and
renormalises the quaternion — reuse it in any custom integrator.

---

## `Integrator` (abstract base)

```python
from drone.integrators import Integrator

class MyIntegrator(Integrator):
    def step(self, state, derivatives_fn, dt) -> torch.Tensor:
        ...
```
