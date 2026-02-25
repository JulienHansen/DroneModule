# Rate Profiles

RC stick shaping functions that convert a normalised stick input to a
body-rate setpoint.  Matches the firmware conventions of Betaflight,
RaceFlight, Actual-rates, and KISS.

```python
from drone import betaflight_rate_profile

omega_sp = betaflight_rate_profile(stick)   # [N, 3] rad/s
```

---

## Overview

In Acro/Rate mode the pilot controls body rates directly.  A raw stick
signal is a normalised value in **[−1, 1]** where ±1 = full deflection.
Rate profiles shape this signal with a non-linear curve before it becomes
a rate setpoint, allowing:

- A **soft centre** (low sensitivity for precision hover)
- **Expo** / **super-expo** to keep a wide usable range in the mid-stick
- A **hard limit** at full stick

All four functions share the same interface:

```
stick : [N, 3]   normalised  ∈ [−1, 1]   (roll, pitch, yaw)
output: [N, 3]   rad/s
```

Output is always in **rad/s** (converted from the firmware's internal deg/s).

---

## `betaflight_rate_profile`

Betaflight triple-rate model with super-expo centre compression.

```python
omega = betaflight_rate_profile(
    stick,
    rc_rate    = None,        # [3]  default [1.0, 1.0, 1.0]
    super_expo = None,        # [3]  default [0.0, 0.0, 0.0]
    expo       = None,        # [3]  default [0.0, 0.0, 0.0]
    limit      = None,        # [3]  deg/s    default [670, 670, 670]
)
```

| Parameter | Shape | Description |
|---|---|---|
| `rc_rate` | `[3]` | Overall rate multiplier (0–3.0). 1.0 ≈ 200 °/s at full stick with defaults |
| `super_expo` | `[3]` | Centre compression factor (0–0.99). Higher → more centre dead-zone |
| `expo` | `[3]` | Expo factor (0–1.0). Softens centre without affecting full stick |
| `limit` | `[3]` | Hard rate limit [deg/s] applied after shaping |

The Betaflight formula:

```
rate_deg_s = 200 · rc_rate · stick
           · (1 + expo · (stick² − 1))
           / (1 − super_expo · |stick|)
rate_deg_s = clamp(rate_deg_s, −limit, +limit)
```

---

## `raceflight_rate_profile`

RaceFlight two-parameter model (rate + acro+).

```python
omega = raceflight_rate_profile(
    stick,
    rate     = None,   # [3]  default [1.0, 1.0, 1.0]  — max rate [°/s × 100]
    expo     = None,   # [3]  default [0.0, 0.0, 0.0]
)
```

Simpler curve than Betaflight; commonly preferred for its more linear feel.

---

## `actual_rate_profile`

"Actual Rates" model — directly sets centre sensitivity and max rate.

```python
omega = actual_rate_profile(
    stick,
    center_sensitivity = None,  # [3]  default [70.0, 70.0, 70.0]   deg/s per unit stick
    max_rate           = None,  # [3]  default [670.0, 670.0, 670.0] deg/s at full stick
    expo               = None,  # [3]  default [0.0, 0.0, 0.0]
)
```

The most intuitive model because:
- `center_sensitivity` directly sets the feel around centre
- `max_rate` directly sets the angular rate at full stick

---

## `kiss_rate_profile`

KISS firmware rate model (rate + expo).

```python
omega = kiss_rate_profile(
    stick,
    rate = None,   # [3]  default [1.0, 1.0, 1.0]
    expo = None,   # [3]  default [0.0, 0.0, 0.0]
)
```

---

## Common usage

```python
import torch
from drone import betaflight_rate_profile

N = 4
stick = torch.zeros(N, 3)
stick[:, 0] = 0.5    # 50 % roll stick

# Default Betaflight curve
omega_sp = betaflight_rate_profile(stick)    # [N, 3] rad/s

# Custom high-rate setup
omega_sp = betaflight_rate_profile(
    stick,
    rc_rate=torch.tensor([1.8, 1.8, 1.0]),
    super_expo=torch.tensor([0.7, 0.7, 0.6]),
    expo=torch.tensor([0.0, 0.0, 0.0]),
    limit=torch.tensor([1200., 1200., 600.]),   # deg/s
)
```

---

## Unit note

All `limit` parameters are in **deg/s** (firmware convention).
The functions convert their output to **rad/s** before returning.

```python
# Full stick, default params: output well below 200 rad/s (≈ 3490 deg/s)
omega = betaflight_rate_profile(torch.ones(1, 3))
assert omega.abs().max() < 200.0   # rad/s, not deg/s
```

---

## Connecting to a rate controller

Rate profile output feeds directly into the innermost loop of any
controller that accepts `body_rate` commands:

```python
from drone import CrazyfliePIDController, betaflight_rate_profile

stick     = rc_input[..., :3]              # [N, 3]  normalised
omega_sp  = betaflight_rate_profile(stick) # [N, 3]  rad/s
thrust_cmd = rc_input[..., 3:4]            # [N, 1]  normalised throttle → PWM

thrust, moment = ctrl(
    root_state,
    target_body_rates=omega_sp,
    thrust_cmd=thrust_cmd,
    command_level="body_rate",
)
```
