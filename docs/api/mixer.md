# QuadMixer

Maps a collective thrust + body moments wrench to per-motor angular speeds.

```python
from drone import load_config, QuadMixer

cfg   = load_config("configs/crazyflie.yaml")
mixer = QuadMixer.from_drone_config(cfg, device="cpu")

omega = mixer(thrust, moment)   # [N,1], [N,3] → [N,4] rad/s
```

---

## Theory

Each motor `i` produces:

```
F_i = k_thrust · ωi²     [N]          (thrust)
τ_i = k_drag   · ωi²     [N·m]        (reaction torque)
```

The total wrench on the body is assembled into a linear system:

```
wrench = B · ωsq
```

where `ωsq = [ω0², ω1², ω2², ω3²]` and **B** is the 4×4 allocation matrix.
Inverting **B** gives motor speeds from the desired wrench:

```
ωsq = B⁻¹ · wrench
ωi  = √( clamp(ωsqi, ωmin², ωmax²) )
```

---

## Frame convention

**FLU** body frame (Forward-Left-Up). Motor numbering (viewed from above):

```
X-config:
   M0(CCW)  M1(CW)
      \      /
       center
      /      \
   M2(CCW)  M3(CW)

  M0: front-left   (+d, +d)
  M1: front-right  (+d, −d)
  M2: back-right   (−d, −d)
  M3: back-left    (−d, +d)
  d = arm_length / √2
```

Sign conventions:

- `Mx > 0` → left side rises (M0, M3 with y > 0 push harder)
- `My > 0` → nose rises (M2, M3 with x < 0 push harder)
- `Mz > 0` → yaw CCW

---

## Constructor

```python
QuadMixer(
    arm_length,
    k_thrust,
    k_drag,
    layout="x",
    speed_min=0.0,
    speed_max=inf,
    device="cpu",
)
```

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `arm_length` | `float` | m | Center-to-motor distance |
| `k_thrust` | `float` | N·s² | Thrust coefficient: `F = k_thrust · ω²` |
| `k_drag` | `float` | N·m·s² | Drag coefficient: `τ = k_drag · ω²` |
| `layout` | `str` | — | `"x"` (default) or `"+"` |
| `speed_min` | `float` | rad/s | Minimum motor speed (clamped) |
| `speed_max` | `float` | rad/s | Maximum motor speed (clamped) |
| `device` | `str` | — | PyTorch device |

---

## Class method: `from_drone_config`

```python
mixer = QuadMixer.from_drone_config(
    drone_config,   # DroneConfig from load_config()
    device="cpu",
)
```

Reads all parameters from the `drone.motor` section of the YAML.
Raises `ValueError` if that section is absent.

---

## `__call__`

```python
omega = mixer(thrust, moment)
```

| Argument | Shape | Unit |
|---|---|---|
| `thrust` | `[N, 1]` | N |
| `moment` | `[N, 3]` | N·m (Mx, My, Mz) |

| Return | Shape | Unit |
|---|---|---|
| `omega` | `[N, 4]` | rad/s |

Motor order: M0, M1, M2, M3 (see frame convention above).

---

## `to`

```python
mixer = mixer.to("cuda")
```

Moves internal tensors to a different device in-place. Returns `self`.

---

## YAML config section

```yaml
drone:
  motor:
    arm_length: 0.046      # m
    k_thrust:   1.285e-8   # N·s²
    k_drag:     7.645e-11  # N·m·s²
    layout:     x          # 'x' or '+'
    speed_min:  0.0        # rad/s
    speed_max:  2618.0     # rad/s  (~25 000 RPM)
```

See [YAML Config Format](../config/yaml-format.md) for the full schema.

---

## Full pipeline example

```python
from drone import (
    load_config, CrazyfliePIDController, QuadMixer
)
import torch

cfg   = load_config("configs/crazyflie.yaml")
ctrl  = CrazyfliePIDController.from_drone_config(cfg, num_envs=4, dt=0.002)
mixer = QuadMixer.from_drone_config(cfg)

# Single control step
root_state = torch.zeros(4, 13); root_state[:, 3] = 1.0
target_pos = torch.tensor([[0., 0., 1.]]).repeat(4, 1)

thrust, moment = ctrl(root_state, target_pos=target_pos,
                      command_level="position")
omega = mixer(thrust, moment)   # [4, 4] motor speeds in rad/s
```
