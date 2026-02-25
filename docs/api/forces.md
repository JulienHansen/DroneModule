# Forces

External force and torque models applied during rigid-body integration.

```python
from drone.forces import ForceModel, Gravity, BodyDrag
```

---

## Convention

Every force model returns two tensors:

| Tensor | Shape | Frame | Unit |
|---|---|---|---|
| `force` | `[N, 3]` | **World** frame | N |
| `torque` | `[N, 3]` | **Body** frame | N·m |

Forces are accumulated and summed before integration:

```
F_total   = F_thrust_world + Σ force_i    (world frame)
τ_total   = moment_ctrl    + Σ torque_i   (body frame)
```

---

## `ForceModel` (abstract base)

```python
from drone.forces import ForceModel

class MyForce(ForceModel):
    def compute(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state: [N, 13]  →  (force [N,3], torque [N,3])
        ...

    def to(self, device: str) -> "MyForce":   # optional
        ...
        return self
```

### `compute(state)`

| Parameter | Description |
|---|---|
| `state` | `[N, 13]` — `[pos(3)\|quat(4)\|lin_vel(3)\|ang_vel(3)]` |

Returns `(force [N, 3], torque [N, 3])`.

---

## `Gravity`

Constant downward force along world -Z.

```
F = [0, 0, −mass · g]    τ = [0, 0, 0]
```

```python
Gravity(mass, g=9.81, device="cpu")
```

| Parameter | Type | Description |
|---|---|---|
| `mass` | `float` | Drone mass [kg] |
| `g` | `float` | Gravitational acceleration [m/s²]. Default 9.81 |
| `device` | `str` | PyTorch device |

### `from_drone_config`

```python
grav = Gravity.from_drone_config(cfg, device="cpu")
```

Reads `mass` from `cfg.physics.mass`.

---

## `BodyDrag`

Linear aerodynamic drag proportional to velocity.

```
F = −k ⊙ v    (world frame)    τ = [0, 0, 0]
```

```python
BodyDrag(coefficients, device="cpu")
```

| Parameter | Type | Description |
|---|---|---|
| `coefficients` | `float \| list[float]` | Drag coefficient(s) [N·s/m]. Scalar → isotropic. List → per-axis `[kx, ky, kz]` |
| `device` | `str` | PyTorch device |

```python
drag = BodyDrag(0.05)                        # isotropic
drag = BodyDrag([0.05, 0.05, 0.12])          # more drag vertically
```

### `from_drone_config`

```python
drag = BodyDrag.from_drone_config(cfg, device="cpu")
```

Reads `drone.drag_coefficients` from the YAML if present; defaults to
zero drag if the field is absent.

YAML example:

```yaml
drone:
  drag_coefficients: [0.05, 0.05, 0.10]   # [N·s/m]
```

---

## Writing a custom force model

Any object subclassing `ForceModel` can be added to a `Drone`:

```python
import torch
from drone.forces import ForceModel

class WindDisturbance(ForceModel):
    """Constant wind force in the world frame."""

    def __init__(self, wind_vec: list[float], device: str = "cpu"):
        self.device = torch.device(device)
        self._wind = torch.tensor(wind_vec, dtype=torch.float32, device=self.device)

    def to(self, device):
        self.device = torch.device(device)
        self._wind = self._wind.to(device)
        return self

    def compute(self, state):
        N = state.shape[0]
        force  = self._wind.unsqueeze(0).expand(N, 3)
        torque = torch.zeros(N, 3, device=self.device)
        return force, torque
```

Usage:

```python
from drone import Drone, Gravity

drone = Drone.from_drone_config(
    cfg, controller=ctrl,
    forces=[
        Gravity.from_drone_config(cfg),
        WindDisturbance([0.3, 0.0, 0.0]),   # 0.3 N headwind
    ],
)
```
