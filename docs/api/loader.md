# Configuration Loader

```python
from drone import load_config, DroneConfig
```

---

## `load_config`

```python
cfg = load_config("configs/crazyflie.yaml")   # str or Path
```

Parses a drone YAML file and returns a `DroneConfig` dataclass.
Raises `KeyError` if a required field is missing.

---

## `DroneConfig`

```python
@dataclass
class DroneConfig:
    physics:       DronePhysicsConfig
    attitude:      AttitudeControllerConfig
    position:      PositionControllerConfig
    cascade_pid: dict | None              # raw params dict, or None
    lee:           LeeControllerConfig | None
```

### `DronePhysicsConfig`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Drone name |
| `mass` | `float` | Mass [kg] |
| `inertia.ixx/iyy/izz` | `float` | Principal moments of inertia [kg·m²] |
| `max_thrust` | `float` | Total maximum thrust (all motors) [N] |
| `motor` | `MotorConfig \| None` | Motor/frame geometry (required for `QuadMixer`) |

### `MotorConfig`

Populated from the `drone.motor` YAML section. Required by `QuadMixer`.

| Field | Type | Unit | Description |
|---|---|---|---|
| `arm_length` | `float` | m | Center-to-motor distance |
| `k_thrust` | `float` | N·s² | Thrust coefficient: `F = k_thrust · ω²` |
| `k_drag` | `float` | N·m·s² | Drag coefficient: `τ = k_drag · ω²` |
| `layout` | `str` | — | `"x"` or `"+"` quad layout |
| `speed_min` | `float` | rad/s | Minimum motor speed (clamped by mixer) |
| `speed_max` | `float` | rad/s | Maximum motor speed (clamped by mixer) |

### `LeeControllerConfig`

Populated from the `controllers.lee` YAML section.

| Field | Type | Unit | Description |
|---|---|---|---|
| `position_gain` | `list[float]` | N/m | `k_pos` per axis |
| `velocity_gain` | `list[float]` | N·s/m | `k_vel` per axis |
| `attitude_gain` | `list[float]` | N·m/rad | `k_R` per axis |
| `angular_rate_gain` | `list[float]` | N·m·s/rad | `k_Ω` per axis |
| `max_acceleration` | `float` | m/s² | Clip on desired acceleration norm (default `inf`) |

### `AttitudeControllerConfig`

| Field | Type | Description |
|---|---|---|
| `freq_rate_hz` | `float` | Inner (rate) loop frequency [Hz] |
| `freq_angle_hz` | `float` | Outer (angle) loop frequency [Hz] |
| `rate.{roll,pitch,yaw}` | `PIDConfig` | Rate loop PID per axis |
| `angle.{roll,pitch,yaw}` | `PIDConfig` | Angle loop PID per axis |

### `PositionControllerConfig`

| Field | Type | Description |
|---|---|---|
| `freq_vel_hz` | `float` | Inner (velocity) loop frequency [Hz] |
| `freq_pos_hz` | `float` | Outer (position) loop frequency [Hz] |
| `max_thrust_scale` | `float` | Safety factor on `max_thrust` (0–1) |
| `max_horizontal_angle_deg` | `float` | Max roll/pitch from velocity controller [deg] |
| `velocity.{vx,vy,vz}` | `PIDConfig` | Velocity loop PID per axis |
| `position.{x,y,z}` | `PIDConfig` | Position loop PID per axis |

### `PIDConfig`

```python
@dataclass
class PIDConfig:
    kp:    float   # proportional gain
    ki:    float   # integral gain
    kd:    float   # derivative gain
    tau:   float   # derivative low-pass time constant [s]
    limit: float   # symmetric output saturation (±limit)
```

---

## Access examples

```python
cfg = load_config("configs/crazyflie.yaml")

# Physics
print(cfg.physics.mass)               # 0.027
print(cfg.physics.inertia.ixx)        # 1.657e-5
print(cfg.physics.max_thrust)         # 0.638

# Motor geometry (for QuadMixer)
print(cfg.physics.motor.arm_length)   # 0.046
print(cfg.physics.motor.k_thrust)     # 1.285e-8

# Attitude rate gains (roll axis)
print(cfg.attitude.rate.roll.kp)      # 50.0

# Position gains (z axis)
print(cfg.position.position.z.kp)     # 5.0
print(cfg.position.position.z.ki)     # 0.0

# Raw CascadePID params (dict)
print(cfg.cascade_pid["rate_kp"])   # [250.0, 250.0, 120.0]

# Lee geometric controller gains
print(cfg.lee.position_gain)          # [0.5, 0.5, 0.7]
print(cfg.lee.attitude_gain)          # [0.06, 0.06, 0.03]
```
