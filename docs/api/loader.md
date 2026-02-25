# Configuration Loader

```python
from drone_control import load_config, DroneConfig
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
    crazyflie_pid: dict | None          # raw params dict, or None
```

### `DronePhysicsConfig`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Drone name |
| `mass` | `float` | Mass [kg] |
| `inertia.ixx/iyy/izz` | `float` | Principal moments of inertia [kg·m²] |
| `max_thrust` | `float` | Total maximum thrust (all motors) [N] |

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

# Attitude rate gains (roll axis)
print(cfg.attitude.rate.roll.kp)      # 50.0

# Position gains (z axis)
print(cfg.position.position.z.kp)    # 5.0
print(cfg.position.position.z.ki)    # 0.0

# Raw CrazyfliePID params (dict)
print(cfg.crazyflie_pid["rate_kp"])   # [250.0, 250.0, 120.0]
```
