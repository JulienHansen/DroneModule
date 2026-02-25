# YAML Config Format

A config file has these sections:

| Section | Required | Used by |
|---|---|---|
| `drone` | Yes | All controllers |
| `drone.motor` | No | `QuadMixer` |
| `controllers.attitude` | Yes | `AttitudeControllerConfig` |
| `controllers.position` | Yes | `PositionControllerConfig` |
| `controllers.crazyflie_pid` | No | `CrazyfliePIDController` |
| `controllers.lee` | No | `LeePositionController` |

---

## `drone` — physical parameters

```yaml
drone:
  name: "Crazyflie 2.1"
  mass: 0.027            # [kg]
  inertia:               # diagonal inertia tensor [kg·m²]
    ixx: 1.657e-5
    iyy: 1.657e-5
    izz: 2.900e-5
  max_thrust: 0.638      # total thrust of all motors [N]
```

---

## `drone.motor` — motor and frame geometry

Required by `QuadMixer`. If absent, `cfg.physics.motor` is `None`.

```yaml
drone:
  motor:
    arm_length: 0.046      # m    — center-to-motor distance
    k_thrust:   1.285e-8   # N·s² — F = k_thrust · ω²
    k_drag:     7.645e-11  # N·m·s² — τ = k_drag · ω²
    layout:     x          # 'x' (default) or '+'
    speed_min:  0.0        # rad/s
    speed_max:  2618.0     # rad/s  (~25 000 RPM)
```

`k_thrust` and `k_drag` are identified experimentally on a thrust stand.
For the Crazyflie 2.1 these values come from Förster (2015).

---

## `controllers.crazyflie_pid` — CrazyfliePIDController gains

This section is passed directly as the `params` dict to
`CrazyfliePIDController`. All fields are optional; missing fields fall back
to the built-in firmware defaults.

### Loop rates

```yaml
crazyflie_pid:
  sim_rate_hz:             500.0   # expected call rate of the controller [Hz]
  pid_posvel_loop_rate_hz: 100.0   # position + velocity loop rate [Hz]
  pid_loop_rate_hz:        500.0   # attitude + rate loop rate [Hz]
```

`posvel_decimation = round(sim_rate / posvel_rate)` — how many sim steps
are skipped between pos/vel updates.

### Position loop — `error [m] → velocity setpoint [m/s]`

```yaml
  pos_kp:  [2.0, 2.0, 2.0]
  pos_ki:  [0.0, 0.0, 0.5]
  pos_kd:  [0.0, 0.0, 0.0]
  pos_kff: [0.0, 0.0, 0.0]
```

### Velocity loop — `error [m/s] → roll/pitch cmd [deg] or thrust Δ`

```yaml
  vel_kp:  [25.0, 25.0, 25.0]
  vel_ki:  [ 1.0,  1.0, 15.0]
  vel_kd:  [ 0.0,  0.0,  0.0]
  vel_kff: [ 0.0,  0.0,  0.0]
```

!!! note "Unit convention for x/y"
    `vel_kp/ki/kd` for the x and y axes are in **deg/(m/s)** — the
    controller multiplies them by `DEG2RAD` internally so the numbers
    match the firmware's integer register values.

### Attitude loop — `error [rad] → body-rate setpoint [rad/s]`

```yaml
  att_kp:  [6.0, 6.0, 6.0]
  att_ki:  [3.0, 3.0, 1.0]
  att_kd:  [0.0, 0.0, 0.35]
  att_kff: [0.0, 0.0, 0.0]
```

### Rate loop — `error [rad/s] → angular acceleration [rad/s²]`

```yaml
  rate_kp:  [250.0, 250.0, 120.0]
  rate_ki:  [500.0, 500.0,  16.7]
  rate_kd:  [  2.5,   2.5,   0.0]
  rate_kff: [  0.0,   0.0,   0.0]
```

Values match `platform_defaults_cf2.h` from the Crazyflie open-source firmware.

### Saturation limits

```yaml
  att_integral_limit_deg:  [20.0, 20.0, 360.0]   # integral cap [deg]
  rate_integral_limit_deg: [33.3, 33.3, 166.7]   # integral cap [deg/s]
  pos_vel_max: [1.0, 1.0, 1.0]                   # velocity setpoint cap [m/s]
  roll_max_deg:  20.0
  pitch_max_deg: 20.0
  yaw_max_delta: 0.0        # 0 = unlimited; >0 clamps yaw change per step
```

### Thrust parameters

```yaml
  thrust_base:      30000.0   # hover throttle estimate [PWM 0–65535]
  thrust_min:       20000.0   # minimum throttle [PWM]
  thrust_cmd_max:   65535.0   # full-scale throttle [PWM]
  vel_thrust_scale: 1000.0    # (m/s²) → ΔPWMconversion
```

`thrust_cmd_scale` ([N/PWM]) is **derived automatically** from
`drone.max_thrust / thrust_cmd_max` and does not need to be set manually.

---

## `controllers.lee` — LeePositionController gains

Optional. If absent, `cfg.lee` is `None`.

```yaml
controllers:
  lee:
    position_gain:     [0.50, 0.50, 0.70]    # k_pos  [N/m]
    velocity_gain:     [0.20, 0.20, 0.30]    # k_vel  [N·s/m]
    attitude_gain:     [0.06, 0.06, 0.030]   # k_R    [N·m/rad]
    angular_rate_gain: [0.002, 0.002, 0.001] # k_Ω    [N·m·s/rad]
    # max_acceleration: inf                  # optional [m/s²]
```

Gains are in SI units. Values above are tuned for the Crazyflie 2.1 via
pole placement at ω_pos ≈ 4 rad/s, ω_att ≈ 60 rad/s, ζ = 0.7.

---

## `controllers.attitude` — AttitudeControllerConfig

Used to populate `cfg.attitude` (available to all controllers via
`DroneConfig`).

```yaml
controllers:
  attitude:
    freq_rate_hz:  500
    freq_angle_hz: 500
    rate:
      roll:  { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
      pitch: { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
      yaw:   { kp: 50.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 50.265 }
    angle:
      roll:  { kp: 4.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }
      pitch: { kp: 4.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }
      yaw:   { kp: 3.0, ki: 0.0, kd: 0.0, tau: 0.01, limit: 15.708 }
```

`limit` is the **symmetric output saturation** bound (±limit).
Units: rate loop [rad/s²], angle loop [rad/s].

---

## `controllers.position` — PositionControllerConfig

```yaml
  position:
    freq_vel_hz: 100
    freq_pos_hz: 100
    max_thrust_scale: 0.8            # effective max = 0.8 × drone.max_thrust
    max_horizontal_angle_deg: 30.0
    velocity:
      vx: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
      vy: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
      vz: { kp: 1.0, ki: 0.0, kd: 0.0, tau: 0.10, limit: 9999.0 }
    position:
      x: { kp: 5.0, ki: 0.0, kd: 1.0, tau: 0.10, limit: 10.0 }
      y: { kp: 5.0, ki: 0.0, kd: 1.0, tau: 0.10, limit: 10.0 }
      z: { kp: 5.0, ki: 0.0, kd: 3.5, tau: 0.10, limit:  5.0 }
```

---

## Adding a new drone

1. Copy an existing config file:
   ```bash
   cp configs/crazyflie.yaml configs/my_drone.yaml
   ```
2. Update the `drone` section (mass, inertia, max_thrust).
3. Update `drone.motor` if you plan to use `QuadMixer` (measure `k_thrust`
   and `k_drag` on a thrust stand, or use manufacturer data).
4. Tune the `crazyflie_pid` gains using `tune_from_physics`
   (see the [tuning guide](../tuning.md)) or manually.
5. Load it:
   ```python
   cfg   = load_config("configs/my_drone.yaml")
   ctrl  = CrazyfliePIDController.from_drone_config(cfg, num_envs=N, dt=dt)
   mixer = QuadMixer.from_drone_config(cfg)
   ```
