# Architecture

## Loop hierarchy

`CrazyfliePIDController` implements a **4-level cascaded PID** architecture
that mirrors the Crazyflie 2.x firmware:

```
                 ┌──────────────────────────────────────────────────┐
 target_pos ──▶  │  Position loop   (pos_pid)    @ 100 Hz           │──▶ vel_sp
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
 vel_sp     ──▶  │  Velocity loop   (vel_pid)    @ 100 Hz           │──▶ roll/pitch/thrust_cmd
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
 att_sp     ──▶  │  Attitude loop   (att_pid)    @ 500 Hz           │──▶ rate_sp
                 └──────────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────────┐
 rate_sp    ──▶  │  Rate loop       (manual PID) @ 500 Hz           │──▶ moment [N·m]
                 └──────────────────────────────────────────────────┘
```

You can **enter the cascade at any level** via `command_level`.

---

## Multi-rate scheduling (decimation)

The simulation runs at a fixed timestep `dt` (typically 2 ms = 500 Hz).
Slower loops are skipped on most steps:

```
sim step:   0    1    2    3    4    5    6    7    8    9   10  ...
pos/vel:    ✓                        ✓                        ✓   (100 Hz)
att/rate:   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   (500 Hz)
```

`posvel_decimation = round(sim_rate / posvel_rate)` = 500/100 = **5**
`att_decimation    = round(sim_rate / att_rate)`    = 500/500 = **1**

Between decimated steps, velocity and position setpoints are **held constant**.

---

## Position loop

**Input:** position error [m]
**Output:** velocity setpoint [m/s], clamped to `pos_vel_max`

```
e_pos = target_pos − pos
vel_sp = clamp(pos_kp·e_pos + pos_ki·∫e_pos, −vel_max, +vel_max)
```

---

## Velocity loop

**Input:** velocity error in body-horizontal frame [m/s]
**Output:** roll/pitch setpoints [rad] + raw thrust command [PWM]

Velocity is first projected into the body-horizontal frame using the
current yaw angle, so that commands are always given in the world frame:

```
vel_body_x =  cos(ψ)·vx + sin(ψ)·vy
vel_body_y = −sin(ψ)·vx + cos(ψ)·vy
```

Horizontal output (converted from deg to rad internally):

```
pitch_cmd = clamp( vel_kp_x · e_vbx, −pitch_max, +pitch_max )
roll_cmd  = clamp(−vel_kp_y · e_vby, −roll_max,  +roll_max  )
```

Vertical output:

```
thrust_cmd = thrust_base + vel_kp_z · e_vz · vel_thrust_scale
thrust_cmd = clamp(thrust_cmd, thrust_min, thrust_max)
```

---

## Attitude loop

**Input:** attitude error [rad] (roll, pitch, yaw)
**Output:** body-rate setpoint [rad/s]

```
e_att  = wrap(att_sp − att_actual)
rate_sp = att_kp·e_att + att_ki·∫e_att + att_kd·ė_att
```

---

## Rate loop & Euler equation

**Input:** body-rate error [rad/s]
**Output:** moment [N·m]

Uses the **full rigid-body Euler equation**:

```
τ = J · α_ref + ω × (J · ω)
```

where:

- `α_ref` = rate PID output (angular acceleration setpoint [rad/s²])
- `ω × (J·ω)` = gyroscopic compensation term

The derivative is computed **on the measurement** (not the error) to avoid
derivative kick when the rate setpoint changes:

```
α_ref = kp·e_ω + ki·∫e_ω − kd·ω̇_meas
```

---

## Yaw setpoint management

Yaw is handled separately from roll/pitch because it integrates over time:

- **Absolute target** (`target_yaw`): setpoint is set directly, optionally
  clamped to within `yaw_max_delta` of the current heading.
- **Rate target** (`target_yaw_rate`): setpoint is integrated each step
  (`yaw_sp += yaw_rate · dt`) and wrapped to `[−π, π]`.
- If neither is provided, yaw setpoint holds its last value.

---

## PID implementation

All four loops use `PID_Vectorized`, which supports:

| Feature | Detail |
|---|---|
| Integration | Trapezoidal rule |
| Derivative filter | Tustin bilinear (`tau > 0`) or backward difference (`tau = 0`) |
| Derivative on measurement | Pass `measurement_dot` to avoid kick |
| Anti-windup | Back-calculation from output saturation |
| Integral clamping | Separate `integral_limit` before output saturation |
| Feed-forward | `kff * feedforward` added to output |
| Multi-axis | Single instance handles all 3 axes simultaneously (`[N, 3]` tensors) |

---

## Output units

| Output | Shape | Unit |
|---|---|---|
| `thrust` | `[N, 1]` | Newtons [N] |
| `moment` | `[N, 3]` | Newton-metres [N·m] |

Internally, thrust is managed in raw PWM units (0–65535) and converted to
Newtons using `thrust_cmd_scale = max_thrust / thrust_cmd_max`, derived
automatically from the drone's `max_thrust` in the YAML config.
