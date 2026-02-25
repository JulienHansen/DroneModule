# Modern Quadcopter Firmware Architecture

This document describes the complete sensor-to-actuator pipeline of a modern quadcopter flight controller, from raw IMU samples to PWM/DShot commands sent to the ESCs. Three reference firmwares are compared throughout: **Betaflight** (racing/freestyle), **PX4** (autonomous UAV), and **Crazyflie** (research micro-drone).

---

## 1. High-Level Pipeline

```
┌──────────────┐   raw        ┌──────────────┐  filtered    ┌──────────────────┐
│   Sensors    │ ──────────▶  │  IMU Filter  │ ──────────▶  │ State Estimator  │
│ IMU/Baro/GPS │              │  (LPF / RPM) │              │  (EKF/Mahony)    │
└──────────────┘              └──────────────┘              └────────┬─────────┘
                                                                      │ pose, vel, rates
                                                                      ▼
┌──────────────┐  setpoint    ┌──────────────┐              ┌────────────────────┐
│ RC / Mission │ ──────────▶  │  Reference   │ ──────────▶  │  Control Cascade   │
│   Planner    │              │  Generator   │              │  pos→vel→att→rate  │
└──────────────┘              └──────────────┘              └────────┬───────────┘
                                                                      │ thrust + moments
                                                                      ▼
                                                             ┌────────────────────┐
                                                             │  Control Allocator │
                                                             │    (QuadMixer)     │
                                                             └────────┬───────────┘
                                                                      │ ω per motor
                                                                      ▼
                                                             ┌────────────────────┐
                                                             │   Motor / ESC      │
                                                             │ DShot / OneShot    │
                                                             └────────────────────┘
```

---

## 2. Sensors and Acquisition

### 2.1 IMU

The Inertial Measurement Unit is the heart of every flight controller. It provides:

| Signal | Typical sensor | Rate |
|--------|---------------|------|
| Angular velocity (gyroscope) | ICM-42688-P, MPU-6000 | 3.2–8 kHz |
| Linear acceleration (accelerometer) | Same die | 1–4 kHz |
| Magnetic heading (magnetometer, optional) | QMC5883L | 100–200 Hz |

The gyroscope is the most critical signal: it feeds the innermost (rate) control loop. Any latency or noise here directly degrades attitude tracking.

**Oversampling.** Modern firmwares (Betaflight ≥ 4.0) run the IMU SPI bus at 8 MHz and read the sensor at its full ODR (e.g., 8 kHz for the ICM-42688-P), then decimate to the control loop rate. This converts quantisation noise into broadband noise that is more easily filtered.

### 2.2 Barometer and GPS

| Sensor | Purpose | Typical rate |
|--------|---------|-------------|
| Barometer (BMP388, DPS310) | Altitude hold, climb rate | 50–200 Hz |
| GPS (M10, ZED-F9P) | Position hold, return-to-home | 1–20 Hz |
| Optical flow (PMW3901) | Low-altitude position hold (no GPS) | 100–400 Hz |
| LiDAR / ToF (VL53L5CX) | Terrain-following, landing | 30–60 Hz |

GPS and barometer readings are always fused inside the state estimator because their rates are too low to feed control loops directly.

---

## 3. IMU Signal Processing

Before state estimation, raw sensor data passes through several filtering stages.

### 3.1 Static hardware anti-alias filter

MEMS IMUs include an on-chip low-pass filter configurable via SPI. It is set to slightly below the Nyquist frequency of the ODR. For a 3.2 kHz ODR the cutoff is typically 1.6 kHz.

### 3.2 RPM-notch filters (Betaflight Dynamic Notch)

Motor vibrations appear as narrow harmonic peaks in the gyro spectrum at exact multiples of the motor electrical frequency:

```
f_vibe = N_poles/2 × RPM/60   (Hz)
```

**Bidirectional DShot** (Betaflight ≥ 4.1, PX4 ≥ 1.13) reads back actual motor RPM via the ESC telemetry line at ~1 kHz. Each firmware loop updates per-motor notch filters centered on the 1st, 2nd, and 3rd harmonics. Because the notch follows the exact motor speed, the filter can be very narrow (high Q ≈ 250–500), introducing negligible phase delay at other frequencies.

Without RPM feedback, firmwares fall back to a dynamic-frequency notch that tracks peaks in the real-time FFT of the gyro signal (Betaflight Dynamic Notch v2).

### 3.3 Gyro low-pass filter (LPF)

A biquad (2nd-order IIR) or PT1 (1st-order) filter removes broadband high-frequency noise. Typical cutoffs:

- **Racing quad**: 200–400 Hz (minimise phase delay → better feel)
- **Cinema/freestyle**: 100–200 Hz (smoother at cost of latency)
- **Crazyflie**: 80 Hz biquad (small, light, resonant frame)

**Phase lag matters.** Each biquad at 200 Hz introduces roughly 3–8° of phase delay at 50 Hz (the typical mechanical bandwidth). Excessive filtering is a primary cause of oscillations because the PID derivative term amplifies delayed noise.

### 3.4 Accelerometer LPF

The accelerometer is not used by the rate loop. It only feeds state estimation and is filtered aggressively (10–30 Hz cutoff) because vibration contamination is severe.

---

## 4. State Estimation

### 4.1 Complementary filter (Mahony / Madgwick)

Used in Betaflight, Crazyflie, and any firmware prioritising speed over accuracy.

**Mahony filter** (Crazyflie default):

```
ω_mes = (acc × g_est + mag × m_est) × k_P + ∫(·)×k_I
ω_filt = ω_gyro + ω_mes
q̇ = ½ q ⊗ [0, ω_filt]
q ← q / ‖q‖
```

The vector cross-product `acc × g_est` gives a rotation error proportional to the tilt. The integral term removes gyro bias. This runs at the full IMU rate (500–1000 Hz on Crazyflie).

**Madgwick filter** (slightly more accurate, same cost): uses gradient descent on the quaternion to minimise the algebraic error between predicted and measured gravity/magnetic directions.

Both filters have one tunable parameter (the fusion gain `β` for Madgwick, `k_P/k_I` for Mahony) that sets the trade-off between gyro tracking (low gain = trusts gyro, drifts slowly) and accelerometer correction (high gain = rejects drift, noisier).

### 4.2 Extended Kalman Filter (EKF)

Used in PX4 (`ekf2`) and for position estimation in Crazyflie.

The EKF maintains a full state vector:

```
x = [p, v, q, b_g, b_a, b_mag, b_wind, ...]ᵀ
```

where `p` is position, `v` velocity, `q` attitude quaternion, and the `b_*` terms are biases. The prediction step integrates the IMU using Newton–Euler dynamics; the correction step fuses GPS, barometer, magnetometer, optical flow, and visual-inertial odometry (VIO) with individual noise models.

PX4's `ekf2` runs at 100–250 Hz. The innovation (measurement residual) is monitored continuously; large innovations trigger automatic sensor fault detection.

### 4.3 Position estimation

For indoor hover without GPS, PX4 and Crazyflie use **optical flow + barometer fusion**:

```
v_body ≈ f_focal × (flow_px / dt) / altitude
p += ∫ R × v_body × dt   (EKF prediction)
p_corrected = EKF(baro, flow)
```

For outdoor precision (RTK), a **moving baseline** GPS heading + 5 cm RTK position replaces the magnetometer and barometer respectively.

---

## 5. Reference Generation

### 5.1 RC input decoding

The radio receiver sends commands on SBUS, CRSF, or ELRS at 50–1000 Hz. Raw PWM values (1000–2000 µs) are normalised to [-1, 1]:

```
stick = (pwm - 1500) / 500
```

Sticks in this package's `rate_profiles.py` operate on exactly this normalised input.

### 5.2 Flight mode stack

| Mode | What RC controls | Who generates rate setpoint |
|------|-----------------|----------------------------|
| **Acro** | Body rates directly | Rate profile (this package) |
| **Angle** | Euler angle setpoints | Outer angle PID |
| **Altitude hold** | Climb rate only | Outer Z velocity PID |
| **Position hold** | 2D velocity, yaw rate | Outer XY position PID |
| **Autonomous** | Waypoints / trajectory | Trajectory planner |

### 5.3 Rate profiles

In Acro mode the stick is passed through a non-linear shaping curve before it becomes a rate setpoint. Four common profiles are implemented in this package (`betaflight_rate_profile`, `raceflight_rate_profile`, `actual_rate_profile`, `kiss_rate_profile`). They all share the same anatomy:

```
rate_sp = non_linear_curve(stick, rc_rate, expo, super_expo, limit)
         ──────────────────────────────────────────────────────────
         converts normalised stick [-1,1] → body rate [rad/s]
```

A high `rc_rate` gives fast, nervous response. Adding `super_expo` provides a deadband near centre while still reaching the same maximum rate at full stick. See [API Reference](../api/controller.md) for details.

### 5.4 Trajectory planning (autonomous mode)

For waypoint flight, PX4 runs a **jerk-limited polynomial trajectory planner** (`FlightTaskAutoMapper`) that generates smooth position/velocity/acceleration references at 50 Hz. Crazyflie uses a similar approach with the Crazyflie Commander stack and the optional `trajectorypy` high-level controller.

---

## 6. Control Cascade

All modern quadcopter firmwares use a **cascade (nested-loop) architecture**. Each outer loop runs slower than its inner counterpart, and the output of one loop is the setpoint of the next.

```
Position  →  Velocity  →  Attitude  →  Body Rate  →  Motor Torque
10–50 Hz      50–100 Hz   100–500 Hz    1–8 kHz        —
```

### 6.1 Position loop (10–50 Hz)

**Input**: position setpoint [m], measured position [m]
**Output**: velocity setpoint [m/s]

Simple proportional or PD controller. Integrator is usually omitted here and added in the velocity loop instead.

```
v_sp = kp_pos × (p_sp - p) + kd_pos × (ṗ_sp - ṗ)
```

In this package: `PosController_Vectorized.run_pos()`.

### 6.2 Velocity loop (50–100 Hz)

**Input**: velocity setpoint [m/s], measured velocity [m/s]
**Output**: desired linear acceleration [m/s²] (then decomposed into thrust + attitude)

```
a_des = kp_vel × (v_sp - v) + ki_vel × ∫(v_sp - v) + mass × g × ẑ   # world frame
```

The total desired force vector `F_des = mass × a_des` is then decomposed:

- **Thrust**: `T = ‖F_des‖` (scalar)
- **Desired attitude**: align body-z with `F_des / ‖F_des‖`
- **Roll/pitch setpoint**: `φ_sp = arctan(-F_des,y / F_des,z)`, `θ_sp = arctan(F_des,x / F_des,z)`

A horizontal angle limit is applied to prevent unreasonable commands: `φ_sp, θ_sp ∈ [-30°, 30°]`.

In this package: `PosController_Vectorized.run_vel()`.

### 6.3 Attitude (angle) loop (100–500 Hz)

**Input**: attitude setpoint (Euler RPY or quaternion), measured attitude
**Output**: body-rate setpoint [rad/s]

**PID approach** (Betaflight angle mode, Crazyflie):

```
ω_sp = kp_att × (q_sp ⊖ q_meas)   # error expressed in body frame
```

The quaternion error `q_sp ⊖ q_meas = q_meas⁻¹ ⊗ q_sp` is converted to a rotation vector (axis × angle). Yaw is often handled separately to avoid coupling with roll/pitch.

**Geometric (SO(3)) approach** (Lee controller, PX4 mc_att_control):

The attitude error is computed on the Lie group directly:

```
e_R = ½ vee(R_desᵀ R - Rᵀ R_des)
ω_sp = -kR × e_R - kΩ × (ω - Rᵀ R_des ω_des)
```

This avoids singularities (gimbal lock) present in Euler-angle PID and has globally stable convergence proofs under mild conditions. See `LeePositionController` in this package.

### 6.4 Rate loop (500 Hz – 8 kHz)

The rate loop is the fastest and most critical control loop. It runs at the gyroscope sample rate (or half of it on constrained hardware).

**Input**: body-rate setpoint [rad/s] from attitude loop, measured body rates [rad/s]
**Output**: angular acceleration commands → body moments [N·m] → `(T, Mx, My, Mz)` wrench

#### Classical PID (Betaflight, Crazyflie)

```
e = ω_sp - ω_meas

P = kp × e
I += ki × e × dt                    # with anti-windup
D = -kd × (ω_meas - ω_meas_prev)/dt # D on measurement, not error (prevents derivative kick)

output = P + I + D
```

The derivative is applied to the **measurement** (not the error) to avoid impulse spikes when the setpoint steps. Betaflight further applies a dedicated D-term low-pass filter (cutoff 70–150 Hz) separate from the gyro LPF.

**Feed-forward** (Betaflight ≥ 3.5): add a term proportional to the rate setpoint derivative to reduce phase lag on sharp inputs:

```
FF = kff × (ω_sp - ω_sp_prev) / dt
output += FF
```

#### INDI — Incremental Nonlinear Dynamic Inversion

Used in advanced firmwares (Paparazzi, some PX4 configurations, TU Delft research):

The idea is to use the gyro acceleration (numerical derivative of ω, filtered) as a direct measurement of the applied torque, and compute motor increments that produce the needed torque correction:

```
dω/dt_measured ≈ (ω[k] - ω[k-1]) / dt

Δu = G⁻¹ × (dω/dt_desired - dω/dt_measured)
u[k] = u[k-1] + Δu
```

where `G` is the control effectiveness matrix (relates motor speed increments to angular accelerations). INDI is inherently robust to model uncertainty because it operates on increments and relies on measurement rather than a plant model. Its main cost is sensitivity to gyro noise (requires aggressive differentiation).

---

## 7. Control Allocation

Once the controller outputs a **wrench** `[T, Mx, My, Mz]`, the allocation step maps it to per-motor speed commands.

### 7.1 The allocation matrix

For a quadrotor, each motor `i` at position `(xi, yi)` contributes:

```
T  = k_t × Σ ωi²
Mx = k_t × Σ yi × ωi²          (roll)
My = k_t × Σ (-xi) × ωi²       (pitch)
Mz = k_d × Σ si × ωi²          (yaw, si = ±1 spin direction)
```

This gives a linear system `wrench = B × ωsq` where `ωsq = [ω0², ..., ω3²]`. The 4×4 allocation matrix `B` is inverted once at startup:

```
ωsq = B⁻¹ × wrench
ωi  = √(max(ωsqi, ωmin²))
```

`QuadMixer` in this package implements this exactly. The key parameters are the motor's **thrust coefficient** `k_t` (N·s²) and **drag coefficient** `k_d` (N·m·s²), which are identified experimentally (thrust stand measurement).

### 7.2 Saturation and prioritisation

When the desired wrench cannot be achieved (e.g., maximum throttle while demanding a large roll), the allocation must choose what to sacrifice. Two common strategies:

1. **Simple clamping** (Betaflight): clamp `ωi` to `[ωmin, ωmax]` and accept the wrench error.
2. **Prioritised allocation** (PX4, academic): iteratively scale down lower-priority axes (yaw first, then roll/pitch, then thrust) to remain within motor limits while satisfying higher-priority commands.

### 7.3 Over-actuated systems

Hex- and octorotors have more motors than DOF (6 or 8 vs 4). The pseudo-inverse `B† = Bᵀ(BBᵀ)⁻¹` minimises the 2-norm of motor speeds for a given wrench. Some allocators additionally minimise power consumption (weighted pseudo-inverse) or distribute load evenly.

---

## 8. Motor and ESC Interface

### 8.1 Protocol evolution

| Protocol | Type | Rate | Latency | Notes |
|----------|------|------|---------|-------|
| Standard PWM | Analog, unidirectional | 50–400 Hz | ~1 ms | Original RC servo protocol |
| OneShot125 | Analog, faster | 2 kHz | ~500 µs | Narrower pulse, same idea |
| Multishot | Analog | 32 kHz | ~30 µs | Rarely used now |
| DShot150/300/600 | Digital, unidirectional | 2–8 kHz | ~50 µs | Dominant standard since 2016 |
| DShot1200 | Digital | 16 kHz | ~25 µs | High-speed, needs quality wiring |
| Bidirectional DShot | Digital, bidirectional | 2–8 kHz | ~50 µs | ESC sends RPM eRPM back |
| UAVCAN / DroneCAN | CAN bus | 1 kHz | ~1 ms | Robotics/autopilot use |
| FDCAN | CAN-FD | 4 kHz | ~250 µs | Future high-bandwidth CAN |

**DShot** encodes the motor command as a 16-bit word: 11 bits of throttle (0–2047), 1 telemetry request bit, and 4-bit CRC. The digital nature eliminates calibration and provides error detection.

**Bidirectional DShot** (also called DShot EDT, extended telemetry) reuses the signal wire: after the FC transmits a DShot frame, it tristates the pin and the ESC transmits back the eRPM using inverted DShot encoding. This enables RPM-linked notch filters as described in Section 3.2.

### 8.2 ESC firmware

Modern ESCs run **BLHeli_32**, **AM32**, or **BLHeli_S**. They implement:

- **FOC (Field-Oriented Control)** or trapezoidal (6-step) commutation
- **Active freewheeling**: MOSFETs synchronously rectify during deceleration, recovering energy and reducing motor temperature
- **Demag compensation**: corrects back-EMF measurement errors at low RPM (startup)
- **RPM governor** (some firmwares): inner RPM loop that makes the motor's effective output proportional to throttle, linearising the `ω → T` relationship for the flight controller

### 8.3 Motor dynamics and latency budget

The electrical time constant of a typical FPV brushless motor is τ_e ≈ 0.1–1 ms. The mechanical time constant depends on the propeller inertia: typically τ_m ≈ 20–80 ms for a 3" racing quad, up to 150 ms for a 5" with heavier props.

Total latency budget (from gyro sample to motor output):

```
IMU read      :   ~0.1 ms
Gyro filter   :   ~0.3–2 ms  (depends on LPF cutoff)
State estimate:   ~0.1 ms
Control loops :   ~0.1 ms
DShot frame   :   ~0.05 ms (DShot300 at 8 kHz)
ESC processing:   ~0.1 ms
Motor step    :   20–80 ms (mechanical τ)
──────────────────────────
Total (non-mech): ~1–3 ms
```

The mechanical time constant dominates, which is why aerobatic quads use very stiff, light propellers.

---

## 9. Failsafe and Safety Systems

### 9.1 RC link loss

When the FC detects missing RC frames (typically 100–500 ms timeout), it enters **failsafe**:

1. **Stage 1** (drop): immediately drop throttle to zero (racing/FPV use)
2. **Stage 2** (return-to-home): execute a programmed RTH sequence (autonomous use)

Betaflight implements "Drop", "Land", or "GPS Rescue" depending on the vehicle class and GPS availability.

### 9.2 Sensor health monitoring

- **IMU sanity check**: if gyro reads > ±4000 deg/s (physical limit), it flags a hardware fault
- **EKF innovation check** (PX4): large innovation (measurement residual vs prediction) in GPS or baro triggers sensor rejection
- **Motor desync detection** (BLHeli_32): if the ESC loses commutation timing, it re-arms the motor and reports a fault via DShot telemetry
- **Voltage/current monitoring**: Li-Po under-voltage triggers a landing or RTH

### 9.3 Arming logic

Flight controllers require an explicit arming sequence before motors can spin:

1. **Throttle low** (stick < 1060 µs or normalised < -0.88)
2. **Arm gesture** (yaw right and hold 0.5 s, or dedicated arm switch)
3. **Pre-arm checks**: GPS lock, EKF health, motor test, battery level

This prevents accidental spin-up on the bench.

---

## 10. Firmware Comparisons

### 10.1 Betaflight

**Target**: FPV racing, freestyle, cinematic
**Language**: C, heavily optimised for STM32 MCUs (F4/F7/H7)
**Control rate**: 1–8 kHz rate loop (PID), RPM-linked notch + dynamic notch, bidirectional DShot

| Feature | Details |
|---------|---------|
| Estimation | Mahony complementary filter (attitude only) |
| Outer loops | Angle mode only; no onboard position hold without GPS module |
| Mixer | Static X/+ allocation, saturation via clamping |
| Feed-forward | Stick velocity → rate FF term |
| INDI | Not standard; experimental "PIDSUM_LIMIT" variant |
| DShot | DShot150–1200, bidirectional DShot EDT |
| Configurator | Betaflight Configurator (Chromium-based GUI) |

Betaflight is the reference for latency-minimised attitude control. The 8 kHz loop (F7/H7 target) is rarely bottlenecked by computation.

### 10.2 PX4

**Target**: Autonomous UAVs, delivery drones, research
**Language**: C++ on NuttX RTOS (Pixhawk) or Linux (companion computer)
**Control rate**: 250–500 Hz attitude, 50–100 Hz position

| Feature | Details |
|---------|---------|
| Estimation | `ekf2`: full-state EKF (pos, vel, att, biases) fusing IMU + GPS + baro + mag + optflow + VIO |
| Outer loops | Full position/velocity cascade, trajectory planner, auto-takeoff/land, RTH |
| Mixer | Geometric + prioritised allocation via `ControlAllocator` module |
| Rate controller | PID with anti-windup + feed-forward |
| INDI | Supported as `mc_rate_control` option (`MC_AT_EN`) |
| DShot | DShot150–600 via dedicated timer DMA driver |
| Configurator | QGroundControl, MAVLink protocol |

PX4's key strength is its modular uORB publish-subscribe middleware: any module can subscribe to `vehicle_local_position`, `vehicle_attitude`, etc. without knowing the sensor source.

### 10.3 Crazyflie firmware

**Target**: Research, indoor swarms, education
**Language**: C on FreeRTOS (STM32F405)
**Control rate**: 500 Hz PID (attitude + rate), 100 Hz position

| Feature | Details |
|---------|---------|
| Estimation | Mahony (attitude) + EKF (position, using UWB/LPS, optflow, or lighthouse) |
| Outer loops | Full cascade; position setpoints from CRTP over radio |
| Mixer | Direct PWM to motor driver (no DShot); software mixer |
| Rate controller | Cascade PID matching `pid_attitude.c` (matches this package's `CrazyfliePIDController`) |
| INDI | Not standard |
| Protocols | CRTP over 2.4 GHz radio (custom), USB, UART |
| Configurator | cfclient Python GUI, `cflib` Python API |

Crazyflie's defining characteristic is its **radio CRTP protocol** and high-level commander stack, which allows a Python program running on a laptop to stream position setpoints at 100 Hz to a swarm of drones with minimal firmware-side changes.

---

## 11. Connecting to This Package

The `drone_control` package implements the **control cascade** (Sections 6) and **control allocation** (Section 7) layers:

```
This package:
  PosController_Vectorized   → position + velocity loops (§6.1, §6.2)
  CrazyfliePIDController     → full cascade (§6.1–6.4) matching Crazyflie firmware
  LeePositionController      → geometric position + attitude control (§6.3 SO(3) variant)
  QuadMixer                  → control allocation (§7.1)
  betaflight_rate_profile    → RC reference generation (§5.3)
  tune_from_physics          → pole-placement gain tuning

External (not in package):
  Sensor acquisition         → hardware driver / ROS topic
  IMU filtering              → e.g., scipy IIR in Python, or hardware firmware
  State estimation           → Madgwick/Mahony/EKF (e.g., ahrs, pyekf, IsaacLab)
  ESC communication          → DShot via UART or direct hardware
```

A typical simulation loop with this package:

```python
from drone_control import load_config, LeePositionController, QuadMixer

cfg    = load_config("configs/crazyflie.yaml")
ctrl   = LeePositionController.from_drone_config(cfg, num_envs=N, device="cuda")
mixer  = QuadMixer.from_drone_config(cfg, device="cuda")

# Simulation step (called at ~500 Hz)
thrust, moment = ctrl(root_state, pos_sp, vel_sp, acc_sp, yaw_sp, yaw_rate_sp)
omega = mixer(thrust, moment)   # [N, 4] rad/s → send to sim motors
```

---

## 12. Further Reading

- T. Lee, M. Leok, N. H. McClamroch — *Geometric tracking control of a quadrotor UAV on SE(3)*, CDC 2010
- F. Kendoul — *Survey of advances in guidance, navigation and control of UAVs*, JFR 2012
- S. Bouabdallah — *Design and control of quadrotors with application to autonomous flying*, EPFL 2007
- [Betaflight wiki](https://betaflight.com/docs/) — rate profiles, RPM filter, DSHOT
- [PX4 developer guide](https://docs.px4.io/) — EKF2, control allocator, INDI
- [Crazyflie firmware](https://github.com/bitcraze/crazyflie-firmware) — `src/modules/src/controller_pid.c`, `stabilizer.c`
