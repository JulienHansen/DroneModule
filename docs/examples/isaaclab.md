# IsaacLab Integration

This page shows how to connect every command level of the `drone` package
to an IsaacLab `DirectRLEnv`.  All five variants share the same
boilerplate (scene setup, observations, rewards, dones, robot reset) —
only `__init__` controller setup, `_pre_physics_step`, and the reset of
the controller change between them.

---

## Adapter pattern

Our controllers always output:

| Tensor | Shape | Unit |
|---|---|---|
| `thrust` | `[N, 1]` | N |
| `moment` | `[N, 3]` | N·m |

IsaacLab applies forces/torques per rigid body via:

```python
self._robot.set_external_force_and_torque(
    forces,   # [num_envs, num_bodies, 3]  — body frame by default
    torques,  # [num_envs, num_bodies, 3]
    body_ids=self._body_id,
)
```

The mapping is always:

```python
self._forces[:, 0, 2]  = thrust[:, 0]  # thrust along body-z [N]
self._torques[:, 0, :] = moment         # body moments [N·m]
```

`_apply_action` is therefore **identical for all five variants**:

```python
def _apply_action(self):
    self._robot.set_external_force_and_torque(
        self._forces, self._torques, body_ids=self._body_id
    )
```

---

## Root state assembly

All our controllers accept a `[N, 13]` state tensor:

```
[ pos(3) | quat(4) | lin_vel(3) | ang_vel(3) ]
```

From IsaacLab data:

```python
root_state = torch.cat([
    self._robot.data.root_pos_w,       # [N, 3]  world position
    self._robot.data.root_quat_w,      # [N, 4]  [w, x, y, z]
    self._robot.data.root_lin_vel_w,   # [N, 3]  world frame
    self._robot.data.root_ang_vel_b,   # [N, 3]  body frame
], dim=-1)
```

Pass `body_rates_in_body_frame=True` so the controller does not rotate
`ang_vel_b` again.

---

## Shared `__init__` boilerplate

The following setup is common to all variants.  The only line that
changes is the controller construction:

```python
from drone import load_config

CFG_PATH = "path/to/configs/crazyflie.yaml"

def __init__(self, cfg, render_mode=None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)

    # Shared buffers
    self._forces  = torch.zeros(self.num_envs, 1, 3, device=self.device)
    self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
    self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    self._body_id = self._robot.find_bodies("body")[0]

    # Controller is built from YAML config — see each variant below
    self._drone_cfg = load_config(CFG_PATH)
    self._ctrl = ...   # ← variant-specific
```

---

## Variant 1 — Full position cascade

**Action space:** `[x, y, z, yaw]`  target position in world frame.
**Controller:** `CrazyfliePIDController` at `command_level="position"`.
The full four-level cascade (position → velocity → attitude → rate) runs internally.

```python
# ── __init__ ────────────────────────────────────────────────────────────
from drone import CrazyfliePIDController

self._ctrl = CrazyfliePIDController.from_drone_config(
    self._drone_cfg,
    num_envs=self.num_envs,
    dt=self.cfg.sim.dt * self.cfg.decimation,   # effective controller dt
    device=self.device,
)

# ── _pre_physics_step ───────────────────────────────────────────────────
def _pre_physics_step(self, actions: torch.Tensor):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    # Map actions → absolute position setpoint [m]
    target_pos = self._terrain.env_origins.clone()
    target_pos[:, :2] += self._actions[:, :2] * 2.0   # ±2 m in XY
    target_pos[:,  2]  = 0.5 + (self._actions[:, 2] + 1.0) * 0.5  # 0.5–1.5 m

    root_state = torch.cat([
        self._robot.data.root_pos_w,
        self._robot.data.root_quat_w,
        self._robot.data.root_lin_vel_w,
        self._robot.data.root_ang_vel_b,
    ], dim=-1)

    thrust, moment = self._ctrl(
        root_state,
        target_pos=target_pos,
        target_yaw=self._actions[:, 3:4] * math.pi,   # ±π yaw target
        command_level="position",
        body_rates_in_body_frame=True,
    )

    self._forces[:, 0, 2]  = thrust[:, 0]
    self._torques[:, 0, :] = moment

# ── _reset_idx (controller reset only) ─────────────────────────────────
self._ctrl.reset(env_ids)
```

---

## Variant 2 — Velocity cascade

**Action space:** `[vx, vy, vz, yaw_rate]`  velocity setpoint in world frame.
**Controller:** `CrazyfliePIDController` at `command_level="velocity"`.
The position loop is bypassed; the agent controls velocity directly.

```python
# ── __init__ ────────────────────────────────────────────────────────────
MAX_VEL     = 2.0   # m/s
MAX_YAW_DOT = 90.0 * math.pi / 180.0   # rad/s

self._ctrl = CrazyfliePIDController.from_drone_config(
    self._drone_cfg,
    num_envs=self.num_envs,
    dt=self.cfg.sim.dt * self.cfg.decimation,
    device=self.device,
)

# ── _pre_physics_step ───────────────────────────────────────────────────
def _pre_physics_step(self, actions: torch.Tensor):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    target_vel      = self._actions[:, :3] * MAX_VEL        # [N, 3]  m/s
    target_yaw_rate = self._actions[:, 3:4] * MAX_YAW_DOT  # [N, 1]  rad/s

    root_state = torch.cat([
        self._robot.data.root_pos_w,
        self._robot.data.root_quat_w,
        self._robot.data.root_lin_vel_w,
        self._robot.data.root_ang_vel_b,
    ], dim=-1)

    thrust, moment = self._ctrl(
        root_state,
        target_vel=target_vel,
        target_yaw_rate=target_yaw_rate,
        command_level="velocity",
        body_rates_in_body_frame=True,
    )

    self._forces[:, 0, 2]  = thrust[:, 0]
    self._torques[:, 0, :] = moment

# ── _reset_idx ──────────────────────────────────────────────────────────
self._ctrl.reset(env_ids)
```

---

## Variant 3 — Attitude cascade

**Action space:** `[roll_ref, pitch_ref, yaw_rate_ref, thrust_normalized]`.
**Controller:** `CrazyfliePIDController` at `command_level="attitude"`.
Position and velocity PIDs are bypassed; the agent sets tilt and thrust directly.
This matches the environment provided as reference.

```python
# ── __init__ ────────────────────────────────────────────────────────────
from isaaclab.utils.math import euler_xyz_from_quat

MAX_TILT    = 30.0 * math.pi / 180.0
MAX_YAW_DOT = 90.0 * math.pi / 180.0

self._ctrl = CrazyfliePIDController.from_drone_config(
    self._drone_cfg,
    num_envs=self.num_envs,
    dt=self.cfg.sim.dt * self.cfg.decimation,
    device=self.device,
)

robot_mass   = self._robot.root_physx_view.get_masses()[0].sum()
hover_thrust = robot_mass * 9.81
self._min_thrust = 0.5 * hover_thrust
self._max_thrust = 1.8 * hover_thrust

# ── _pre_physics_step ───────────────────────────────────────────────────
def _pre_physics_step(self, actions: torch.Tensor):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    roll_ref  = self._actions[:, 0] * MAX_TILT
    pitch_ref = self._actions[:, 1] * MAX_TILT
    yaw_rate  = self._actions[:, 2] * MAX_YAW_DOT
    thrust_n  = (self._actions[:, 3] + 1.0) / 2.0  # [0, 1]
    thrust_N  = self._min_thrust + thrust_n * (self._max_thrust - self._min_thrust)

    _, _, meas_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
    target_att = torch.stack([roll_ref, pitch_ref, meas_yaw], dim=-1)

    root_state = torch.cat([
        self._robot.data.root_pos_w,
        self._robot.data.root_quat_w,
        self._robot.data.root_lin_vel_w,
        self._robot.data.root_ang_vel_b,
    ], dim=-1)

    # Controller handles only the moment; thrust is commanded directly.
    # Pass thrust in PWM units: thrust_cmd_scale = max_thrust / thrust_cmd_max
    thrust_pwm = thrust_N / self._ctrl.thrust_cmd_scale  # [N] → PWM

    thrust, moment = self._ctrl(
        root_state,
        target_attitude=target_att,
        thrust_cmd=thrust_pwm.unsqueeze(-1),
        target_yaw_rate=yaw_rate.unsqueeze(-1),
        command_level="attitude",
        body_rates_in_body_frame=True,
    )

    self._forces[:, 0, 2]  = thrust[:, 0]
    self._torques[:, 0, :] = moment

# ── _reset_idx ──────────────────────────────────────────────────────────
self._ctrl.reset(env_ids)
```

---

## Variant 4 — Body rate (Acro / Rate mode)

**Action space:** `[roll_rate, pitch_rate, yaw_rate, thrust_normalized]`.
**Controller:** `CrazyfliePIDController` at `command_level="body_rate"`,
with an optional **rate profile** shaping the stick input.
Only the innermost rate PID runs; all outer loops are bypassed.

```python
# ── __init__ ────────────────────────────────────────────────────────────
from drone import betaflight_rate_profile

MAX_RATE = 720.0 * math.pi / 180.0   # rad/s at full stick

self._ctrl = CrazyfliePIDController.from_drone_config(
    self._drone_cfg,
    num_envs=self.num_envs,
    dt=self.cfg.sim.dt * self.cfg.decimation,
    device=self.device,
)

robot_mass   = self._robot.root_physx_view.get_masses()[0].sum()
hover_thrust = robot_mass * 9.81
self._min_thrust = 0.3 * hover_thrust
self._max_thrust = 2.0 * hover_thrust

# ── _pre_physics_step ───────────────────────────────────────────────────
def _pre_physics_step(self, actions: torch.Tensor):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    # Pass stick through a rate profile (optional but realistic)
    omega_sp = betaflight_rate_profile(self._actions[:, :3])  # [N, 3] rad/s

    thrust_n = (self._actions[:, 3] + 1.0) / 2.0
    thrust_N = self._min_thrust + thrust_n * (self._max_thrust - self._min_thrust)
    thrust_pwm = thrust_N / self._ctrl.thrust_cmd_scale

    root_state = torch.cat([
        self._robot.data.root_pos_w,
        self._robot.data.root_quat_w,
        self._robot.data.root_lin_vel_w,
        self._robot.data.root_ang_vel_b,
    ], dim=-1)

    thrust, moment = self._ctrl(
        root_state,
        target_body_rates=omega_sp,
        thrust_cmd=thrust_pwm.unsqueeze(-1),
        command_level="body_rate",
        body_rates_in_body_frame=True,
    )

    self._forces[:, 0, 2]  = thrust[:, 0]
    self._torques[:, 0, :] = moment

# ── _reset_idx ──────────────────────────────────────────────────────────
self._ctrl.reset(env_ids)
```

---

## Variant 5 — Lee geometric controller

**Action space:** `[x, y, z, yaw]`  position setpoint.
**Controller:** `LeePositionController` — singularity-free SO(3) position + attitude control.
Stateless: no `reset()` needed (it is a no-op).

```python
# ── __init__ ────────────────────────────────────────────────────────────
from drone import LeePositionController

self._ctrl = LeePositionController.from_drone_config(
    self._drone_cfg,
    num_envs=self.num_envs,
    device=self.device,
)

# ── _pre_physics_step ───────────────────────────────────────────────────
def _pre_physics_step(self, actions: torch.Tensor):
    self._actions = actions.clone().clamp(-1.0, 1.0)

    target_pos = self._terrain.env_origins.clone()
    target_pos[:, :2] += self._actions[:, :2] * 2.0
    target_pos[:,  2]  = 0.5 + (self._actions[:, 2] + 1.0) * 0.5
    target_yaw = self._actions[:, 3:4] * math.pi   # ±π

    root_state = torch.cat([
        self._robot.data.root_pos_w,
        self._robot.data.root_quat_w,
        self._robot.data.root_lin_vel_w,
        self._robot.data.root_ang_vel_b,
    ], dim=-1)

    thrust, moment = self._ctrl(
        root_state,
        target_pos=target_pos,
        target_vel=None,
        target_acc=None,
        target_yaw=target_yaw,
        target_yaw_rate=None,
        body_rates_in_body_frame=True,
    )

    self._forces[:, 0, 2]  = thrust[:, 0]
    self._torques[:, 0, :] = moment

# ── _reset_idx ──────────────────────────────────────────────────────────
self._ctrl.reset(env_ids)   # no-op for Lee, but keeps code consistent
```

---

## Quick-reference table

| Variant | Action | Command level | Controller | Stateful |
|---|---|---|---|---|
| Position | `[x, y, z, yaw]` | `"position"` | `CrazyfliePIDController` | Yes |
| Velocity | `[vx, vy, vz, yaw_rate]` | `"velocity"` | `CrazyfliePIDController` | Yes |
| Attitude | `[roll, pitch, yaw_rate, T]` | `"attitude"` | `CrazyfliePIDController` | Yes |
| Body rate | `[p, q, r, T]` | `"body_rate"` | `CrazyfliePIDController` | Yes |
| Geometric | `[x, y, z, yaw]` | — | `LeePositionController` | No |

!!! tip "Choosing a command level for RL"
    - **Position**: easiest to train, most forgiving reward shaping.
    - **Velocity**: good balance — agent shapes trajectory, controller handles stability.
    - **Attitude**: requires the agent to understand drone physics; faster dynamics.
    - **Body rate**: closest to real hardware; hardest to train but most transferable to sim-to-real.
    - **Lee**: position-level action like Variant 1 but geometrically exact; better for aggressive manoeuvres.

!!! note "Controller timestep vs simulation timestep"
    The `dt` passed to `CrazyfliePIDController` should be the **effective
    control period**, not the physics dt:

    ```python
    dt = self.cfg.sim.dt * self.cfg.decimation
    ```

    The controller's multi-rate scheduling (`posvel_decimation`,
    `att_decimation`) then subdivides this correctly.

!!! note "Resetting integrators"
    Always call `self._ctrl.reset(env_ids)` in `_reset_idx` for
    `CrazyfliePIDController` — stale integrators from the previous episode
    cause a large spike on the first step of the new one.

    ```python
    def _reset_idx(self, env_ids):
        ...  # reset robot state as usual
        self._ctrl.reset(env_ids)
    ```
