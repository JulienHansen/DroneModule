"""
Tests for CrazyfliePIDController.

Covers:
- Instantiation (direct and from_drone_config)
- All four command levels
- Output shapes and SI units
- Decimation: pos/vel loop only runs every N steps
- Yaw setpoint: absolute, rate, max-delta clamping
- Reset (full and partial)
- Derivative-on-measurement: no kick on setpoint step
- Per-env rate-gain override
"""

from __future__ import annotations

import math
import pytest
import torch

from drone_control import load_config, CrazyfliePIDController

N  = 4
DT = 0.002          # 500 Hz simulation
G  = 9.81


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_root_state(
    N: int,
    pos: float = 0.0,
    yaw: float = 0.0,
) -> torch.Tensor:
    """Build a [N, 13] root_state = [pos(3), quat(4), lin_vel(3), ang_vel(3)].

    Quaternion [w, x, y, z] with pure yaw rotation.
    """
    half = yaw / 2.0
    q = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    root = torch.zeros(N, 13)
    root[:, :3]  = pos
    root[:, 3:7] = q.unsqueeze(0).expand(N, -1)
    return root


def make_ctrl(cf_config=None, params: dict | None = None) -> CrazyfliePIDController:
    p = params or {}
    if cf_config is not None:
        return CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
    return CrazyfliePIDController(dt=DT, num_envs=N, device="cpu", params=p)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_direct_default_params(self):
        ctrl = CrazyfliePIDController(dt=DT, num_envs=N, device="cpu")
        assert ctrl.dt == pytest.approx(DT)
        assert ctrl._num_envs == N

    def test_from_drone_config(self, cf_config):
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        assert ctrl.mass.item() == pytest.approx(cf_config.physics.mass)
        # inertia tensor should be set
        assert ctrl._inertia_tensor is not None
        assert ctrl._inertia_tensor.shape == (N, 3, 3)

    def test_thrust_cmd_scale_derived_from_max_thrust(self, cf_config):
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        expected = cf_config.physics.max_thrust / 65_535.0
        assert ctrl.thrust_cmd_scale == pytest.approx(expected, rel=1e-4)

    def test_yaml_params_loaded(self, cf_config):
        """YAML crazyflie_pid section should override default gains."""
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        # rate kp from YAML = [250, 250, 120]  (matches CF2 firmware platform_defaults_cf2.h)
        assert ctrl.rate_kp[0].item() == pytest.approx(250.0)
        assert ctrl.rate_kp[2].item() == pytest.approx(120.0)

    def test_decimation_set_correctly(self, cf_config):
        """At 500 Hz sim / 100 Hz posvel → decimation = 5."""
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        # sim_rate = 1/0.002 = 500, posvel = 100 → decimation = 5
        assert ctrl.posvel_decimation == 5
        # att = 500 → decimation = 1
        assert ctrl.att_decimation == 1


# ---------------------------------------------------------------------------
# Output shapes & SI units
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.fixture(autouse=True)
    def ctrl(self, cf_config):
        self.ctrl = make_ctrl(cf_config)

    def _call(self, level, **kw):
        rs = make_root_state(N)
        return self.ctrl(rs, command_level=level, **kw)

    def test_position_shapes(self):
        thrust, moment = self._call("position", target_pos=torch.zeros(N, 3))
        assert thrust.shape == (N, 1)
        assert moment.shape == (N, 3)

    def test_velocity_shapes(self):
        thrust, moment = self._call("velocity", target_vel=torch.zeros(N, 3))
        assert thrust.shape == (N, 1)
        assert moment.shape == (N, 3)

    def test_attitude_shapes(self):
        thrust, moment = self._call(
            "attitude",
            target_attitude=torch.zeros(N, 3),
            thrust_cmd=torch.full((N, 1), 30_000.0),
        )
        assert thrust.shape == (N, 1)
        assert moment.shape == (N, 3)

    def test_body_rate_shapes(self):
        thrust, moment = self._call(
            "body_rate",
            target_body_rates=torch.zeros(N, 3),
            thrust_cmd=torch.full((N, 1), 30_000.0),
        )
        assert thrust.shape == (N, 1)
        assert moment.shape == (N, 3)

    def test_thrust_in_newtons(self, cf_config):
        """At hover with zero error, thrust should be close to m·g."""
        rs    = make_root_state(N)
        tgt   = torch.zeros(N, 3)
        # Run several steps so pos/vel loops have settled
        for _ in range(20):
            thrust, _ = self.ctrl(rs, target_pos=tgt, command_level="position")
        m_g = cf_config.physics.mass * G
        # Not exact (controller transient) but should be the right order of magnitude
        assert thrust.mean().item() == pytest.approx(m_g, rel=0.5)

    def test_unknown_command_level_raises(self):
        with pytest.raises(ValueError):
            self._call("teleport")


# ---------------------------------------------------------------------------
# Loop decimation
# ---------------------------------------------------------------------------

class TestDecimation:
    def test_posvel_loop_runs_only_at_decimation(self, cf_config):
        """Position loop: vel_sp should change only on steps 0, 5, 10, ...

        We use a small z-error (0.1 m) with ki_z = 0.5 so that each time the
        pos loop fires its integrator grows, making vel_sp_z measurably different.
        """
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        assert ctrl.posvel_decimation == 5
        assert ctrl.pos_pid.ki[2].item() == pytest.approx(0.5), \
            "ki_z must be non-zero for this test to be meaningful"

        rs  = make_root_state(N)
        # Small z-only reference so output is NOT saturated
        ref = torch.zeros(N, 3); ref[:, 2] = 0.1

        ctrl(rs, target_pos=ref, command_level="position")   # step 0 → pos loop runs
        vel_sp_after_0 = ctrl._vel_sp.clone()

        ctrl(rs, target_pos=ref, command_level="position")   # step 1 → pos loop frozen
        vel_sp_after_1 = ctrl._vel_sp.clone()

        # vel_sp must not change on a non-decimated step
        assert torch.allclose(vel_sp_after_0, vel_sp_after_1), \
            "vel_sp changed on a non-decimated step"

        for _ in range(3):                                    # steps 2, 3, 4
            ctrl(rs, target_pos=ref, command_level="position")

        ctrl(rs, target_pos=ref, command_level="position")   # step 5 → pos loop fires again
        vel_sp_after_5 = ctrl._vel_sp.clone()

        # The integrator will have accumulated one more dt*error so vel_sp_z must differ
        assert not torch.allclose(vel_sp_after_0, vel_sp_after_5, atol=1e-7), \
            "vel_sp did not update at the next decimation boundary"


# ---------------------------------------------------------------------------
# Yaw setpoint
# ---------------------------------------------------------------------------

class TestYawSetpoint:
    @pytest.fixture(autouse=True)
    def ctrl(self, cf_config):
        self.ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")

    def test_absolute_yaw_target(self):
        rs = make_root_state(N, yaw=0.0)
        tgt_yaw = torch.full((N, 1), math.pi / 4)
        self.ctrl(rs, target_yaw=tgt_yaw, command_level="position")
        assert torch.allclose(
            self.ctrl._yaw_sp,
            tgt_yaw,
            atol=1e-5,
        ), f"Yaw setpoint not set: {self.ctrl._yaw_sp}"

    def test_yaw_rate_integration(self):
        rs = make_root_state(N, yaw=0.0)
        rate = torch.full((N, 1), 1.0)          # 1 rad/s
        self.ctrl(rs, target_yaw_rate=rate, command_level="position")
        expected = 1.0 * DT
        assert self.ctrl._yaw_sp.mean().item() == pytest.approx(expected, abs=1e-4)

    def test_yaw_wraps_at_pi(self):
        rs = make_root_state(N, yaw=math.pi - 0.01)
        # Command a small positive rate that pushes past +π
        rate = torch.full((N, 1), 100.0)
        self.ctrl(rs, target_yaw_rate=rate, command_level="position")
        yaw = self.ctrl._yaw_sp.squeeze(-1)
        assert torch.all(yaw >= -math.pi - 1e-4) and torch.all(yaw <= math.pi + 1e-4), \
            f"Yaw not wrapped: {yaw}"

    def test_yaw_max_delta_clamps_setpoint(self):
        ctrl = CrazyfliePIDController(
            dt=DT, num_envs=N, device="cpu",
            params={"yaw_max_delta": 0.1},          # 0.1 rad max delta
        )
        rs = make_root_state(N, yaw=0.0)
        big_yaw = torch.full((N, 1), 2.0)           # 2 rad target
        ctrl(rs, target_yaw=big_yaw, command_level="position")
        delta = (ctrl._yaw_sp - 0.0).abs()
        assert torch.all(delta <= 0.1 + 1e-5), f"yaw_max_delta not respected: {delta}"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    @pytest.fixture(autouse=True)
    def ctrl(self, cf_config):
        self.ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")

    def _warm_up(self, steps=10):
        rs  = make_root_state(N)
        ref = torch.ones(N, 3)
        for _ in range(steps):
            self.ctrl(rs, target_pos=ref, command_level="position")

    def test_full_reset_clears_integrators(self):
        self._warm_up()
        self.ctrl.reset()
        for pid in (self.ctrl.pos_pid, self.ctrl.vel_pid, self.ctrl.att_pid):
            if pid.integrator is not None:
                assert torch.all(pid.integrator == 0.0)
        assert self.ctrl._rate_integral is None

    def test_full_reset_clears_buffers(self):
        self._warm_up()
        self.ctrl.reset()
        assert self.ctrl._vel_sp is None
        assert self.ctrl._rate_sp is None
        assert self.ctrl._step_count == 0

    def test_partial_reset_leaves_other_envs(self):
        self._warm_up(steps=20)
        pre_integral = self.ctrl.pos_pid.integrator.clone()
        self.ctrl.reset(env_ids=torch.tensor([0, 1]))
        assert torch.all(self.ctrl.pos_pid.integrator[0] == 0.0), "env 0 not reset"
        assert torch.all(self.ctrl.pos_pid.integrator[1] == 0.0), "env 1 not reset"
        assert torch.allclose(self.ctrl.pos_pid.integrator[2], pre_integral[2]), "env 2 was reset"


# ---------------------------------------------------------------------------
# Derivative on measurement — no kick on setpoint step
# ---------------------------------------------------------------------------

class TestDerivativeOnMeasurement:
    def test_no_derivative_kick_on_rate_setpoint_step(self, cf_config):
        """
        Derivative is computed on *measurement*, not *error*.
        A sudden change in rate setpoint should NOT cause a derivative spike.
        """
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=1, dt=DT, device="cpu")
        rs   = make_root_state(1)

        # Step 1: zero rate setpoint
        _, moment_before = ctrl(rs, target_body_rates=torch.zeros(1, 3),
                                thrust_cmd=torch.full((1, 1), 30_000.0),
                                command_level="body_rate")

        # Step 2: sudden large rate setpoint change
        _, moment_after = ctrl(rs, target_body_rates=torch.full((1, 3), 5.0),
                               thrust_cmd=torch.full((1, 1), 30_000.0),
                               command_level="body_rate")

        # The moment difference should come only from kp*Δerror, NOT from a D-kick.
        # With derivative on measurement (measurement = 0 both steps), ṁeas_dot ≈ 0
        # so the D term ≈ 0.  The change in moment ≈ kp * rate_error.
        expected_delta = ctrl.rate_kp * 5.0         # kp * Δsetpoint, measurement unchanged
        actual_delta   = (moment_after - moment_before).abs()

        # Allow 10 % tolerance for the inertia scaling
        assert torch.all(actual_delta < expected_delta.abs() * 1.1 + 1e-6), \
            f"Derivative kick detected: {actual_delta} > {expected_delta * 1.1}"


# ---------------------------------------------------------------------------
# Per-env rate gain update
# ---------------------------------------------------------------------------

class TestRateGainUpdate:
    def test_set_rate_gains_all_envs(self, cf_config):
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        new_kp = torch.tensor([100.0, 100.0, 50.0])
        ctrl.set_rate_gains(rate_kp=new_kp)
        assert torch.allclose(ctrl.rate_kp, new_kp)

    def test_set_rate_gains_subset(self, cf_config):
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        original_kp = ctrl.rate_kp.clone()

        new_kp  = torch.tensor([999.0, 999.0, 999.0])
        env_ids = torch.tensor([0, 2])
        ctrl.set_rate_gains(rate_kp=new_kp, env_ids=env_ids)

        assert torch.allclose(ctrl.rate_kp[0], new_kp), "env 0 not updated"
        assert torch.allclose(ctrl.rate_kp[2], new_kp), "env 2 not updated"
        assert torch.allclose(ctrl.rate_kp[1], original_kp), "env 1 was incorrectly modified"
        assert torch.allclose(ctrl.rate_kp[3], original_kp), "env 3 was incorrectly modified"


# ---------------------------------------------------------------------------
# No NaN / Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_over_many_steps(self, cf_config):
        ctrl = CrazyfliePIDController.from_drone_config(cf_config, num_envs=N, dt=DT, device="cpu")
        rs   = make_root_state(N)
        ref  = torch.tensor([[1.0, 0.5, 1.5]] * N)
        for _ in range(500):
            thrust, moment = ctrl(rs, target_pos=ref, command_level="position")
        assert torch.all(torch.isfinite(thrust)), f"Non-finite thrust: {thrust}"
        assert torch.all(torch.isfinite(moment)), f"Non-finite moment: {moment}"
