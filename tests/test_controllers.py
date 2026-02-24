"""
Unit tests for AttController_Vectorized and PosController_Vectorized.

Tests cover:
- Correct cascade PID output values
- Yaw angle wrap-around handling
- Directional saturation of roll/pitch in the velocity loop
- Thrust clamping (upper and lower bounds)
- Reset mechanics
"""

import math
import pytest
import torch

from drone_control import load_config, AttController_Vectorized, PosController_Vectorized

N = 3  # number of parallel environments
G = 9.81


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def att(cf_config):
    return AttController_Vectorized.from_drone_config(
        cf_config, num_envs=N, device=torch.device("cpu")
    )


@pytest.fixture
def pos(cf_config):
    return PosController_Vectorized.from_drone_config(
        cf_config, num_envs=N, device=torch.device("cpu")
    )


@pytest.fixture
def J(cf_config):
    """Diagonal inertia tensor [N, 3, 3] from Crazyflie config."""
    ixx = cf_config.physics.inertia.ixx
    iyy = cf_config.physics.inertia.iyy
    izz = cf_config.physics.inertia.izz
    J_single = torch.diag(torch.tensor([ixx, iyy, izz], dtype=torch.float))
    return J_single.unsqueeze(0).expand(N, -1, -1)


@pytest.fixture
def mass(cf_config):
    return torch.full((N,), cf_config.physics.mass)


# ---------------------------------------------------------------------------
# AttController — angle loop
# ---------------------------------------------------------------------------

class TestAttAngleLoop:
    def test_p_only_first_step(self, att):
        """At the first step (integrator=0, no previous error), output = kp * error."""
        ref  = torch.zeros(N, 3)
        meas = torch.zeros(N, 3)
        ref[:, 0] = 0.2   # roll reference
        out = att.run_angle(ref, meas)
        kp = att.pid_roll.kp.item()
        assert torch.allclose(out[:, 0], torch.full((N,), kp * 0.2), atol=1e-5)

    def test_pitch_independent_of_roll(self, att):
        """Pitch output should only respond to pitch error."""
        ref  = torch.zeros(N, 3)
        meas = torch.zeros(N, 3)
        ref[:, 0] = 1.0   # roll only
        out = att.run_angle(ref, meas)
        assert torch.allclose(out[:, 1], torch.zeros(N), atol=1e-6), \
            "Pitch output should be 0 when pitch error is 0"

    def test_yaw_wrap_positive(self, att):
        """Error > +π must be wrapped to the equivalent negative angle."""
        ref  = torch.zeros(N, 3);  ref[0, 2]  =  3.5
        meas = torch.zeros(N, 3);  meas[0, 2] = -3.0
        # raw error = 6.5 > π → corrected = 6.5 - 2π ≈ 0.217... but check sign
        out = att.run_angle(ref, meas)
        raw_err   = 3.5 - (-3.0)                  # 6.5
        corrected = raw_err - 2.0 * math.pi        # ≈ 0.217 (but < π now)
        # Since 6.5 > π the code does: err = -(2π - err) = err - 2π
        expected_omega = att.pid_yaw.kp.item() * corrected
        assert torch.isclose(out[0, 2], torch.tensor(expected_omega, dtype=torch.float), atol=1e-4)

    def test_yaw_wrap_negative(self, att):
        """Error < -π must be wrapped to the equivalent positive angle."""
        ref  = torch.zeros(N, 3);  ref[0, 2]  = -3.5
        meas = torch.zeros(N, 3);  meas[0, 2] =  3.0
        raw_err   = -3.5 - 3.0                    # -6.5
        corrected = raw_err + 2.0 * math.pi        # ≈ -0.217
        out = att.run_angle(ref, meas)
        expected_omega = att.pid_yaw.kp.item() * corrected
        assert torch.isclose(out[0, 2], torch.tensor(expected_omega, dtype=torch.float), atol=1e-4)

    def test_output_clamped_to_limit(self, att):
        """Output must not exceed the angle-loop saturation limit."""
        ref  = torch.full((N, 3), 100.0)  # huge reference
        meas = torch.zeros(N, 3)
        out  = att.run_angle(ref, meas)
        limit = att.pid_roll.limit_up.item()
        assert torch.all(out.abs() <= limit + 1e-6)


# ---------------------------------------------------------------------------
# AttController — rate loop
# ---------------------------------------------------------------------------

class TestAttRateLoop:
    def test_torque_without_gyroscopic_term(self, att, J):
        """With meas_omega=0, τ = J · α_ref = J · (kp · error)."""
        ref_omega  = torch.ones(N, 3)
        meas_omega = torch.zeros(N, 3)
        tau = att.run_rate(ref_omega, meas_omega, J)

        kp_roll = att.pid_rollrate.kp.item()
        # alpha_ref_roll = kp * 1.0, tau_roll = Ixx * alpha_ref_roll
        Ixx = J[0, 0, 0].item()
        expected_tau_roll = Ixx * kp_roll
        assert torch.isclose(tau[0, 0], torch.tensor(expected_tau_roll), atol=1e-7)

    def test_gyroscopic_term_adds_to_torque(self, att, J):
        """With non-zero omega, the ω × (J ω) term changes the torque."""
        ref_omega  = torch.zeros(N, 3)
        meas_omega = torch.zeros(N, 3)
        meas_omega[:, 0] = 1.0   # roll rate only

        tau_with_omega    = att.run_rate(ref_omega, meas_omega, J)

        att.reset()
        meas_omega_zero = torch.zeros(N, 3)
        tau_without_omega = att.run_rate(ref_omega, meas_omega_zero, J)

        # Gyroscopic coupling means torques differ when omega != 0
        # (for a diagonal inertia with only roll rate, pitch/yaw see the cross term)
        assert not torch.allclose(tau_with_omega, tau_without_omega, atol=1e-8), \
            "Gyroscopic term did not affect the torque output"

    def test_output_shape(self, att, J):
        tau = att.run_rate(torch.zeros(N, 3), torch.zeros(N, 3), J)
        assert tau.shape == (N, 3)


# ---------------------------------------------------------------------------
# AttController — reset
# ---------------------------------------------------------------------------

class TestAttReset:
    def test_full_reset_clears_all_pids(self, att):
        ref  = torch.full((N, 3), 0.5)
        meas = torch.zeros(N, 3)
        for _ in range(10):
            omega_ref = att.run_angle(ref, meas)
        att.reset()
        for pid_name in ("pid_roll", "pid_pitch", "pid_yaw",
                         "pid_rollrate", "pid_pitchrate", "pid_yawrate"):
            pid = getattr(att, pid_name)
            assert torch.all(pid.integrator == 0.0), f"{pid_name}.integrator not reset"
            assert torch.all(pid.error_d1   == 0.0), f"{pid_name}.error_d1 not reset"

    def test_partial_reset(self, att):
        ref  = torch.full((N, 3), 0.5)
        meas = torch.zeros(N, 3)
        for _ in range(10):
            att.run_angle(ref, meas)
        pre = att.pid_roll.error_d1.clone()
        att.reset(env_ids=torch.tensor([0]))
        assert att.pid_roll.error_d1[0] == 0.0,      "env 0 should be reset"
        assert att.pid_roll.error_d1[1] == pre[1],    "env 1 should be untouched"


# ---------------------------------------------------------------------------
# PosController — position loop
# ---------------------------------------------------------------------------

class TestPosLoop:
    def test_p_only_first_step(self, pos):
        ref  = torch.zeros(N, 3);  ref[:, 0] = 2.0
        meas = torch.zeros(N, 3)
        out  = pos.run_pos(ref, meas)
        kp_x    = pos.pid_x.kp.item()
        limit_x = pos.pid_x.limit_up.item()
        expected = min(kp_x * 2.0, limit_x)
        assert torch.allclose(out[:, 0], torch.full((N,), expected), atol=1e-5)

    def test_z_saturates_at_limit(self, pos):
        ref  = torch.zeros(N, 3);  ref[:, 2] = 1000.0
        meas = torch.zeros(N, 3)
        out  = pos.run_pos(ref, meas)
        limit_z = pos.pid_z.limit_up.item()
        assert torch.allclose(out[:, 2], torch.full((N,), limit_z), atol=1e-5)

    def test_axes_independent(self, pos):
        ref  = torch.zeros(N, 3);  ref[:, 0] = 1.0
        meas = torch.zeros(N, 3)
        out  = pos.run_pos(ref, meas)
        assert torch.allclose(out[:, 1], torch.zeros(N), atol=1e-6), "y should be 0"
        assert torch.allclose(out[:, 2], torch.zeros(N), atol=1e-6), "z should be 0"


# ---------------------------------------------------------------------------
# PosController — velocity loop
# ---------------------------------------------------------------------------

class TestVelLoop:
    def test_directional_saturation_magnitude(self, pos, mass):
        """Large horizontal velocity error: |rp_ref| ≤ max_horizontal_angle."""
        ref_ve  = torch.full((N, 3), 50.0)
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        rp_ref, _ = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        mag = torch.linalg.norm(rp_ref, dim=-1)
        assert torch.all(mag <= pos.max_angle_v + 1e-5), \
            f"Roll/pitch magnitude exceeds limit: {mag}"

    def test_directional_saturation_preserves_direction(self, pos, mass):
        """Clamping must only scale magnitude, not rotate the direction."""
        ref_ve  = torch.zeros(N, 3)
        ref_ve[:, 0] = 50.0  # pure x demand → should produce pure pitch
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        rp_ref, _ = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        # For yaw=0 and pure vx demand, pitch should dominate
        assert torch.all(rp_ref[:, 1].abs() > rp_ref[:, 0].abs()), \
            "Direction not preserved after saturation"

    def test_thrust_upper_bound(self, pos, mass):
        """Extreme upward demand: thrust ≤ max_thrust_scale × drone.max_thrust."""
        ref_ve  = torch.zeros(N, 3);  ref_ve[:, 2] = 100.0
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        _, thrust = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        assert torch.all(thrust <= pos.max_thrust + 1e-5), \
            f"Thrust exceeded max: {thrust}"

    def test_thrust_lower_bound(self, pos, mass):
        """Extreme downward demand: thrust ≥ 0.8 × g × mass (hover floor)."""
        ref_ve  = torch.zeros(N, 3);  ref_ve[:, 2] = -100.0
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        _, thrust = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        min_thrust = 0.8 * G * mass
        assert torch.all(thrust >= min_thrust - 1e-5), \
            f"Thrust below hover floor: {thrust}"

    def test_yaw_rotation_maps_vx_to_pitch(self, pos, mass):
        """With yaw=0, vx demand should produce predominantly pitch (axis 1)."""
        ref_ve  = torch.zeros(N, 3);  ref_ve[:, 0] = 1.0
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        rp_ref, _ = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        # For yaw=0: R = [[sin(0), -cos(0)], [cos(0), sin(0)]] = [[0,-1],[1,0]]
        # T1 = kp * 1.0, T12 rotated → pitch = T1/g, roll = 0
        assert torch.all(rp_ref[:, 1].abs() > 1e-6), "vx should drive pitch"
        assert torch.allclose(rp_ref[:, 0], torch.zeros(N), atol=1e-5), "vx should not drive roll"

    def test_output_shapes(self, pos, mass):
        ref_ve  = torch.zeros(N, 3)
        meas_ve = torch.zeros(N, 3)
        yaw     = torch.zeros(N)
        rp_ref, thrust = pos.run_vel(ref_ve, meas_ve, yaw, mass)
        assert rp_ref.shape  == (N, 2)
        assert thrust.shape  == (N,)


# ---------------------------------------------------------------------------
# PosController — reset
# ---------------------------------------------------------------------------

class TestPosReset:
    def test_full_reset(self, pos):
        ref  = torch.full((N, 3), 5.0)
        meas = torch.zeros(N, 3)
        for _ in range(10):
            pos.run_pos(ref, meas)
        pos.reset()
        for pid_name in ("pid_x", "pid_y", "pid_z", "pid_vx", "pid_vy", "pid_vz"):
            pid = getattr(pos, pid_name)
            assert torch.all(pid.error_d1 == 0.0), f"{pid_name}.error_d1 not reset"
