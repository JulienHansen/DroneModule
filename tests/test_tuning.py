"""
Tests for drone_control.tuning.tune_from_physics.
"""

import math
import pytest
from drone_control.tuning import tune_from_physics, TuningResult

DEG2RAD = math.pi / 180.0
G = 9.81


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_result(**kwargs) -> TuningResult:
    defaults = dict(
        mass=0.027,
        inertia=[1.657e-5, 1.657e-5, 2.9e-5],
        bandwidth_rate=30.0,
        damping=0.7,
        max_thrust=0.638,
        thrust_cmd_max=65535.0,
        vel_thrust_scale=1000.0,
    )
    defaults.update(kwargs)
    return tune_from_physics(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_tuning_result(self):
        r = _default_result()
        assert isinstance(r, TuningResult)

    def test_gains_are_three_element_lists(self):
        r = _default_result()
        for attr in ("rate_kp", "rate_ki", "rate_kd",
                     "att_kp",  "att_ki",  "att_kd",
                     "vel_kp",  "vel_ki",  "vel_kd",
                     "pos_kp",  "pos_ki",  "pos_kd"):
            assert len(getattr(r, attr)) == 3, f"{attr} should have length 3"

    def test_to_params_keys(self):
        params = _default_result().to_params()
        expected = {"rate_kp", "rate_ki", "rate_kd",
                    "att_kp",  "att_ki",  "att_kd",
                    "vel_kp",  "vel_ki",  "vel_kd",
                    "pos_kp",  "pos_ki",  "pos_kd"}
        assert set(params.keys()) == expected

    def test_summary_is_string(self):
        assert isinstance(_default_result().summary(), str)

    def test_repr_equals_summary(self):
        r = _default_result()
        assert repr(r) == r.summary()


# ---------------------------------------------------------------------------
# Bandwidth cascade defaults
# ---------------------------------------------------------------------------

class TestBandwidthDefaults:
    def test_default_att_is_rate_over_5(self):
        r = _default_result(bandwidth_rate=30.0)
        assert r.bandwidth_att == pytest.approx(6.0)

    def test_default_vel_is_att_over_5(self):
        r = _default_result(bandwidth_rate=30.0)
        assert r.bandwidth_vel == pytest.approx(1.2)

    def test_default_pos_is_vel_over_5(self):
        r = _default_result(bandwidth_rate=30.0)
        assert r.bandwidth_pos == pytest.approx(0.24)

    def test_explicit_bandwidths_override_defaults(self):
        r = tune_from_physics(
            mass=0.027,
            inertia=[1.657e-5, 1.657e-5, 2.9e-5],
            bandwidth_rate=30.0,
            bandwidth_att=4.0,
            bandwidth_vel=0.5,
            bandwidth_pos=0.05,
            damping=0.7,
        )
        assert r.bandwidth_att == pytest.approx(4.0)
        assert r.bandwidth_vel == pytest.approx(0.5)
        assert r.bandwidth_pos == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Rate loop  (plant = 1/s, PI: s² + kp·s + ki = s² + 2ζω·s + ω²)
# ---------------------------------------------------------------------------

class TestRateLoop:
    def test_kp_formula(self):
        omega, zeta = 30.0, 0.7
        r = _default_result(bandwidth_rate=omega, damping=zeta)
        expected = 2.0 * zeta * omega
        assert r.rate_kp[0] == pytest.approx(expected, rel=1e-4)
        assert r.rate_kp[1] == pytest.approx(expected, rel=1e-4)

    def test_ki_formula(self):
        omega = 30.0
        r = _default_result(bandwidth_rate=omega)
        expected = omega ** 2
        assert r.rate_ki[0] == pytest.approx(expected, rel=1e-4)

    def test_kd_zero_by_default(self):
        r = _default_result()
        assert r.rate_kd == [0.0, 0.0, 0.0]

    def test_kd_with_derivative_time(self):
        omega, zeta, tau_d = 30.0, 0.7, 0.01
        r = _default_result(bandwidth_rate=omega, damping=zeta,
                            derivative_time=tau_d)
        kp = 2.0 * zeta * omega
        assert r.rate_kd[0] == pytest.approx(kp * tau_d, rel=1e-4)

    def test_roll_pitch_equal(self):
        r = _default_result()
        assert r.rate_kp[0] == r.rate_kp[1]
        assert r.rate_ki[0] == r.rate_ki[1]

    def test_all_gains_positive(self):
        r = _default_result()
        assert all(v >= 0 for v in r.rate_kp)
        assert all(v >= 0 for v in r.rate_ki)


# ---------------------------------------------------------------------------
# Attitude loop  (same structure as rate)
# ---------------------------------------------------------------------------

class TestAttitudeLoop:
    def test_kp_formula(self):
        omega_rate, zeta = 30.0, 0.7
        r = _default_result(bandwidth_rate=omega_rate, damping=zeta)
        omega_att = omega_rate / 5.0
        expected = 2.0 * zeta * omega_att
        assert r.att_kp[0] == pytest.approx(expected, rel=1e-4)

    def test_ki_formula(self):
        omega_rate = 30.0
        r = _default_result(bandwidth_rate=omega_rate)
        omega_att = omega_rate / 5.0
        assert r.att_ki[0] == pytest.approx(omega_att ** 2, rel=1e-4)

    def test_kd_always_zero(self):
        r = _default_result(derivative_time=0.1)
        assert r.att_kd == [0.0, 0.0, 0.0]

    def test_three_axes_identical(self):
        r = _default_result()
        assert r.att_kp[0] == r.att_kp[1] == r.att_kp[2]
        assert r.att_ki[0] == r.att_ki[1] == r.att_ki[2]


# ---------------------------------------------------------------------------
# Velocity loop x/y  (plant = g/s, YAML units = deg/(m/s))
# ---------------------------------------------------------------------------

class TestVelocityXYLoop:
    def test_kp_formula(self):
        omega_rate = 30.0
        r = _default_result(bandwidth_rate=omega_rate, damping=0.7)
        omega_vel = omega_rate / 25.0   # rate/5/5
        expected_deg = (2.0 * 0.7 * omega_vel / G) / DEG2RAD
        assert r.vel_kp[0] == pytest.approx(expected_deg, rel=1e-4)
        assert r.vel_kp[1] == pytest.approx(expected_deg, rel=1e-4)

    def test_ki_formula(self):
        omega_rate = 30.0
        r = _default_result(bandwidth_rate=omega_rate)
        omega_vel = omega_rate / 25.0
        expected_deg = (omega_vel ** 2 / G) / DEG2RAD
        assert r.vel_ki[0] == pytest.approx(expected_deg, rel=1e-4)

    def test_xy_equal(self):
        r = _default_result()
        assert r.vel_kp[0] == r.vel_kp[1]
        assert r.vel_ki[0] == r.vel_ki[1]

    def test_kd_zero(self):
        r = _default_result()
        assert r.vel_kd == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Velocity loop z  (scaled by K_z = max_thrust·vel_thrust_scale/(cmd_max·mass))
# ---------------------------------------------------------------------------

class TestVelocityZLoop:
    def _K_z(self, mass, max_thrust, thrust_cmd_max, vel_thrust_scale):
        return (max_thrust / thrust_cmd_max) * vel_thrust_scale / mass

    def test_kp_formula(self):
        mass, max_thrust, cmd_max, vts = 0.027, 0.638, 65535.0, 1000.0
        omega_rate = 30.0
        r = tune_from_physics(
            mass=mass, inertia=[1.657e-5]*3,
            bandwidth_rate=omega_rate, damping=0.7,
            max_thrust=max_thrust, thrust_cmd_max=cmd_max, vel_thrust_scale=vts,
        )
        omega_vel = omega_rate / 25.0
        K_z = self._K_z(mass, max_thrust, cmd_max, vts)
        expected = 2.0 * 0.7 * omega_vel / K_z
        assert r.vel_kp[2] == pytest.approx(expected, rel=1e-4)

    def test_ki_formula(self):
        mass, max_thrust, cmd_max, vts = 0.027, 0.638, 65535.0, 1000.0
        omega_rate = 30.0
        r = tune_from_physics(
            mass=mass, inertia=[1.657e-5]*3,
            bandwidth_rate=omega_rate, damping=0.7,
            max_thrust=max_thrust, thrust_cmd_max=cmd_max, vel_thrust_scale=vts,
        )
        omega_vel = omega_rate / 25.0
        K_z = self._K_z(mass, max_thrust, cmd_max, vts)
        expected = omega_vel ** 2 / K_z
        assert r.vel_ki[2] == pytest.approx(expected, rel=1e-4)

    def test_heavier_drone_lower_gains(self):
        """Heavier drone → smaller K_z → larger PWM gain."""
        light = tune_from_physics(mass=0.027, inertia=[1.657e-5]*3,
                                  bandwidth_rate=30.0)
        heavy = tune_from_physics(mass=0.200, inertia=[1.657e-5]*3,
                                  bandwidth_rate=30.0)
        assert heavy.vel_kp[2] > light.vel_kp[2]


# ---------------------------------------------------------------------------
# Position loop  (pure P, bandwidth = kp)
# ---------------------------------------------------------------------------

class TestPositionLoop:
    def test_kp_equals_bandwidth(self):
        omega_rate = 30.0
        r = _default_result(bandwidth_rate=omega_rate)
        omega_pos = omega_rate / 125.0   # rate/5/5/5
        assert r.pos_kp[0] == pytest.approx(omega_pos, rel=1e-4)

    def test_ki_kd_zero(self):
        r = _default_result()
        assert r.pos_ki == [0.0, 0.0, 0.0]
        assert r.pos_kd == [0.0, 0.0, 0.0]

    def test_explicit_pos_bandwidth(self):
        r = tune_from_physics(
            mass=0.027, inertia=[1.657e-5]*3,
            bandwidth_rate=30.0, bandwidth_pos=0.1,
        )
        assert r.pos_kp[0] == pytest.approx(0.1, rel=1e-4)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_no_warnings_for_good_separation(self):
        r = _default_result(bandwidth_rate=30.0)
        assert r.warnings == []

    def test_warns_on_poor_separation(self):
        # Force rate/att ratio of 2× (< 3× threshold)
        r = tune_from_physics(
            mass=0.027, inertia=[1.657e-5]*3,
            bandwidth_rate=10.0,
            bandwidth_att=6.0,  # ratio = 10/6 ≈ 1.67 → should warn
        )
        assert any("rate→att" in w for w in r.warnings)

    def test_warns_on_nyquist_violation(self):
        r = tune_from_physics(
            mass=0.027, inertia=[1.657e-5]*3,
            bandwidth_rate=200.0,  # very high, will exceed Nyquist at dt=0.002
            sim_dt=0.002,
        )
        assert len(r.warnings) > 0

    def test_no_nyquist_warning_when_dt_not_given(self):
        # bandwidth_rate=200.0 but no sim_dt → no Nyquist warning
        r = tune_from_physics(
            mass=0.027, inertia=[1.657e-5]*3,
            bandwidth_rate=200.0,
        )
        nyquist_warns = [w for w in r.warnings if "Nyquist" in w]
        assert nyquist_warns == []


# ---------------------------------------------------------------------------
# to_params() integration with CrazyfliePIDController
# ---------------------------------------------------------------------------

class TestToParamsIntegration:
    def test_params_accepted_by_controller(self):
        """to_params() output must be accepted without error by the controller."""
        from drone_control import load_config, CrazyfliePIDController
        import os

        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "crazyflie.yaml"
        )
        cfg = load_config(cfg_path)

        result = tune_from_physics(
            mass=cfg.physics.mass,
            inertia=[cfg.physics.inertia.ixx,
                     cfg.physics.inertia.iyy,
                     cfg.physics.inertia.izz],
            bandwidth_rate=30.0,
            damping=0.7,
            max_thrust=cfg.physics.max_thrust,
        )
        params = result.to_params()

        # Construct a controller with the tuned params — should not raise
        ctrl = CrazyfliePIDController(
            dt=0.002,
            num_envs=2,
            mass=cfg.physics.mass,
            params=params,
        )
        assert ctrl is not None


# ---------------------------------------------------------------------------
# Damping variations
# ---------------------------------------------------------------------------

class TestDamping:
    def test_higher_damping_higher_kp(self):
        """Higher ζ → higher kp (2ζω)."""
        low  = _default_result(damping=0.5)
        high = _default_result(damping=1.0)
        assert high.rate_kp[0] > low.rate_kp[0]

    def test_ki_independent_of_damping(self):
        """ki = ω² is independent of ζ."""
        low  = _default_result(damping=0.5)
        high = _default_result(damping=1.0)
        assert low.rate_ki[0] == pytest.approx(high.rate_ki[0], rel=1e-6)
