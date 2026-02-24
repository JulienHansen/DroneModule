"""
Tests for the YAML configuration loader.

Checks that both provided configs load without error, that field values
match what is written in the YAML files, and that load_config() accepts
both str and Path arguments.
"""

import math
from pathlib import Path

import pytest

from drone_control.config.loader import (
    load_config,
    DroneConfig,
    DronePhysicsConfig,
    InertiaConfig,
    AttitudeControllerConfig,
    PositionControllerConfig,
    PIDConfig,
)

REPO_ROOT   = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_crazyflie_no_error(self):
        cfg = load_config(CONFIGS_DIR / "crazyflie.yaml")
        assert isinstance(cfg, DroneConfig)

    def test_load_generic_quad_no_error(self):
        cfg = load_config(CONFIGS_DIR / "generic_quad_250mm.yaml")
        assert isinstance(cfg, DroneConfig)

    def test_accepts_string_path(self):
        cfg = load_config(str(CONFIGS_DIR / "crazyflie.yaml"))
        assert isinstance(cfg, DroneConfig)

    def test_returns_correct_types(self):
        cfg = load_config(CONFIGS_DIR / "crazyflie.yaml")
        assert isinstance(cfg.physics,  DronePhysicsConfig)
        assert isinstance(cfg.attitude, AttitudeControllerConfig)
        assert isinstance(cfg.position, PositionControllerConfig)
        assert isinstance(cfg.physics.inertia, InertiaConfig)
        assert isinstance(cfg.attitude.rate.roll, PIDConfig)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Crazyflie values
# ---------------------------------------------------------------------------

class TestCrazyflieValues:
    @pytest.fixture(autouse=True)
    def cfg(self):
        self.cfg = load_config(CONFIGS_DIR / "crazyflie.yaml")

    def test_name(self):
        assert self.cfg.physics.name == "Crazyflie 2.1"

    def test_mass(self):
        assert self.cfg.physics.mass == pytest.approx(0.027)

    def test_inertia(self):
        assert self.cfg.physics.inertia.ixx == pytest.approx(1.657e-5)
        assert self.cfg.physics.inertia.iyy == pytest.approx(1.657e-5)
        assert self.cfg.physics.inertia.izz == pytest.approx(2.900e-5)

    def test_max_thrust(self):
        assert self.cfg.physics.max_thrust == pytest.approx(0.638)

    def test_rate_loop_gains(self):
        rate = self.cfg.attitude.rate
        assert rate.roll.kp  == pytest.approx(50.0)
        assert rate.pitch.kp == pytest.approx(50.0)
        assert rate.yaw.kp   == pytest.approx(50.0)
        # All rate ki/kd are 0 for Crazyflie
        assert rate.roll.ki  == pytest.approx(0.0)
        assert rate.roll.kd  == pytest.approx(0.0)

    def test_rate_limit_equals_8_revs_per_s(self):
        """Limit should be 8 × 2π rad/s²."""
        expected = 8.0 * 2.0 * math.pi
        assert self.cfg.attitude.rate.roll.limit == pytest.approx(expected, rel=1e-3)

    def test_angle_loop_gains(self):
        angle = self.cfg.attitude.angle
        assert angle.roll.kp  == pytest.approx(4.0)
        assert angle.pitch.kp == pytest.approx(4.0)
        assert angle.yaw.kp   == pytest.approx(3.0)

    def test_angle_limit_equals_2p5_revs_per_s(self):
        """Limit should be 2.5 × 2π rad/s."""
        expected = 2.5 * 2.0 * math.pi
        assert self.cfg.attitude.angle.roll.limit == pytest.approx(expected, rel=1e-3)

    def test_position_freqs(self):
        assert self.cfg.position.freq_vel_hz == pytest.approx(100.0)
        assert self.cfg.position.freq_pos_hz == pytest.approx(100.0)

    def test_max_thrust_scale(self):
        assert self.cfg.position.max_thrust_scale == pytest.approx(0.8)

    def test_max_horizontal_angle(self):
        assert self.cfg.position.max_horizontal_angle_deg == pytest.approx(30.0)

    def test_position_gains(self):
        p = self.cfg.position.position
        assert p.x.kp == pytest.approx(5.0)
        assert p.z.kd == pytest.approx(3.5)

    def test_velocity_gains(self):
        v = self.cfg.position.velocity
        assert v.vx.kp == pytest.approx(1.0)
        assert v.vy.kp == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Generic quad values
# ---------------------------------------------------------------------------

class TestGenericQuadValues:
    @pytest.fixture(autouse=True)
    def cfg(self):
        self.cfg = load_config(CONFIGS_DIR / "generic_quad_250mm.yaml")

    def test_mass_heavier_than_crazyflie(self):
        assert self.cfg.physics.mass > 0.027

    def test_max_thrust_larger_than_crazyflie(self):
        assert self.cfg.physics.max_thrust > 0.638

    def test_rate_loop_has_integral(self):
        """The generic quad uses a non-zero ki on the rate loop."""
        assert self.cfg.attitude.rate.roll.ki > 0.0

    def test_rate_loop_has_derivative(self):
        assert self.cfg.attitude.rate.roll.kd > 0.0

    def test_higher_max_angle(self):
        """The generic quad allows a more aggressive lean angle."""
        assert self.cfg.position.max_horizontal_angle_deg > 30.0
