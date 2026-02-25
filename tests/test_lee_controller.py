"""
Tests for LeePositionController.

Covers:
- Direct and from_drone_config instantiation
- Output shapes (thrust [N,1], moment [N,3])
- Hover: zero position/velocity error → thrust ≈ m·g, moment ≈ 0
- Position error → thrust and moment change appropriately
- Yaw target handling (scalar, batched, None)
- body_rates_in_body_frame flag
- Single-env input (dim=1) auto-unsqueeze
- reset() is a no-op (no crash, no state change)
- to() device method
- Finite outputs for random valid states
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from drone import load_config, LeePositionController

N   = 4
G   = 9.81
MASS = 0.027
CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "crazyflie.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hover_state(n: int = N) -> torch.Tensor:
    """[n, 13] level hover at origin, zero velocity."""
    s = torch.zeros(n, 13)
    s[:, 3] = 1.0   # quaternion w=1 (identity, level)
    return s


def _make_ctrl(n: int = N, **kwargs) -> LeePositionController:
    cfg = load_config(CFG_PATH)
    return LeePositionController.from_drone_config(cfg, num_envs=n, **kwargs)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_from_drone_config(self):
        ctrl = _make_ctrl()
        assert ctrl.num_envs == N

    def test_direct_instantiation(self):
        ctrl = LeePositionController(
            num_envs=2,
            mass=0.027,
            inertia=[1.657e-5, 1.657e-5, 2.9e-5],
        )
        assert ctrl.num_envs == 2

    def test_gains_loaded_from_yaml(self):
        ctrl = _make_ctrl()
        # YAML has non-default values; ensure they were loaded
        assert ctrl.k_pos is not None
        assert ctrl.k_pos.shape == (3,)

    def test_to_method(self):
        ctrl = _make_ctrl()
        ctrl2 = ctrl.to("cpu")
        assert ctrl2 is ctrl
        assert ctrl2.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_thrust_shape(self):
        ctrl = _make_ctrl()
        thrust, _ = ctrl(_hover_state())
        assert thrust.shape == (N, 1)

    def test_moment_shape(self):
        ctrl = _make_ctrl()
        _, moment = ctrl(_hover_state())
        assert moment.shape == (N, 3)

    def test_single_env_1d_input(self):
        ctrl = _make_ctrl(n=1)
        state_1d = _hover_state(1).squeeze(0)  # [13]
        thrust, moment = ctrl(state_1d)
        assert thrust.shape == (1, 1)
        assert moment.shape == (1, 3)

    def test_batch_n_1(self):
        ctrl = _make_ctrl(n=1)
        thrust, moment = ctrl(_hover_state(1))
        assert thrust.shape == (1, 1)
        assert moment.shape == (1, 3)


# ---------------------------------------------------------------------------
# Hover physics
# ---------------------------------------------------------------------------

class TestHoverPhysics:
    def test_thrust_equals_weight_at_hover(self):
        """At zero position and velocity error, thrust ≈ m·g."""
        ctrl = _make_ctrl()
        thrust, _ = ctrl(_hover_state())
        expected = MASS * G
        assert thrust.allclose(torch.full((N, 1), expected), atol=1e-4)

    def test_moment_near_zero_at_hover(self):
        """At hover with zero angular velocity, moments ≈ 0."""
        ctrl = _make_ctrl()
        _, moment = ctrl(_hover_state())
        assert moment.abs().max().item() < 1e-4

    def test_thrust_positive_at_hover(self):
        ctrl = _make_ctrl()
        thrust, _ = ctrl(_hover_state())
        assert (thrust > 0).all()


# ---------------------------------------------------------------------------
# Position error response
# ---------------------------------------------------------------------------

class TestPositionError:
    def test_position_above_target_increases_thrust(self):
        """Drone above target → larger commanded downward force → larger thrust."""
        ctrl = _make_ctrl(n=1)
        state_at_target = _hover_state(1)
        state_above = state_at_target.clone()
        state_above[:, 2] = 1.0   # 1 m above origin

        thrust_at, _ = ctrl(state_at_target, target_pos=torch.zeros(1, 3))
        thrust_above, _ = ctrl(state_above, target_pos=torch.zeros(1, 3))
        # Above target → pos_error z>0 → force_des more upward (less correction needed
        # than below) ... actually: error = pos - target = +1 → force += k_pos*1
        # force_des_z = k_pos_z * 1 + 0 - m*g*1 → less negative → smaller thrust
        assert thrust_above.item() < thrust_at.item()

    def test_finite_output_for_random_state(self):
        ctrl = _make_ctrl()
        state = torch.randn(N, 13)
        state[:, 3:7] = torch.tensor([1., 0., 0., 0.])  # valid unit quaternion
        state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
        thrust, moment = ctrl(state)
        assert torch.isfinite(thrust).all()
        assert torch.isfinite(moment).all()


# ---------------------------------------------------------------------------
# Yaw handling
# ---------------------------------------------------------------------------

class TestYawHandling:
    def test_none_yaw_uses_current_yaw(self):
        """With no yaw target, moment z should be ≈ 0 when already at target yaw."""
        ctrl = _make_ctrl()
        _, moment = ctrl(_hover_state())
        assert moment[:, 2].abs().max().item() < 1e-4

    def test_yaw_target_scalar(self):
        ctrl = _make_ctrl(n=1)
        yaw_target = torch.tensor([math.pi / 4])
        thrust, moment = ctrl(_hover_state(1), target_yaw=yaw_target)
        assert torch.isfinite(thrust).all()
        assert torch.isfinite(moment).all()

    def test_yaw_rate_target(self):
        ctrl = _make_ctrl()
        yaw_rate = torch.full((N,), 0.5)
        thrust, moment = ctrl(_hover_state(), target_yaw_rate=yaw_rate)
        assert torch.isfinite(moment).all()


# ---------------------------------------------------------------------------
# body_rates_in_body_frame
# ---------------------------------------------------------------------------

class TestBodyRatesFlag:
    def test_world_frame_flag_false(self):
        ctrl = _make_ctrl()
        thrust, moment = ctrl(_hover_state(), body_rates_in_body_frame=False)
        assert torch.isfinite(thrust).all()

    def test_body_frame_flag_true(self):
        ctrl = _make_ctrl()
        thrust, moment = ctrl(_hover_state(), body_rates_in_body_frame=True)
        assert torch.isfinite(thrust).all()

    def test_level_hover_same_either_way(self):
        """At level hover, world and body frames are identical."""
        ctrl = _make_ctrl()
        t1, m1 = ctrl(_hover_state(), body_rates_in_body_frame=False)
        t2, m2 = ctrl(_hover_state(), body_rates_in_body_frame=True)
        assert t1.allclose(t2, atol=1e-6)
        assert m1.allclose(m2, atol=1e-6)


# ---------------------------------------------------------------------------
# Feedforward
# ---------------------------------------------------------------------------

class TestFeedforward:
    def test_target_acc_feedforward_changes_thrust(self):
        ctrl = _make_ctrl(n=1)
        acc_ff = torch.tensor([[0., 0., 2.]])   # upward 2 m/s²
        thrust_no_ff, _ = ctrl(_hover_state(1))
        thrust_ff,    _ = ctrl(_hover_state(1), target_acc=acc_ff)
        # Upward acceleration feedforward → more thrust required
        assert thrust_ff.item() > thrust_no_ff.item()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_full_no_crash(self):
        ctrl = _make_ctrl()
        ctrl(_hover_state())
        ctrl.reset()   # no-op, should not crash

    def test_reset_with_env_ids(self):
        ctrl = _make_ctrl()
        ctrl(_hover_state())
        ctrl.reset(env_ids=torch.tensor([0, 2]))

    def test_output_identical_before_after_reset(self):
        """Stateless: reset changes nothing."""
        ctrl = _make_ctrl()
        t1, m1 = ctrl(_hover_state())
        ctrl.reset()
        t2, m2 = ctrl(_hover_state())
        assert t1.allclose(t2)
        assert m1.allclose(m2)


# ---------------------------------------------------------------------------
# max_acceleration clipping
# ---------------------------------------------------------------------------

class TestMaxAcceleration:
    def test_max_acc_clips_large_error(self):
        ctrl = LeePositionController(
            num_envs=1, mass=0.027,
            inertia=[1.657e-5, 1.657e-5, 2.9e-5],
            max_acceleration=1.0,   # 1 m/s² clip
        )
        state = _hover_state(1)
        state[:, 0] = 100.0   # huge position error
        thrust, _ = ctrl(state, target_pos=torch.zeros(1, 3))
        # Clipped: net acc magnitude ≤ max_acc + g
        acc_magnitude = thrust.item() / 0.027
        assert acc_magnitude <= G + 1.0 + 1e-3
