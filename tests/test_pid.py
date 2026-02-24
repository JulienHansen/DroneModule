"""
Unit tests for PID_Vectorized.

Each test targets one specific behaviour of the PID controller:
proportional response, integral with anti-windup, derivative filtering,
and reset mechanics.
"""

import pytest
import torch

from drone_control.controllers.pid import PID_Vectorized

N      = 3
DEVICE = torch.device("cpu")
TS     = 0.01  # 100 Hz timestep


def make_pid(**kwargs) -> PID_Vectorized:
    defaults = dict(num_envs=N, device=DEVICE,
                    kp=0.0, ki=0.0, kd=0.0,
                    limit_up=999.0, limit_down=-999.0,
                    tau=0.01)
    defaults.update(kwargs)
    return PID_Vectorized(**defaults)


# ---------------------------------------------------------------------------
# Proportional
# ---------------------------------------------------------------------------

class TestProportional:
    def test_output_equals_kp_times_error(self):
        pid = make_pid(kp=5.0)
        error = torch.tensor([1.0, 2.0, 3.0])
        out = pid.forward(error, TS)
        assert torch.allclose(out, error * 5.0)

    def test_positive_saturation(self):
        pid = make_pid(kp=5.0, limit_up=7.0, limit_down=-7.0)
        error = torch.tensor([0.5, 1.5, 3.0])  # kp*err = [2.5, 7.5, 15.0]
        out = pid.forward(error, TS)
        assert torch.allclose(out, torch.clamp(error * 5.0, -7.0, 7.0))

    def test_negative_saturation(self):
        pid = make_pid(kp=5.0, limit_up=10.0, limit_down=-10.0)
        error = torch.tensor([-3.0, -2.0, -1.0])
        out = pid.forward(error, TS)
        assert torch.allclose(out, torch.clamp(error * 5.0, -10.0, 10.0))


# ---------------------------------------------------------------------------
# Integral + anti-windup
# ---------------------------------------------------------------------------

class TestIntegral:
    def test_integrator_accumulates(self):
        pid = make_pid(ki=10.0, limit_up=999.0, limit_down=-999.0)
        error = torch.ones(N)
        prev = torch.zeros(N)
        for _ in range(5):
            out = pid.forward(error, TS)
            assert torch.all(out > prev), "Integral output should grow each step"
            prev = out.clone()

    def test_antiwindup_limits_integrator(self):
        """After hitting saturation, the integrator must stop growing."""
        pid = make_pid(ki=10.0, limit_up=1.0, limit_down=-1.0)
        error = torch.ones(N)
        for _ in range(30):
            out = pid.forward(error, TS)
        # Output must be saturated
        assert torch.allclose(out, torch.ones(N))
        # Integrator must not have grown unboundedly  (anti-windup keeps it near 0.1)
        assert torch.all(pid.integrator < 0.2), \
            f"Anti-windup failed: integrator = {pid.integrator}"

    def test_ki_zero_disables_integrator(self):
        pid = make_pid(kp=1.0, ki=0.0)
        error = torch.ones(N)
        for _ in range(10):
            pid.forward(error, TS)
        assert torch.all(pid.integrator == 0.0), "Integrator should stay zero when ki=0"


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------

class TestDerivative:
    def test_responds_to_error_change(self):
        pid = make_pid(kd=5.0, tau=0.05)
        pid.forward(torch.ones(N), TS)             # step 1: error = 1
        out = pid.forward(torch.full((N,), 2.0), TS)  # step 2: error jumps to 2
        assert torch.all(out > 0.0), "D-term should produce positive output for increasing error"

    def test_zero_on_constant_error_after_warmup(self):
        """Derivative filter decays: after many identical steps output → kp * error only."""
        pid = make_pid(kp=1.0, kd=10.0, tau=0.001)  # very short tau → fast decay
        error = torch.ones(N)
        for _ in range(200):
            out = pid.forward(error, TS)
        # After 200 steps with constant error, differentiator ≈ 0
        assert torch.all(pid.differentiator.abs() < 1e-3), \
            f"Differentiator did not decay: {pid.differentiator}"

    def test_filter_tau_affects_magnitude(self):
        """Larger tau → more filtering → smaller initial D spike."""
        pid_fast = make_pid(kd=1.0, tau=0.001)
        pid_slow = make_pid(kd=1.0, tau=1.0)
        step = torch.ones(N)
        out_fast = pid_fast.forward(step, TS)
        out_slow = pid_slow.forward(step, TS)
        assert torch.all(out_fast.abs() > out_slow.abs()), \
            "Smaller tau (less filtering) should give larger derivative kick"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def _warm_up(self, pid, steps=5):
        for _ in range(steps):
            pid.forward(torch.ones(N), TS)

    def test_full_reset_clears_all_states(self):
        pid = make_pid(kp=1.0, ki=1.0, kd=1.0)
        self._warm_up(pid)
        pid.reset()
        assert torch.all(pid.integrator    == 0.0)
        assert torch.all(pid.differentiator == 0.0)
        assert torch.all(pid.error_d1      == 0.0)
        assert torch.all(pid.u             == 0.0)

    def test_partial_reset_leaves_other_envs_intact(self):
        pid = make_pid(kp=1.0, ki=1.0)
        self._warm_up(pid)
        pre_integ = pid.integrator.clone()
        pid.reset(env_ids=torch.tensor([0]))
        assert pid.integrator[0] == 0.0,          "env 0 should be reset"
        assert pid.integrator[1] == pre_integ[1], "env 1 should be untouched"
        assert pid.integrator[2] == pre_integ[2], "env 2 should be untouched"

    def test_reset_then_forward_is_identical_to_fresh(self):
        """Output after reset should match a freshly constructed PID."""
        pid_a = make_pid(kp=2.0, ki=0.5)
        pid_b = make_pid(kp=2.0, ki=0.5)
        self._warm_up(pid_a)
        pid_a.reset()
        error = torch.tensor([0.3, 0.6, 0.9])
        assert torch.allclose(pid_a.forward(error, TS), pid_b.forward(error, TS))
