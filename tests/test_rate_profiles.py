"""
Tests for RC rate profiles.

Covers:
- Output shape: [N, 3]
- Zero stick input → zero output
- Antisymmetry: f(-x) = -f(x)
- Full stick → output clamped at limit (in rad/s)
- Output device matches input device
- Finite output for random input in [-1, 1]
- Custom parameter overrides respected
- All four profiles: betaflight, raceflight, actual, kiss
"""

from __future__ import annotations

import math

import pytest
import torch

from drone import (
    betaflight_rate_profile,
    raceflight_rate_profile,
    actual_rate_profile,
    kiss_rate_profile,
)

N        = 8
DEG2RAD  = math.pi / 180.0
ALL_PROFILES = [
    betaflight_rate_profile,
    raceflight_rate_profile,
    actual_rate_profile,
    kiss_rate_profile,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(n: int = N) -> torch.Tensor:
    return torch.zeros(n, 3)

def _ones(n: int = N) -> torch.Tensor:
    return torch.ones(n, 3)

def _random(n: int = N) -> torch.Tensor:
    return torch.rand(n, 3) * 2.0 - 1.0  # uniform [-1, 1]


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_shape(self, profile):
        out = profile(_zeros())
        assert out.shape == (N, 3), f"{profile.__name__} wrong shape"

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_single_env(self, profile):
        out = profile(_zeros(1))
        assert out.shape == (1, 3)

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_large_batch(self, profile):
        out = profile(_zeros(256))
        assert out.shape == (256, 3)


# ---------------------------------------------------------------------------
# Zero input → zero output
# ---------------------------------------------------------------------------

class TestZeroInput:
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_zero_stick_zero_rate(self, profile):
        out = profile(_zeros())
        assert out.abs().max().item() < 1e-6, f"{profile.__name__} non-zero at zero stick"


# ---------------------------------------------------------------------------
# Antisymmetry
# ---------------------------------------------------------------------------

class TestAntisymmetry:
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_antisymmetric(self, profile):
        x = _random()
        assert profile(x).allclose(-profile(-x), atol=1e-5), \
            f"{profile.__name__} not antisymmetric"


# ---------------------------------------------------------------------------
# Output in rad/s (not deg/s)
# ---------------------------------------------------------------------------

class TestRadianOutput:
    def test_betaflight_full_stick_in_radians(self):
        """Default limit is 2000 deg/s = 34.9 rad/s, not 2000 rad/s."""
        out = betaflight_rate_profile(_ones())
        assert out.abs().max().item() < 200.0, "Output looks like deg/s, not rad/s"

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_full_stick_below_1000_rad_s(self, profile):
        out = profile(_ones())
        assert out.abs().max().item() < 1000.0, \
            f"{profile.__name__} full-stick output suspiciously large (wrong unit?)"


# ---------------------------------------------------------------------------
# Limit clamping
# ---------------------------------------------------------------------------

class TestLimitClamping:
    def test_betaflight_respects_custom_limit(self):
        limit = torch.tensor([500.0, 500.0, 500.0])   # deg/s
        out   = betaflight_rate_profile(_ones() * 1.0, limit=limit)
        assert out.abs().max().item() <= 500.0 * DEG2RAD + 1e-5

    def test_kiss_respects_custom_limit(self):
        limit = torch.tensor([300.0, 300.0, 300.0])
        out   = kiss_rate_profile(_ones(), limit=limit)
        assert out.abs().max().item() <= 300.0 * DEG2RAD + 1e-5

    def test_actual_respects_custom_limit(self):
        limit = torch.tensor([800.0, 800.0, 800.0])
        out   = actual_rate_profile(_ones(), limit=limit)
        assert out.abs().max().item() <= 800.0 * DEG2RAD + 1e-5


# ---------------------------------------------------------------------------
# Finite output
# ---------------------------------------------------------------------------

class TestFiniteOutput:
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_finite_for_random_input(self, profile):
        for _ in range(10):
            out = profile(_random())
            assert torch.isfinite(out).all(), \
                f"{profile.__name__} non-finite output for random input"

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_finite_at_boundaries(self, profile):
        """Stick exactly at ±1 should not produce inf or NaN."""
        out_pos = profile(_ones())
        out_neg = profile(-_ones())
        assert torch.isfinite(out_pos).all()
        assert torch.isfinite(out_neg).all()


# ---------------------------------------------------------------------------
# Device propagation
# ---------------------------------------------------------------------------

class TestDevice:
    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_output_on_cpu(self, profile):
        x = _zeros().to("cpu")
        out = profile(x)
        assert out.device.type == "cpu"


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestCustomParams:
    def test_betaflight_higher_rc_rate_gives_higher_output(self):
        low  = betaflight_rate_profile(
            _ones() * 0.5, rc_rate=torch.tensor([0.5, 0.5, 0.5]))
        high = betaflight_rate_profile(
            _ones() * 0.5, rc_rate=torch.tensor([2.0, 2.0, 2.0]))
        assert high.abs().max() > low.abs().max()

    def test_raceflight_higher_rate_gives_higher_output(self):
        low  = raceflight_rate_profile(
            _ones() * 0.5, rate=torch.tensor([0.3, 0.3, 0.3]))
        high = raceflight_rate_profile(
            _ones() * 0.5, rate=torch.tensor([1.5, 1.5, 1.5]))
        assert high.abs().max() > low.abs().max()

    def test_actual_higher_max_rate_gives_higher_output(self):
        low  = actual_rate_profile(
            _ones() * 0.5, max_rate=torch.tensor([200., 200., 200.]))
        high = actual_rate_profile(
            _ones() * 0.5, max_rate=torch.tensor([1000., 1000., 1000.]))
        assert high.abs().max() > low.abs().max()

    def test_kiss_higher_rate_gives_higher_output(self):
        low  = kiss_rate_profile(
            _ones() * 0.5, rate=torch.tensor([0.5, 0.5, 0.5]))
        high = kiss_rate_profile(
            _ones() * 0.5, rate=torch.tensor([2.0, 2.0, 2.0]))
        assert high.abs().max() > low.abs().max()


# ---------------------------------------------------------------------------
# Betaflight-specific: super_expo_active flag
# ---------------------------------------------------------------------------

class TestBetaflightSuperExpo:
    def test_super_expo_vs_linear(self):
        x = _ones() * 0.8
        super_on  = betaflight_rate_profile(x, super_expo_active=True)
        super_off = betaflight_rate_profile(x, super_expo_active=False)
        # Super-expo increases output at high stick deflection
        assert super_on.abs().max() != super_off.abs().max()

    def test_super_expo_off_finite(self):
        out = betaflight_rate_profile(_random(), super_expo_active=False)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Cross-profile consistency
# ---------------------------------------------------------------------------

class TestCrossProfile:
    def test_all_profiles_agree_at_zero(self):
        """All profiles should return exactly zero for zero input."""
        for profile in ALL_PROFILES:
            out = profile(_zeros())
            assert out.abs().sum().item() == 0.0, \
                f"{profile.__name__} non-zero at zero"

    def test_all_profiles_produce_positive_for_positive_input(self):
        x = _ones() * 0.5
        for profile in ALL_PROFILES:
            out = profile(x)
            assert (out > 0).all(), \
                f"{profile.__name__} not positive for positive input"
