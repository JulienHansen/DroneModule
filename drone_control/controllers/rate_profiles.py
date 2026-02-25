"""
RC rate profiles — maps normalised stick input to body-rate setpoints.

Each function accepts:
    rc_input : [N, 3]  normalised stick deflection in **[-1, 1]**
                       column order: (roll, pitch, yaw)

and returns:
    [N, 3]  body-rate setpoints in **rad/s**

Internal computation follows firmware conventions (deg/s); the output is
converted to rad/s so it can be passed directly to ``CrazyfliePIDController``
or ``LeePositionController`` as ``target_body_rates``.

Default parameters reproduce the factory presets of each firmware.
"""

from __future__ import annotations

import math

import torch

_DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Betaflight
# ---------------------------------------------------------------------------

def betaflight_rate_profile(
    rc_input: torch.Tensor,
    rc_rate:   torch.Tensor | None = None,
    super_rate: torch.Tensor | None = None,
    rc_expo:   torch.Tensor | None = None,
    limit:     torch.Tensor | None = None,
    *,
    super_expo_active: bool = True,
) -> torch.Tensor:
    """
    Betaflight rate profile.

    Parameters
    ----------
    rc_input : [N, 3]
        Normalised stick deflection in [-1, 1] (roll, pitch, yaw).
    rc_rate : [3], optional
        RC-rate multiplier.  Default: ``[1.55, 1.55, 1.50]``.
    super_rate : [3], optional
        Super-rate (expo on top of expo).  Default: ``[0.73, 0.73, 0.73]``.
    rc_expo : [3], optional
        Centre-feel expo.  Default: ``[0.30, 0.30, 0.30]``.
    limit : [3], optional
        Output clamp in **deg/s**.  Default: ``[2000, 2000, 2000]``.
    super_expo_active : bool
        Use the super-expo path (True, default) or the legacy linear path.

    Returns
    -------
    [N, 3] body-rate setpoints in **rad/s**.
    """
    dev = rc_input.device
    if rc_rate    is None: rc_rate    = torch.tensor([1.55, 1.55, 1.50])
    if super_rate is None: super_rate = torch.tensor([0.73, 0.73, 0.73])
    if rc_expo    is None: rc_expo    = torch.tensor([0.30, 0.30, 0.30])
    if limit      is None: limit      = torch.tensor([2000.0, 2000.0, 2000.0])

    rc_rate    = rc_rate.view(1, 3).to(dev)
    super_rate = super_rate.view(1, 3).to(dev)
    rc_expo    = rc_expo.view(1, 3).to(dev)
    limit      = limit.view(1, 3).to(dev)

    # RC rate > 2 shaping
    rc_rate = torch.where(rc_rate > 2, rc_rate + (rc_rate - 2) * 14.54, rc_rate)

    # Centre-feel expo (cubic blend)
    rc_shaped = rc_input * (rc_input.abs() ** 3) * rc_expo + rc_input * (1.0 - rc_expo)

    if super_expo_active:
        rc_factor   = 1.0 / torch.clamp(1.0 - rc_shaped.abs() * super_rate, 0.01, 1.0)
        rate_deg_s  = 200.0 * rc_rate * rc_shaped * rc_factor
    else:
        rate_deg_s  = ((rc_rate * 100.0 + 27.0) * rc_shaped / 16.0) / 4.1

    rate_deg_s = torch.clamp(rate_deg_s, -limit, limit)
    return rate_deg_s * _DEG2RAD


# ---------------------------------------------------------------------------
# RaceFlight / FlightOne
# ---------------------------------------------------------------------------

def raceflight_rate_profile(
    rc_input: torch.Tensor,
    rc_rate: torch.Tensor | None = None,
    expo:    torch.Tensor | None = None,
    rate:    torch.Tensor | None = None,
    limit:   torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RaceFlight (FlightOne) rate profile.

    Parameters
    ----------
    rc_input : [N, 3]
        Normalised stick deflection in [-1, 1].
    rc_rate : [3], optional
        RC-rate multiplier.  Default: ``[1.0, 1.0, 1.0]``.
    expo : [3], optional
        Expo coefficient.  Default: ``[0.4, 0.4, 0.4]``.
    rate : [3], optional
        Rate multiplier.  Default: ``[0.75, 0.75, 0.75]``.
    limit : [3], optional
        Output clamp in **deg/s**.  Default: ``[2000, 2000, 2000]``.

    Returns
    -------
    [N, 3] body-rate setpoints in **rad/s**.
    """
    dev = rc_input.device
    if rc_rate is None: rc_rate = torch.tensor([1.0,    1.0,    1.0])
    if expo    is None: expo    = torch.tensor([0.4,    0.4,    0.4])
    if rate    is None: rate    = torch.tensor([0.75,   0.75,   0.75])
    if limit   is None: limit   = torch.tensor([2000.0, 2000.0, 2000.0])

    rc_rate = rc_rate.view(1, 3).to(dev)
    expo    = expo.view(1, 3).to(dev)
    rate    = rate.view(1, 3).to(dev)
    limit   = limit.view(1, 3).to(dev)

    rc_shaped  = rc_input * (rc_input.abs() ** 3) * expo + rc_input * (1.0 - expo)
    rate_deg_s = rc_shaped * rate * rc_rate * 667.0

    rate_deg_s = torch.clamp(rate_deg_s, -limit, limit)
    return rate_deg_s * _DEG2RAD


# ---------------------------------------------------------------------------
# Actual (Betaflight ≥ 4.2)
# ---------------------------------------------------------------------------

def actual_rate_profile(
    rc_input: torch.Tensor,
    center_sensitivity: torch.Tensor | None = None,
    max_rate:           torch.Tensor | None = None,
    expo:               torch.Tensor | None = None,
    limit:              torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Betaflight "Actual" rate profile (introduced in BF 4.2).

    Parameters
    ----------
    rc_input : [N, 3]
        Normalised stick deflection in [-1, 1].
    center_sensitivity : [3], optional
        Sensitivity at stick centre [0–1].  Default: ``[1.0, 1.0, 1.0]``.
    max_rate : [3], optional
        Maximum rate at full stick deflection [deg/s].  Default: ``[1100, 1100, 1100]``.
    expo : [3], optional
        Expo coefficient.  Default: ``[0.3, 0.3, 0.3]``.
    limit : [3], optional
        Output clamp in **deg/s**.  Default: ``[2000, 2000, 2000]``.

    Returns
    -------
    [N, 3] body-rate setpoints in **rad/s**.
    """
    dev = rc_input.device
    if center_sensitivity is None: center_sensitivity = torch.tensor([1.0,    1.0,    1.0])
    if max_rate           is None: max_rate           = torch.tensor([1100.0, 1100.0, 1100.0])
    if expo               is None: expo               = torch.tensor([0.3,    0.3,    0.3])
    if limit              is None: limit              = torch.tensor([2000.0, 2000.0, 2000.0])

    center_sensitivity = center_sensitivity.view(1, 3).to(dev)
    max_rate           = max_rate.view(1, 3).to(dev)
    expo               = expo.view(1, 3).to(dev)
    limit              = limit.view(1, 3).to(dev)

    stick      = rc_input.abs()
    expo_curve = stick ** 3 * expo + stick * (1.0 - expo)
    rate_deg_s = torch.sign(rc_input) * (
        center_sensitivity + (1.0 - center_sensitivity) * expo_curve
    ) * max_rate

    rate_deg_s = torch.clamp(rate_deg_s, -limit, limit)
    return rate_deg_s * _DEG2RAD


# ---------------------------------------------------------------------------
# KISS
# ---------------------------------------------------------------------------

def kiss_rate_profile(
    rc_input: torch.Tensor,
    rate:     torch.Tensor | None = None,
    rc_curve: torch.Tensor | None = None,
    limit:    torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KISS rate profile.

    Parameters
    ----------
    rc_input : [N, 3]
        Normalised stick deflection in [-1, 1].
    rate : [3], optional
        Rate multiplier.  Default: ``[1.5, 1.5, 1.5]``.
    rc_curve : [3], optional
        Expo coefficient.  Default: ``[0.3, 0.3, 0.3]``.
    limit : [3], optional
        Output clamp in **deg/s**.  Default: ``[2000, 2000, 2000]``.

    Returns
    -------
    [N, 3] body-rate setpoints in **rad/s**.
    """
    dev = rc_input.device
    if rate     is None: rate     = torch.tensor([1.5,    1.5,    1.5])
    if rc_curve is None: rc_curve = torch.tensor([0.3,    0.3,    0.3])
    if limit    is None: limit    = torch.tensor([2000.0, 2000.0, 2000.0])

    rate     = rate.view(1, 3).to(dev)
    rc_curve = rc_curve.view(1, 3).to(dev)
    limit    = limit.view(1, 3).to(dev)

    expo_input = rc_input * (rc_input.abs() ** 3) * rc_curve + rc_input * (1.0 - rc_curve)
    rate_deg_s = expo_input * rate * 1000.0

    rate_deg_s = torch.clamp(rate_deg_s, -limit, limit)
    return rate_deg_s * _DEG2RAD
