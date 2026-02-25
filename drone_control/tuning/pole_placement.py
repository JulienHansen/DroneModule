"""
PID gain tuning via pole placement.

Theory
──────
For each loop we model the plant as a pure integrator (or gain + integrator)
and use pole placement to derive kp, ki, kd analytically from three inputs:

    - Physical parameters of the drone  (mass, inertia, max_thrust)
    - Desired closed-loop bandwidth     ω₀  [rad/s]
    - Desired damping ratio             ζ   (typically 0.7 = critically-damped)

Loop hierarchy and bandwidth separation
─────────────────────────────────────────
Tune from the *inside* outward.  Each outer loop must be at least 5× slower
than the one it wraps, otherwise the loops interact and the system can go
unstable:

    rate  →  att  →  vel  →  pos
    ω₀            ω₀/5      ω₀/25    ω₀/125

If you only specify ``bandwidth_rate``, the other bandwidths are set
automatically following this rule.

Derivations
───────────
**Rate loop**  (error [rad/s] → α [rad/s²]):
    Plant:   G(s) = 1/s   (omega = integral of alpha)
    PI ctrl: C(s) = kp + ki/s
    Char. poly: s² + kp·s + ki  =  s² + 2ζω·s + ω²
        kp_rate = 2·ζ·ω_rate
        ki_rate = ω_rate²
        kd_rate = kp_rate · τ_d   (derivative time, default 0 → kd=0)

**Attitude loop**  (error [rad] → rate sp [rad/s]):
    Same structure (plant = 1/s assuming rate loop tracks perfectly):
        kp_att = 2·ζ·ω_att
        ki_att = ω_att²
        kd_att = 0

**Velocity loop x/y**  (error [m/s] → roll/pitch angle [deg in YAML]):
    Plant: a ≈ g·θ → v̇ = g·θ → G(s) = g/s
    PI on velocity:
        kp_vel_xy [rad/(m/s)] = 2·ζ·ω_vel / g
        ki_vel_xy [rad/s/(m/s)] = ω_vel² / g
    Converted to firmware deg units (YAML stores deg/(m/s)):
        kp_vel_xy_deg = kp_vel_xy · (180/π)

**Velocity loop z**  (error [m/s] → thrust_cmd [PWM]):
    Effective plant gain:  K = max_thrust · vel_thrust_scale / (thrust_cmd_max · mass)
    PI on vz:
        kp_vel_z = 2·ζ·ω_vel / K
        ki_vel_z = ω_vel²    / K

**Position loop**  (error [m] → vel sp [m/s]):
    Plant: v = integral of a, but vel loop already controls v → G(s) = 1/s
    Pure P (bandwidth = kp):
        kp_pos = ω_pos
        ki_pos = 0  (add a small value for z steady-state if desired)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

DEG2RAD = math.pi / 180.0
G = 9.81   # m/s²


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    """
    PID gains computed by pole placement.

    All gains are in the same units as the ``crazyflie_pid`` YAML section
    and can be passed directly as ``params`` to ``CrazyfliePIDController``.
    """

    # ── Bandwidths used ──────────────────────────────────────────────────────
    bandwidth_rate: float   # [rad/s]
    bandwidth_att:  float   # [rad/s]
    bandwidth_vel:  float   # [rad/s]
    bandwidth_pos:  float   # [rad/s]
    damping:        float

    # ── Rate loop ────────────────────────────────────────────────────────────
    rate_kp:  list[float]
    rate_ki:  list[float]
    rate_kd:  list[float]

    # ── Attitude loop ────────────────────────────────────────────────────────
    att_kp:   list[float]
    att_ki:   list[float]
    att_kd:   list[float]

    # ── Velocity loop ────────────────────────────────────────────────────────
    vel_kp:   list[float]
    vel_ki:   list[float]
    vel_kd:   list[float]

    # ── Position loop ────────────────────────────────────────────────────────
    pos_kp:   list[float]
    pos_ki:   list[float]
    pos_kd:   list[float]

    # ── Warnings emitted during computation ──────────────────────────────────
    warnings: list[str] = field(default_factory=list)

    def to_params(self) -> dict:
        """
        Return a dict compatible with ``CrazyfliePIDController(params=...)``.

        Feed-forward terms and saturation limits are left at their defaults;
        override them manually if needed.
        """
        return {
            "rate_kp":  self.rate_kp,
            "rate_ki":  self.rate_ki,
            "rate_kd":  self.rate_kd,
            "att_kp":   self.att_kp,
            "att_ki":   self.att_ki,
            "att_kd":   self.att_kd,
            "vel_kp":   self.vel_kp,
            "vel_ki":   self.vel_ki,
            "vel_kd":   self.vel_kd,
            "pos_kp":   self.pos_kp,
            "pos_ki":   self.pos_ki,
            "pos_kd":   self.pos_kd,
        }

    def summary(self) -> str:
        """Human-readable summary of the computed gains."""
        lines = [
            "── Pole-placement tuning result ─────────────────────────────",
            f"  Bandwidths  rate={self.bandwidth_rate:.1f}  att={self.bandwidth_att:.1f}"
            f"  vel={self.bandwidth_vel:.2f}  pos={self.bandwidth_pos:.3f}  rad/s",
            f"  Damping ζ = {self.damping}",
            "",
            "  Rate loop",
            f"    kp = {self._fmt(self.rate_kp)}",
            f"    ki = {self._fmt(self.rate_ki)}",
            f"    kd = {self._fmt(self.rate_kd)}",
            "",
            "  Attitude loop",
            f"    kp = {self._fmt(self.att_kp)}",
            f"    ki = {self._fmt(self.att_ki)}",
            f"    kd = {self._fmt(self.att_kd)}",
            "",
            "  Velocity loop  (x/y in deg/(m/s), z in PWM/(m/s))",
            f"    kp = {self._fmt(self.vel_kp)}",
            f"    ki = {self._fmt(self.vel_ki)}",
            f"    kd = {self._fmt(self.vel_kd)}",
            "",
            "  Position loop",
            f"    kp = {self._fmt(self.pos_kp)}",
            f"    ki = {self._fmt(self.pos_ki)}",
            f"    kd = {self._fmt(self.pos_kd)}",
        ]
        if self.warnings:
            lines += ["", "  ⚠ Warnings:"] + [f"    - {w}" for w in self.warnings]
        lines.append("─────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    @staticmethod
    def _fmt(vals: list[float]) -> str:
        return "[" + ", ".join(f"{v:.4g}" for v in vals) + "]"

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def tune_from_physics(
    mass:          float,
    inertia:       list[float],         # [Ixx, Iyy, Izz]  [kg·m²]
    bandwidth_rate: float,              # rate loop ω₀      [rad/s]
    bandwidth_att:  Optional[float] = None,
    bandwidth_vel:  Optional[float] = None,
    bandwidth_pos:  Optional[float] = None,
    damping:        float = 0.7,        # ζ (0.7 ≈ critical, 1.0 = overdamped)
    derivative_time: float = 0.0,       # τ_d: kd = kp·τ_d  [s], 0 = no derivative
    # Drone thrust constants (needed to scale the z-velocity gains)
    max_thrust:       float = 0.638,    # [N]
    thrust_cmd_max:   float = 65535.0,  # [PWM]
    vel_thrust_scale: float = 1000.0,   # [PWM / (m/s²·s)]
    # Optional: emit a warning when ω₀ is suspiciously high
    sim_dt: Optional[float] = None,
) -> TuningResult:
    """
    Compute PID gains for ``CrazyfliePIDController`` via pole placement.

    Parameters
    ----------
    mass : float
        Drone mass [kg].
    inertia : list of float
        Principal moments of inertia ``[Ixx, Iyy, Izz]`` [kg·m²].
    bandwidth_rate : float
        Desired closed-loop bandwidth of the **rate loop** [rad/s].
        Typical range for small drones: 20–60 rad/s.
    bandwidth_att : float, optional
        Attitude loop bandwidth [rad/s].  Default: ``bandwidth_rate / 5``.
    bandwidth_vel : float, optional
        Velocity loop bandwidth [rad/s].  Default: ``bandwidth_att / 5``.
    bandwidth_pos : float, optional
        Position loop bandwidth [rad/s].  Default: ``bandwidth_vel / 5``.
    damping : float
        Damping ratio ζ for all PI loops.  0.7 gives a good balance between
        speed and overshoot (≈ 5 %).  1.0 is critically damped (no overshoot).
    derivative_time : float
        Derivative time constant τ_d [s].  ``kd = kp · τ_d``.  Set to 0
        (default) to use a pure PI controller.
    max_thrust : float
        Maximum combined thrust of all motors [N].  Used to scale z-velocity
        gains.  Matches ``drone.max_thrust`` in the YAML config.
    thrust_cmd_max : float
        Full-scale PWM command value.  Default: 65535 (16-bit).
    vel_thrust_scale : float
        PWM scale factor for the z-velocity output.  Matches
        ``vel_thrust_scale`` in the YAML ``crazyflie_pid`` section.
    sim_dt : float, optional
        Simulation timestep [s].  When provided, a warning is emitted if any
        bandwidth exceeds ``0.1 / sim_dt`` (10 % of Nyquist).

    Returns
    -------
    TuningResult
        Computed gains and diagnostic information.  Call ``.to_params()`` to
        get a dict ready for ``CrazyfliePIDController``.

    Examples
    --------
    >>> from drone_control import load_config, CrazyfliePIDController
    >>> from drone_control.tuning import tune_from_physics
    >>>
    >>> cfg    = load_config("configs/crazyflie.yaml")
    >>> result = tune_from_physics(
    ...     mass=cfg.physics.mass,
    ...     inertia=[cfg.physics.inertia.ixx,
    ...              cfg.physics.inertia.iyy,
    ...              cfg.physics.inertia.izz],
    ...     bandwidth_rate=30.0,
    ...     damping=0.7,
    ...     max_thrust=cfg.physics.max_thrust,
    ...     sim_dt=0.002,
    ... )
    >>> print(result)
    >>>
    >>> ctrl = CrazyfliePIDController.from_drone_config(
    ...     cfg, num_envs=1, dt=0.002
    ... )
    >>> ctrl.set_rate_gains(**{k: result.to_params()[k]
    ...                        for k in ("rate_kp", "rate_ki", "rate_kd")})
    """
    warns: list[str] = []
    Ixx, Iyy, Izz = float(inertia[0]), float(inertia[1]), float(inertia[2])

    # ── Bandwidth cascade (separation factor ≥ 5) ───────────────────────────
    ω_r = float(bandwidth_rate)
    ω_a = float(bandwidth_att) if bandwidth_att is not None else ω_r / 5.0
    ω_v = float(bandwidth_vel) if bandwidth_vel is not None else ω_a / 5.0
    ω_p = float(bandwidth_pos) if bandwidth_pos is not None else ω_v / 5.0
    ζ   = float(damping)

    _check_separation("rate→att",  ω_r, ω_a, warns)
    _check_separation("att→vel",   ω_a, ω_v, warns)
    _check_separation("vel→pos",   ω_v, ω_p, warns)

    if sim_dt is not None:
        nyquist_limit = 0.1 / sim_dt   # 10 % of sampling rate
        for name, ω in [("rate", ω_r), ("att", ω_a), ("vel", ω_v), ("pos", ω_p)]:
            if ω > nyquist_limit:
                warns.append(
                    f"{name} bandwidth {ω:.1f} rad/s exceeds 10 % of Nyquist "
                    f"({nyquist_limit:.1f} rad/s at dt={sim_dt} s) — "
                    "consider a smaller bandwidth or timestep."
                )

    # ── Rate loop  (PI + optional D) ────────────────────────────────────────
    # Plant: G(s) = 1/s  (omega = integral of alpha, gains in [1/s] space)
    # Char. poly (PI): s² + kp·s + ki = s² + 2ζω·s + ω²
    kp_rate_rp = 2.0 * ζ * ω_r               # roll / pitch (same Ixx=Iyy here)
    ki_rate_rp = ω_r ** 2
    kd_rate_rp = kp_rate_rp * derivative_time

    kp_rate_yaw = 2.0 * ζ * ω_r              # same bandwidth on yaw
    ki_rate_yaw = ω_r ** 2
    kd_rate_yaw = kp_rate_yaw * derivative_time

    rate_kp = [round(kp_rate_rp, 4), round(kp_rate_rp, 4), round(kp_rate_yaw, 4)]
    rate_ki = [round(ki_rate_rp, 4), round(ki_rate_rp, 4), round(ki_rate_yaw, 4)]
    rate_kd = [round(kd_rate_rp, 4), round(kd_rate_rp, 4), round(kd_rate_yaw, 4)]

    # ── Attitude loop  (PI) ─────────────────────────────────────────────────
    # Plant: G(s) = 1/s  (angle = integral of rate, assuming rate loop perfect)
    # Char. poly (PI): s² + kp·s + ki = s² + 2ζω·s + ω²
    kp_att = 2.0 * ζ * ω_a
    ki_att = ω_a ** 2
    kd_att = 0.0   # derivative on angle error is rarely useful

    att_kp = [round(kp_att, 4)] * 3
    att_ki = [round(ki_att, 4)] * 3
    att_kd = [round(kd_att, 4)] * 3

    # ── Velocity loop x/y  (PI) ─────────────────────────────────────────────
    # Plant: a ≈ g·θ → v̇ = g·θ → G(s) = g/s  (θ tracked perfectly by att loop)
    # PI: kp·G(s)/s → char. poly: s² + kp·g·s + ki·g
    # Matching: kp = 2ζω/g [rad/(m/s)], ki = ω²/g [1/s]
    # YAML stores in deg/(m/s): multiply by 180/π
    kp_vel_xy_deg = (2.0 * ζ * ω_v / G) / DEG2RAD
    ki_vel_xy_deg = (ω_v ** 2     / G) / DEG2RAD

    # ── Velocity loop z  (PI) ───────────────────────────────────────────────
    # Plant: dv_z/dt = F_z/m, where F_z = thrust_cmd_scale · delta_PWM
    #        delta_PWM = vel_out_z · vel_thrust_scale
    #   → dv_z/dt = (thrust_cmd_scale · vel_thrust_scale / m) · vel_out_z
    #   → effective plant gain: K_z = max_thrust · vel_thrust_scale / (thrust_cmd_max · mass)
    thrust_cmd_scale = max_thrust / thrust_cmd_max
    K_z = thrust_cmd_scale * vel_thrust_scale / mass

    kp_vel_z = 2.0 * ζ * ω_v / K_z
    ki_vel_z = ω_v ** 2      / K_z

    vel_kp = [round(kp_vel_xy_deg, 4), round(kp_vel_xy_deg, 4), round(kp_vel_z, 4)]
    vel_ki = [round(ki_vel_xy_deg, 4), round(ki_vel_xy_deg, 4), round(ki_vel_z, 4)]
    vel_kd = [0.0, 0.0, 0.0]

    # ── Position loop  (P) ──────────────────────────────────────────────────
    # Plant: G(s) = 1/s  (position = integral of velocity)
    # Pure P: closed-loop pole at -kp_pos → bandwidth = kp_pos
    kp_pos = ω_p

    pos_kp = [round(kp_pos, 4)] * 3
    pos_ki = [0.0, 0.0, 0.0]
    pos_kd = [0.0, 0.0, 0.0]

    return TuningResult(
        bandwidth_rate=ω_r,
        bandwidth_att=ω_a,
        bandwidth_vel=ω_v,
        bandwidth_pos=ω_p,
        damping=ζ,
        rate_kp=rate_kp, rate_ki=rate_ki, rate_kd=rate_kd,
        att_kp=att_kp,   att_ki=att_ki,   att_kd=att_kd,
        vel_kp=vel_kp,   vel_ki=vel_ki,   vel_kd=vel_kd,
        pos_kp=pos_kp,   pos_ki=pos_ki,   pos_kd=pos_kd,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _check_separation(
    label: str,
    ω_inner: float,
    ω_outer: float,
    warns: list[str],
    min_ratio: float = 3.0,
) -> None:
    """Warn if the bandwidth separation between two adjacent loops is too small."""
    if ω_outer <= 0:
        return
    ratio = ω_inner / ω_outer
    if ratio < min_ratio:
        warns.append(
            f"{label} bandwidth ratio is {ratio:.1f}× "
            f"(inner={ω_inner:.2f}, outer={ω_outer:.2f} rad/s). "
            f"Recommend ≥ {min_ratio}× to avoid loop interaction."
        )
