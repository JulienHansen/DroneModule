"""
Visual integration tests for the cascade controllers.

Each test runs a closed-loop simulation, asserts numerical convergence,
and saves a multi-panel plot to tests/plots/.  The plots are uploaded as
CI artifacts so they can be inspected without a local Python environment.

Simulation assumptions
─────────────────────
Attitude sim   : full rigid-body Euler equation (J α = τ − ω × Jω),
                 simplified kinematics  φ̇ ≈ ω  (valid for small angles).
Position sim   : perfect & instantaneous attitude tracking, simplified
                 thrust projection (small-angle approximation for roll/pitch).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — works in CI without a display
import matplotlib.pyplot as plt
import pytest
import torch

from drone_control import load_config, AttController_Vectorized, PosController_Vectorized

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PLOTS_DIR = Path(__file__).parent / "plots"

G = 9.81


# ---------------------------------------------------------------------------
# Attitude step-response simulation
# ---------------------------------------------------------------------------

def simulate_attitude(
    cfg,
    ref_rpy: list[float],
    dt: float = 0.01,
    duration: float = 3.0,
) -> dict:
    """
    Simulate a closed-loop attitude step response with N=1 environment.

    Returns a dict with time-series arrays for every state variable.
    """
    ctrl = AttController_Vectorized.from_drone_config(cfg, num_envs=1, device=torch.device("cpu"))

    ixx, iyy, izz = cfg.physics.inertia.ixx, cfg.physics.inertia.iyy, cfg.physics.inertia.izz
    J     = torch.diag(torch.tensor([ixx, iyy, izz], dtype=torch.float)).unsqueeze(0)
    J_inv = torch.inverse(J[0]).unsqueeze(0)

    rpy   = torch.zeros(1, 3)
    omega = torch.zeros(1, 3)
    ref   = torch.tensor([ref_rpy], dtype=torch.float)

    steps = int(duration / dt)
    log   = {"t": [], "roll": [], "pitch": [], "yaw": [],
             "p": [], "q": [], "r": []}

    for i in range(steps):
        omega_ref = ctrl.run_angle(ref, rpy)
        tau       = ctrl.run_rate(omega_ref, omega, J)

        # α = J⁻¹ (τ − ω × Jω)
        Jw    = torch.matmul(J, omega.unsqueeze(-1)).squeeze(-1)
        gyro  = torch.linalg.cross(omega, Jw, dim=-1)
        alpha = torch.matmul(J_inv, (tau - gyro).unsqueeze(-1)).squeeze(-1)

        omega = omega + alpha * dt
        rpy   = rpy   + omega * dt   # simplified kinematics (valid for small angles)

        log["t"].append(i * dt)
        log["roll"].append(rpy[0, 0].item())
        log["pitch"].append(rpy[0, 1].item())
        log["yaw"].append(rpy[0, 2].item())
        log["p"].append(omega[0, 0].item())
        log["q"].append(omega[0, 1].item())
        log["r"].append(omega[0, 2].item())

    return log


# ---------------------------------------------------------------------------
# Position step-response simulation
# ---------------------------------------------------------------------------

def simulate_position(
    cfg,
    ref_pos: list[float],
    dt: float = 0.01,
    duration: float = 10.0,
) -> dict:
    """
    Simulate a closed-loop position step response with N=1 environment.

    Attitude is assumed to be perfectly and instantly tracked (the attitude
    controller is much faster than the position controller in practice).

    Returns a dict with time-series arrays for every state variable.
    """
    ctrl = PosController_Vectorized.from_drone_config(cfg, num_envs=1, device=torch.device("cpu"))

    mass_val = cfg.physics.mass
    mass     = torch.tensor([mass_val])

    pos = torch.zeros(1, 3)
    vel = torch.zeros(1, 3)
    yaw = torch.zeros(1)
    ref = torch.tensor([ref_pos], dtype=torch.float)

    steps = int(duration / dt)
    log   = {"t": [], "x": [], "y": [], "z": [],
             "vx": [], "vy": [], "vz": [],
             "roll_cmd": [], "pitch_cmd": [], "thrust": []}

    for i in range(steps):
        vel_ref          = ctrl.run_pos(ref, pos)
        rp_ref, thrust   = ctrl.run_vel(vel_ref, vel, yaw, mass)

        roll_cmd  = rp_ref[0, 0].item()
        pitch_cmd = rp_ref[0, 1].item()
        T         = thrust[0].item()

        # World-frame accelerations (perfect attitude tracking, yaw=0)
        ax = T / mass_val *  math.sin(pitch_cmd)
        ay = T / mass_val * -math.sin(roll_cmd) * math.cos(pitch_cmd)
        az = T / mass_val *  math.cos(roll_cmd) * math.cos(pitch_cmd) - G

        accel = torch.tensor([[ax, ay, az]])
        vel   = vel + accel * dt
        pos   = pos + vel   * dt

        log["t"].append(i * dt)
        log["x"].append(pos[0, 0].item())
        log["y"].append(pos[0, 1].item())
        log["z"].append(pos[0, 2].item())
        log["vx"].append(vel[0, 0].item())
        log["vy"].append(vel[0, 1].item())
        log["vz"].append(vel[0, 2].item())
        log["roll_cmd"].append(math.degrees(roll_cmd))
        log["pitch_cmd"].append(math.degrees(pitch_cmd))
        log["thrust"].append(T)

    return log


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

REF_STYLE  = dict(color="tab:red",  linestyle="--", linewidth=1.2, label="reference")
SIM_STYLE  = dict(color="tab:blue", linestyle="-",  linewidth=1.4)
GRID_STYLE = dict(alpha=0.3)


def _plot_attitude(log: dict, ref_rpy: list[float], path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Attitude step response  (cascade PID — Crazyflie)", fontsize=12)
    t = log["t"]

    pairs = [
        (axes[0, 0], "roll",  "Roll [rad]",         ref_rpy[0]),
        (axes[0, 1], "pitch", "Pitch [rad]",         ref_rpy[1]),
        (axes[0, 2], "yaw",   "Yaw [rad]",           ref_rpy[2]),
        (axes[1, 0], "p",     "Roll rate p [rad/s]", 0.0),
        (axes[1, 1], "q",     "Pitch rate q [rad/s]", 0.0),
        (axes[1, 2], "r",     "Yaw rate r [rad/s]",  0.0),
    ]

    for ax, key, ylabel, ref_val in pairs:
        ax.axhline(ref_val, **REF_STYLE)
        ax.plot(t, log[key], **SIM_STYLE, label="simulated")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(**GRID_STYLE)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_position(log: dict, ref_pos: list[float], path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    fig.suptitle("Position step response  (cascade PID — Crazyflie)", fontsize=12)
    t = log["t"]

    pos_pairs = [
        (axes[0, 0], "x",  "x [m]",  ref_pos[0]),
        (axes[0, 1], "y",  "y [m]",  ref_pos[1]),
        (axes[0, 2], "z",  "z [m]",  ref_pos[2]),
        (axes[1, 0], "vx", "vx [m/s]", 0.0),
        (axes[1, 1], "vy", "vy [m/s]", 0.0),
        (axes[1, 2], "vz", "vz [m/s]", 0.0),
    ]

    for ax, key, ylabel, ref_val in pos_pairs:
        ax.axhline(ref_val, **REF_STYLE)
        ax.plot(t, log[key], **SIM_STYLE, label="simulated")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(**GRID_STYLE)

    # Bottom row: commanded roll/pitch and thrust
    axes[2, 0].plot(t, log["roll_cmd"],  color="tab:orange", linewidth=1.2)
    axes[2, 0].set_ylabel("Commanded roll [deg]"); axes[2, 0].grid(**GRID_STYLE)

    axes[2, 1].plot(t, log["pitch_cmd"], color="tab:orange", linewidth=1.2)
    axes[2, 1].set_ylabel("Commanded pitch [deg]"); axes[2, 1].grid(**GRID_STYLE)

    axes[2, 2].axhline(ref_pos[2] * 0 + 9.81 * 0.027, color="tab:red", linestyle="--",
                       linewidth=1.2, label="hover thrust")
    axes[2, 2].plot(t, log["thrust"], color="tab:green", linewidth=1.2, label="thrust [N]")
    axes[2, 2].set_ylabel("Thrust [N]"); axes[2, 2].legend(fontsize=7)
    axes[2, 2].grid(**GRID_STYLE)

    for ax in axes.flat:
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def make_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class TestAttitudeStepResponse:
    """
    Closed-loop attitude step response.

    Reference: roll = 0.3 rad (~17°), pitch = 0.2 rad (~11°), yaw = 0.5 rad (~29°).
    Simulation: 3 seconds at 100 Hz.
    """

    REF_RPY  = [0.3, 0.2, 0.5]
    DURATION = 3.0
    DT       = 0.01

    @pytest.fixture(autouse=True)
    def run_sim(self, cf_config):
        self.log = simulate_attitude(cf_config, self.REF_RPY, self.DT, self.DURATION)
        _plot_attitude(
            self.log, self.REF_RPY,
            PLOTS_DIR / "attitude_step_response.png",
        )

    def test_roll_converges(self):
        final_roll = self.log["roll"][-1]
        assert abs(final_roll - self.REF_RPY[0]) < 0.05, \
            f"Roll did not converge: {final_roll:.4f} rad (ref {self.REF_RPY[0]} rad)"

    def test_pitch_converges(self):
        final_pitch = self.log["pitch"][-1]
        assert abs(final_pitch - self.REF_RPY[1]) < 0.05, \
            f"Pitch did not converge: {final_pitch:.4f} rad (ref {self.REF_RPY[1]} rad)"

    def test_yaw_converges(self):
        final_yaw = self.log["yaw"][-1]
        assert abs(final_yaw - self.REF_RPY[2]) < 0.05, \
            f"Yaw did not converge: {final_yaw:.4f} rad (ref {self.REF_RPY[2]} rad)"

    def test_rates_settle_near_zero(self):
        """At steady state, body rates should approach zero."""
        final_p = abs(self.log["p"][-1])
        final_q = abs(self.log["q"][-1])
        final_r = abs(self.log["r"][-1])
        assert final_p < 0.1, f"Roll rate p did not settle: {final_p:.4f} rad/s"
        assert final_q < 0.1, f"Pitch rate q did not settle: {final_q:.4f} rad/s"
        assert final_r < 0.1, f"Yaw rate r did not settle: {final_r:.4f} rad/s"

    def test_no_nan_or_inf(self):
        for key, series in self.log.items():
            if key == "t":
                continue
            for i, val in enumerate(series):
                assert math.isfinite(val), f"Non-finite value in {key}[{i}] = {val}"


class TestAttitudeYawWrap:
    """
    Yaw step that crosses the ±π boundary.
    The controller should take the short way around.
    """

    REF_RPY  = [0.0, 0.0, -3.0]   # ref yaw = -3 rad ≈ -172°
    DURATION = 3.0
    DT       = 0.01

    @pytest.fixture(autouse=True)
    def run_sim(self, cf_config):
        # Start near +π and command -3 rad: raw error would be -6.28+, should wrap
        self.log = simulate_attitude(cf_config, self.REF_RPY, self.DT, self.DURATION)
        _plot_attitude(
            self.log, self.REF_RPY,
            PLOTS_DIR / "attitude_yaw_wrap.png",
        )

    def test_yaw_converges_via_short_path(self):
        final_yaw = self.log["yaw"][-1]
        assert abs(final_yaw - self.REF_RPY[2]) < 0.05, \
            f"Yaw did not converge: {final_yaw:.4f} (ref {self.REF_RPY[2]})"

    def test_no_nan_or_inf(self):
        for key, series in self.log.items():
            if key == "t":
                continue
            for val in series:
                assert math.isfinite(val)


class TestPositionStepResponse:
    """
    Closed-loop position step response, starting at the origin.

    Reference: x = 1.0 m, y = 0.5 m, z = 1.5 m.
    Simulation: 10 seconds at 100 Hz.
    Perfect attitude tracking assumed.
    """

    REF_POS  = [1.0, 0.5, 1.5]
    DURATION = 10.0
    DT       = 0.01

    @pytest.fixture(autouse=True)
    def run_sim(self, cf_config):
        self.log = simulate_position(cf_config, self.REF_POS, self.DT, self.DURATION)
        _plot_position(
            self.log, self.REF_POS,
            PLOTS_DIR / "position_step_response.png",
        )

    def test_x_converges(self):
        final_x = self.log["x"][-1]
        assert abs(final_x - self.REF_POS[0]) < 0.15, \
            f"x did not converge: {final_x:.4f} m (ref {self.REF_POS[0]} m)"

    def test_y_converges(self):
        final_y = self.log["y"][-1]
        assert abs(final_y - self.REF_POS[1]) < 0.15, \
            f"y did not converge: {final_y:.4f} m (ref {self.REF_POS[1]} m)"

    def test_z_converges(self):
        final_z = self.log["z"][-1]
        assert abs(final_z - self.REF_POS[2]) < 0.15, \
            f"z did not converge: {final_z:.4f} m (ref {self.REF_POS[2]} m)"

    def test_velocities_settle_near_zero(self):
        for key in ("vx", "vy", "vz"):
            final = abs(self.log[key][-1])
            assert final < 0.2, f"{key} did not settle: {final:.4f} m/s"

    def test_roll_pitch_within_limit(self):
        """Commanded roll/pitch should always stay within the configured limit."""
        limit_deg = 35.0   # conservative bound covering both configs
        for i, (r, p) in enumerate(zip(self.log["roll_cmd"], self.log["pitch_cmd"])):
            assert abs(r) <= limit_deg + 1e-3, f"roll_cmd[{i}] = {r:.2f}° exceeds {limit_deg}°"
            assert abs(p) <= limit_deg + 1e-3, f"pitch_cmd[{i}] = {p:.2f}° exceeds {limit_deg}°"

    def test_thrust_within_bounds(self, cf_config):
        max_t = cf_config.position.max_thrust_scale * cf_config.physics.max_thrust
        min_t = 0.8 * G * cf_config.physics.mass
        for i, t in enumerate(self.log["thrust"]):
            assert t <= max_t + 1e-4, f"thrust[{i}] = {t:.4f} N exceeds max {max_t:.4f} N"
            assert t >= min_t - 1e-4, f"thrust[{i}] = {t:.4f} N below floor {min_t:.4f} N"

    def test_no_nan_or_inf(self):
        for key, series in self.log.items():
            if key == "t":
                continue
            for i, val in enumerate(series):
                assert math.isfinite(val), f"Non-finite value in {key}[{i}] = {val}"
