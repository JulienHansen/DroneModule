"""
Visual integration tests for CrazyfliePIDController.

Each test runs a closed-loop simulation, asserts numerical convergence,
and saves a plot to tests/plots/ (uploaded as CI artefact).

Simulation notes
────────────────
* ``rate_kd`` is zeroed for these tests.  The firmware value (2.5) is
  calibrated for real hardware where sensor noise is filtered by the
  physical plant; with exact forward-Euler integration it makes the
  closed-loop unstable (effective d-gain = kd/dt ≈ 1250).

* The position test uses a **small-angle** approximation:
    - orientation stays at identity (instant attitude tracking)
    - horizontal forces:  ax ≈ −pitch·g,  ay ≈ roll·g
    - vertical force:     az = thrust/m − g
  This avoids full attitude integration while keeping realistic
  translational dynamics.

* The body-rate test integrates ω directly (no full rotation).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from drone_control import load_config, CrazyfliePIDController

PLOTS_DIR = Path(__file__).parent / "plots"
DT   = 0.002
N    = 1
G    = 9.81


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_ctrl(cf_config) -> CrazyfliePIDController:
    ctrl = CrazyfliePIDController.from_drone_config(
        cf_config, num_envs=N, dt=DT, device="cpu"
    )
    # Zero rate derivative: the firmware value is tuned for filtered hardware
    # measurements; in exact simulation it causes closed-loop instability.
    ctrl.set_rate_gains(rate_kd=torch.zeros(3))
    return ctrl


def _hover_state(pos=None) -> torch.Tensor:
    """Root state [N, 13] at rest with identity orientation."""
    rs = torch.zeros(N, 13)
    rs[:, 3] = 1.0              # quat w = 1
    if pos is not None:
        rs[:, :3] = torch.as_tensor(pos, dtype=torch.float32)
    return rs


# ---------------------------------------------------------------------------
# Position step-response  (small-angle approximation)
# ---------------------------------------------------------------------------

class TestPositionStepResponse:
    """Fly from origin to [1, 0, 1] m; assert convergence within 8 s."""

    def test_converges_to_target(self, cf_config):
        ctrl = _make_ctrl(cf_config)
        mass = cf_config.physics.mass
        target = torch.tensor([[1.0, 0.0, 1.0]])

        pos = torch.zeros(N, 3)
        vel = torch.zeros(N, 3)
        # orientation stays at identity → quat & omega fixed
        quat  = torch.tensor([[1., 0., 0., 0.]]).repeat(N, 1)
        omega = torch.zeros(N, 3)

        history = {"t": [], "x": [], "z": [], "thrust": []}

        for i in range(int(10.0 / DT)):
            rs = torch.cat([pos, quat, vel, omega], dim=-1)
            thrust, _ = ctrl(rs, target_pos=target, command_level="position")

            # Small-angle force decomposition
            a = torch.zeros(N, 3)
            a[:, 2] = thrust[:, 0] / mass - G
            if ctrl._att_sp is not None:
                a[:, 0] = +ctrl._att_sp[:, 1] * G   # ax ≈ +pitch·g
                a[:, 1] = -ctrl._att_sp[:, 0] * G   # ay ≈ −roll·g

            vel = vel + a  * DT
            pos = pos + vel * DT

            history["t"].append(i * DT)
            history["x"].append(pos[0, 0].item())
            history["z"].append(pos[0, 2].item())
            history["thrust"].append(thrust[0, 0].item())

        settled = int(8.0 / DT)
        err = math.hypot(history["x"][settled] - 1.0,
                         history["z"][settled] - 1.0)
        assert err < 0.1, f"Position not converged after 8 s: err={err:.3f} m"
        assert all(math.isfinite(v) for v in history["x"])
        assert all(math.isfinite(v) for v in history["z"])

        self._plot(history, target[0].tolist())

    @staticmethod
    def _plot(h, target):
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(h["t"], h["x"], label="x"); ax1.plot(h["t"], h["z"], label="z")
        ax1.axhline(target[0], color="C0", ls="--", lw=0.8)
        ax1.axhline(target[2], color="C1", ls="--", lw=0.8)
        ax1.set_ylabel("Position [m]"); ax1.legend(); ax1.grid(True)
        ax1.set_title("Position step response — CrazyfliePIDController")

        ax2.plot(h["t"], h["thrust"])
        ax2.axhline(0.027 * 9.81, color="gray", ls="--", lw=0.8, label="m·g")
        ax2.set_ylabel("Thrust [N]"); ax2.set_xlabel("Time [s]")
        ax2.legend(); ax2.grid(True)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "position_step_response.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Body-rate step response
# ---------------------------------------------------------------------------

class TestAttitudeStepResponse:
    """Command 2 rad/s roll rate; assert tracking within 0.5 s."""

    def test_roll_rate_tracks_setpoint(self, cf_config):
        ctrl = _make_ctrl(cf_config)
        J = torch.tensor([
            cf_config.physics.inertia.ixx,
            cf_config.physics.inertia.iyy,
            cf_config.physics.inertia.izz,
        ])

        omega = torch.zeros(N, 3)
        thrust_cmd = torch.full((N, 1), 30_000.0)
        rate_sp    = torch.tensor([[2.0, 0.0, 0.0]]).repeat(N, 1)

        history = {"t": [], "omega_x": [], "omega_y": []}

        for i in range(int(1.0 / DT)):
            rs = _hover_state()
            rs[:, 10:13] = omega          # inject measured body rates

            _, moment = ctrl(
                rs,
                target_body_rates=rate_sp,
                thrust_cmd=thrust_cmd,
                command_level="body_rate",
                body_rates_in_body_frame=True,
            )

            # Integrate: α = τ / J  (diagonal inertia, no gyro coupling needed here)
            omega = omega + (moment / J) * DT

            history["t"].append(i * DT)
            history["omega_x"].append(omega[0, 0].item())
            history["omega_y"].append(omega[0, 1].item())

        settled = int(0.5 / DT)
        roll_err = abs(history["omega_x"][settled] - 2.0)
        assert roll_err < 0.3, f"Roll rate not tracking: err={roll_err:.3f} rad/s"
        assert all(math.isfinite(v) for v in history["omega_x"])

        self._plot(history, target_roll_rate=2.0)

    @staticmethod
    def _plot(h, target_roll_rate):
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["t"], h["omega_x"], label="ω_x (roll)")
        ax.plot(h["t"], h["omega_y"], label="ω_y (pitch)")
        ax.axhline(target_roll_rate, color="C0", ls="--", lw=0.8, label="setpoint")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Body rate [rad/s]")
        ax.set_title("Body-rate step response — CrazyfliePIDController")
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "attitude_step_response.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Yaw-rate integration
# ---------------------------------------------------------------------------

class TestYawRateIntegration:
    """Command 1 rad/s yaw; verify the setpoint wraps within [−π, π]."""

    def test_yaw_rotates_without_nan(self, cf_config):
        ctrl = _make_ctrl(cf_config)
        target   = torch.zeros(N, 3); target[:, 2] = 0.5
        yaw_rate = torch.full((N, 1), 1.0)

        history = {"t": [], "yaw_sp": []}

        # Two full rotations
        for i in range(int(4.0 * math.pi / DT)):
            rs = _hover_state(pos=[0., 0., 0.5])
            ctrl(rs, target_pos=target, target_yaw_rate=yaw_rate,
                 command_level="position")
            history["t"].append(i * DT)
            history["yaw_sp"].append(ctrl._yaw_sp[0, 0].item())

        assert all(math.isfinite(v) for v in history["yaw_sp"]), \
            "NaN in yaw setpoint"
        assert all(-math.pi - 1e-3 <= v <= math.pi + 1e-3
                   for v in history["yaw_sp"]), "Yaw escaped [−π, π]"

        self._plot(history)

    @staticmethod
    def _plot(h):
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["t"], h["yaw_sp"])
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Yaw setpoint [rad]")
        ax.set_title("Yaw-rate integration — CrazyfliePIDController")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "yaw_rate_integration.png", dpi=120)
        plt.close(fig)
