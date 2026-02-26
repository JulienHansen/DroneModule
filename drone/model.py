"""
Drone — rigid-body digital twin.

Connects a controller, a force model list, and a numerical integrator into a
single object that advances the drone state by one timestep.

Usage
-----
>>> from drone import load_config, Drone
>>> from drone.controllers import LeePositionController
>>> from drone.forces import Gravity, BodyDrag
>>>
>>> cfg  = load_config("configs/crazyflie.yaml")
>>> ctrl = LeePositionController.from_drone_config(cfg, num_envs=4)
>>> drone = Drone.from_drone_config(cfg, controller=ctrl, num_envs=4)
>>>
>>> state = drone.hover_state(num_envs=4)   # [4, 13] at origin, level
>>> target = torch.tensor([[0., 0., 1.]]).repeat(4, 1)
>>>
>>> for _ in range(500):   # 500 × 2 ms = 1 s
...     state = drone.step(state, dt=0.002, target_pos=target,
...                        target_vel=None, target_acc=None,
...                        target_yaw=None, target_yaw_rate=None)
"""

from __future__ import annotations

from typing import Optional

import torch

from .config.loader import DroneConfig
from .forces.base import ForceModel
from .forces.gravity import Gravity
from .integrators import Integrator, EulerIntegrator
from .utils.math_utils import matrix_from_quat
from .utils.mixer import QuadMixer


# ---------------------------------------------------------------------------
# Quaternion kinematics helper
# ---------------------------------------------------------------------------

def _quat_derivative(
    quat: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """
    Quaternion kinematic equation: q̇ = ½ · q ⊗ [0, ω]

    Parameters
    ----------
    quat  : ``[N, 4]``  unit quaternion  [w, x, y, z]
    omega : ``[N, 3]``  body angular velocity [rad/s]

    Returns
    -------
    ``[N, 4]``  time derivative of the quaternion.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]

    dw = -0.5 * (x * ox + y * oy + z * oz)
    dx =  0.5 * (w * ox + y * oz - z * oy)
    dy =  0.5 * (w * oy + z * ox - x * oz)
    dz =  0.5 * (w * oz + x * oy - y * ox)

    return torch.stack([dw, dx, dy, dz], dim=-1)  # [N, 4]


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

class Drone:
    """
    Rigid-body digital twin of a quadrotor drone.

    Combines a feedback **controller**, a list of **force models**, and a
    **numerical integrator** to simulate the full sensor-to-actuator loop
    at each timestep.

    The simulation step follows a Zero-Order Hold (ZOH) scheme:

    1. Compute ``(thrust, moment)`` from the controller **once** per step.
    2. Accumulate all external forces and torques.
    3. Integrate rigid-body Newton-Euler dynamics.

    Parameters
    ----------
    config : DroneConfig
        Loaded drone configuration.
    controller : callable
        Any object with ``__call__(state, **kwargs) -> (thrust [N,1], moment [N,3])``.
        Compatible with :class:`~drone.controllers.CascadePIDController` and
        :class:`~drone.controllers.LeePositionController`.
    forces : list[ForceModel] or None
        External force/torque models.  Defaults to ``[Gravity(mass)]`` if
        ``None`` — the minimum required for a physical simulation.
    integrator : Integrator or None
        Numerical integrator.  Defaults to :class:`~drone.integrators.EulerIntegrator`.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        config: DroneConfig,
        controller,
        forces: Optional[list[ForceModel]] = None,
        integrator: Optional[Integrator] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.controller = controller
        self.device = torch.device(device)

        # Force models — gravity is mandatory for any physical simulation
        self.forces: list[ForceModel] = (
            forces if forces is not None
            else [Gravity.from_drone_config(config, device=device)]
        )

        # Integrator
        self.integrator: Integrator = integrator or EulerIntegrator()

        # Motor mixer (optional — only if drone.motor is in the YAML)
        self.mixer: Optional[QuadMixer] = (
            QuadMixer.from_drone_config(config, device=device)
            if config.physics.motor is not None else None
        )

        # ── Physics tensors ──────────────────────────────────────────────
        self._mass = float(config.physics.mass)

        diag = torch.tensor(
            [config.physics.inertia.ixx,
             config.physics.inertia.iyy,
             config.physics.inertia.izz],
            dtype=torch.float32,
            device=self.device,
        )
        self._J     = torch.diag(diag).unsqueeze(0)            # [1, 3, 3]
        self._J_inv = torch.linalg.inv(self._J)                # [1, 3, 3]

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_drone_config(
        cls,
        config: DroneConfig,
        controller,
        forces: Optional[list[ForceModel]] = None,
        integrator: Optional[Integrator] = None,
        device: str = "cpu",
    ) -> "Drone":
        """
        Build a :class:`Drone` from a :class:`~drone.config.loader.DroneConfig`.

        Example
        -------
        >>> cfg   = load_config("configs/crazyflie.yaml")
        >>> ctrl  = LeePositionController.from_drone_config(cfg, num_envs=4)
        >>> drone = Drone.from_drone_config(cfg, controller=ctrl)
        """
        return cls(config, controller, forces, integrator, device)

    # ── Convenience constructors ─────────────────────────────────────────

    @staticmethod
    def hover_state(num_envs: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Return a ``[num_envs, 13]`` state tensor at the origin, level and at rest.

        Useful as a starting point for simulation.
        """
        state = torch.zeros(num_envs, 13, device=device)
        state[:, 3] = 1.0   # quaternion w = 1  (identity rotation)
        return state

    # ── Device management ────────────────────────────────────────────────

    def to(self, device: str) -> "Drone":
        """Move all internal tensors to *device* in-place. Returns ``self``."""
        self.device = torch.device(device)
        self._J     = self._J.to(device)
        self._J_inv = self._J_inv.to(device)
        for f in self.forces:
            f.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
        return self

    # ── Main simulation step ─────────────────────────────────────────────

    def step(
        self,
        state: torch.Tensor,
        dt: float,
        **ctrl_kwargs,
    ) -> torch.Tensor:
        """
        Advance the drone state by one timestep.

        Parameters
        ----------
        state : ``[N, 13]``
            Current state: ``[pos(3) | quat(4) | lin_vel(3) | ang_vel(3)]``.
        dt : float
            Simulation timestep [s].
        **ctrl_kwargs
            Keyword arguments forwarded to the controller
            (e.g. ``target_pos``, ``command_level``, ``target_yaw`` …).

        Returns
        -------
        state_next : ``[N, 13]``

        Example
        -------
        >>> # CascadePIDController
        >>> state = drone.step(state, dt=0.002,
        ...                    target_pos=sp, command_level="position")
        >>>
        >>> # LeePositionController
        >>> state = drone.step(state, dt=0.002,
        ...                    target_pos=sp, target_vel=None,
        ...                    target_acc=None, target_yaw=None,
        ...                    target_yaw_rate=None)
        """
        state = state.to(device=self.device, dtype=torch.float32)

        # 1. Control — computed once (ZOH: held constant for the whole step)
        thrust, moment = self.controller(state, **ctrl_kwargs)

        # 2. Integrate dynamics with the fixed control input
        def deriv_fn(s: torch.Tensor) -> torch.Tensor:
            return self._derivatives(s, thrust, moment)

        return self.integrator.step(state, deriv_fn, dt)

    def motor_speeds(
        self,
        thrust: torch.Tensor,
        moment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert collective thrust + body moments to per-motor speeds [rad/s].

        Requires ``drone.motor`` to be present in the YAML config.

        Parameters
        ----------
        thrust : ``[N, 1]``  collective thrust [N]
        moment : ``[N, 3]``  body moments [N·m]

        Returns
        -------
        ``[N, 4]``  motor angular speeds [rad/s]
        """
        if self.mixer is None:
            raise ValueError(
                "No motor config found. Add a 'drone.motor' section to your "
                "YAML config to use motor_speeds()."
            )
        return self.mixer(thrust, moment)

    # ── Rigid-body derivatives ────────────────────────────────────────────

    def _derivatives(
        self,
        state: torch.Tensor,
        thrust: torch.Tensor,
        moment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full state derivative ``[N, 13]`` for Newton-Euler dynamics.

        Layout of the returned tensor (mirrors the state layout):
            ``[ vel(3) | quat_dot(4) | lin_acc(3) | ang_acc(3) ]``
        """
        pos   = state[:, 0:3]    # world frame   (unused in derivative, kept for clarity)
        quat  = state[:, 3:7]    # [w, x, y, z]
        vel   = state[:, 7:10]   # world frame
        omega = state[:, 10:13]  # body frame

        N = state.shape[0]
        dev = self.device

        # ── External forces (world) and torques (body) ────────────────────
        F_ext   = torch.zeros(N, 3, device=dev)
        tau_ext = torch.zeros(N, 3, device=dev)
        for force_model in self.forces:
            F_i, tau_i = force_model.compute(state)
            F_ext   = F_ext   + F_i
            tau_ext = tau_ext + tau_i

        # ── Thrust along body-z rotated to world frame ────────────────────
        R       = matrix_from_quat(quat)           # [N, 3, 3]
        body_z  = R[:, :, 2]                       # [N, 3]  — third column
        F_thrust = body_z * thrust                 # [N, 3]  (thrust is [N,1] → broadcasts)

        # ── Total force and torque ─────────────────────────────────────────
        F_total   = F_thrust + F_ext               # [N, 3]  world frame
        tau_total = moment   + tau_ext             # [N, 3]  body frame

        # ── Linear acceleration  a = F / m  (world frame) ─────────────────
        lin_acc = F_total / self._mass             # [N, 3]

        # ── Angular acceleration  dω/dt = J⁻¹ (τ − ω × Jω)  (body frame) ─
        J     = self._J
        J_inv = self._J_inv
        if J.shape[0] == 1 and N > 1:
            J     = J.expand(N, -1, -1)
            J_inv = J_inv.expand(N, -1, -1)

        Jw      = torch.bmm(J, omega.unsqueeze(-1)).squeeze(-1)         # [N, 3]
        gyro    = torch.linalg.cross(omega, Jw, dim=-1)                 # ω × Jω
        ang_acc = torch.bmm(
            J_inv,
            (tau_total - gyro).unsqueeze(-1),
        ).squeeze(-1)                                                    # [N, 3]

        # ── Quaternion derivative  q̇ = ½ q ⊗ [0, ω] ──────────────────────
        quat_dot = _quat_derivative(quat, omega)   # [N, 4]

        # ── Pack state_dot [N, 13] ─────────────────────────────────────────
        return torch.cat([vel, quat_dot, lin_acc, ang_acc], dim=-1)
