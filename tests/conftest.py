"""
Shared fixtures for the drone test suite.

All path resolution is relative to the repository root so tests can be
invoked from any working directory (repo root, tests/, etc.).
"""

from pathlib import Path

import pytest
import torch

from drone import load_config

# Absolute path to the repo root, independent of cwd
REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Config fixtures  (session-scoped: loaded once for the whole test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cf_config():
    """DroneConfig for the Crazyflie 2.1."""
    return load_config(CONFIGS_DIR / "crazyflie.yaml")


@pytest.fixture(scope="session")
def gq_config():
    """DroneConfig for the generic 250 mm 5-inch quad."""
    return load_config(CONFIGS_DIR / "generic_quad_250mm.yaml")


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def num_envs():
    return 3
