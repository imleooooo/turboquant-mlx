import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_weight(rng):
    """Small (32, 64) float32 weight matrix for fast tests."""
    return rng.standard_normal((32, 64)).astype(np.float32)


@pytest.fixture
def medium_weight(rng):
    """Medium (128, 256) float32 weight matrix."""
    return rng.standard_normal((128, 256)).astype(np.float32)
