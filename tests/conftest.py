"""Shared test fixtures: one small generated world for the whole test run."""

import pytest

from shopper_twin.data import load_public
from shopper_twin.world import generate_world, get_config


@pytest.fixture(scope="session")
def tiny_world(tmp_path_factory):
    """A 1,000-shopper, 1,000-product world with 90 days (long enough for a 28-day window and its history)."""
    path = generate_world(get_config("tiny", n_days=90), root=tmp_path_factory.mktemp("worlds"), verbose=False)
    return load_public(path)
