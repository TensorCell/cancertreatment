"""Unit tests for nhits.tuning.search_space.

These tests verify the structure of the search-space dict without actually
running a Ray Tune experiment.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_HP_KEYS = {
    "lr",
    "weight_decay",
    "batch_size",
    "hidden_size",
    "num_mlp_layers",
    "dropout",
    "loss",
    "activation",
    "margin_loss",
    "margin_loss_w",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDefaultSearchSpace:
    """Tests for ``default_search_space()``."""

    @pytest.fixture(autouse=True)
    def _space(self):
        from nhits.tuning.search_space import default_search_space
        self.space = default_search_space()

    def test_returns_dict(self):
        assert isinstance(self.space, dict)

    def test_contains_expected_keys(self):
        assert EXPECTED_HP_KEYS <= set(self.space.keys()), (
            f"Missing keys: {EXPECTED_HP_KEYS - set(self.space.keys())}"
        )

    def test_no_unexpected_keys(self):
        unexpected = set(self.space.keys()) - EXPECTED_HP_KEYS
        assert not unexpected, f"Unexpected keys found: {unexpected}"

    def test_lr_is_loguniform(self):
        """lr should be a continuous log-uniform distribution."""
        from ray.tune.search.sample import Float
        assert isinstance(self.space["lr"], Float)

    def test_batch_size_is_categorical(self):
        """batch_size should be a categorical choice."""
        from ray.tune.search.sample import Categorical
        assert isinstance(self.space["batch_size"], Categorical)

    def test_loss_choices(self):
        from ray.tune.search.sample import Categorical
        hp = self.space["loss"]
        assert isinstance(hp, Categorical)
        assert set(hp.categories) == {"mae", "mse", "huber"}

    def test_activation_choices(self):
        from ray.tune.search.sample import Categorical
        hp = self.space["activation"]
        assert isinstance(hp, Categorical)
        assert set(hp.categories) == {"relu", "gelu"}

    def test_margin_loss_choices(self):
        from ray.tune.search.sample import Categorical
        hp = self.space["margin_loss"]
        assert isinstance(hp, Categorical)
        assert set(hp.categories) == {True, False}

    def test_margin_loss_w_is_float_uniform(self):
        from ray.tune.search.sample import Float
        hp = self.space["margin_loss_w"]
        assert isinstance(hp, Float)

    def test_dropout_is_float_uniform(self):
        from ray.tune.search.sample import Float
        hp = self.space["dropout"]
        assert isinstance(hp, Float)

    def test_search_space_is_deterministic(self):
        """Calling the factory twice returns equivalent dicts."""
        from nhits.tuning.search_space import default_search_space
        space2 = default_search_space()
        assert set(self.space.keys()) == set(space2.keys())
