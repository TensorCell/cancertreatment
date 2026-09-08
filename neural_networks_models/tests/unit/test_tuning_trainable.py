"""Unit / smoke tests for nhits.tuning.trainable.

Runs ``tune_nhits`` with a minimal config (1 epoch, tiny model) entirely
in-process via ``ray.tune.run`` with ``num_samples=1``, no W&B, no GPU.

The test ensures:
* ``tune_nhits`` completes without raising.
* ``val_loss`` is reported as a finite float.
* Per-trial checkpoint directories are isolated (no ``checkpoints/nhits/``
  collision with the main training pipeline).
"""

from __future__ import annotations

import math


class TestTuneNhitsSmoke:
    """Smoke-test for the tune_nhits trainable."""

    def test_single_trial_completes(self, tmp_path):
        """tune_nhits should complete and report a finite val_loss."""
        import ray  # noqa: PLC0415
        from ray import tune  # noqa: PLC0415

        from nhits.tuning.trainable import tune_nhits  # noqa: PLC0415

        # Minimal config: tiny model, 1 epoch, synthetic data, CPU only.
        trial_config = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "hidden_size": 64,
            "num_mlp_layers": 2,
            "dropout": 0.0,
            "loss": "mae",
            "activation": "relu",
            "max_epochs": 1,
            "patience": 1,
            "data_path": None,
            "seed": 0,
        }

        if not ray.is_initialized():
            ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)

        try:
            tuner = tune.Tuner(
                tune.with_resources(tune_nhits, resources={"cpu": 1, "gpu": 0}),
                param_space=trial_config,
                tune_config=tune.TuneConfig(num_samples=1),
                run_config=tune.RunConfig(
                    storage_path=str(tmp_path),
                    name="test_single_trial",
                    verbose=0,
                ),
            )
            results = tuner.fit()
        finally:
            ray.shutdown()

        assert len(results) == 1, "Expected exactly one trial result."
        result = results[0]
        assert result.error is None, f"Trial raised an error: {result.error}"

        val_loss = result.metrics.get("val_loss")
        assert val_loss is not None, "val_loss not reported by trial."
        assert math.isfinite(val_loss), f"val_loss is not finite: {val_loss}"
        assert val_loss > 0, f"val_loss should be positive, got {val_loss}"

    def test_checkpoint_dir_isolation(self, tmp_path):
        """Each trial uses its own checkpoint dir, not the global one."""
        import os  # noqa: PLC0415
        import ray  # noqa: PLC0415
        from ray import tune  # noqa: PLC0415

        from nhits.tuning.trainable import tune_nhits  # noqa: PLC0415

        trial_config = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "hidden_size": 64,
            "num_mlp_layers": 2,
            "dropout": 0.0,
            "loss": "mae",
            "activation": "relu",
            "max_epochs": 1,
            "patience": 1,
            "data_path": None,
            "seed": 0,
        }

        if not ray.is_initialized():
            ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)

        try:
            tuner = tune.Tuner(
                tune.with_resources(tune_nhits, resources={"cpu": 1, "gpu": 0}),
                param_space=trial_config,
                tune_config=tune.TuneConfig(num_samples=1),
                run_config=tune.RunConfig(
                    storage_path=str(tmp_path),
                    name="test_isolation",
                    verbose=0,
                ),
            )
            tuner.fit()
        finally:
            ray.shutdown()

        # The global nhits checkpoint dir should NOT be created by the trial
        # (it uses checkpoints/tune/<trial_id>/ instead).
        assert not (tmp_path / "checkpoints" / "nhits").exists(), (
            "Trial wrote to global checkpoint dir instead of per-trial dir."
        )
