"""Train the Paper PPO model.

- Starts with a new model and trains it for 25,000 steps.
- Uses the original environment created for Paper PPO.
- Saves the settings, training logs, a model every 1,000 steps, and the final model.
"""

from __future__ import annotations

import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

REPORT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORT_ROOT / "TumorGrowthEnv"))
import tumor_growth


TUMOUR_IDS = tuple(range(1, 11))
ENVIRONMENT_ID = "tumor_growth:tumor-growth-paper-ppo-v0"
ENVIRONMENT_SOURCE = "tumor_growth_env_paper_ppo.py"

class PPOTrainer:
    def __init__(
        self,
        total_timesteps: int = 25_000,
        rollout_steps: int = 200,
        batch_size: int = 40,
        parallel_runs: int = 10,
        seed: int = 0,
        checkpoint_every: int = 1_000,
        resume: Optional[Path] = None,
        output_dir: Path = REPORT_ROOT / "baselines/results/models/paper_ppo",
        device: str = "cpu",
    ):
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.parallel_runs = parallel_runs
        self.seed = seed
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.output_dir = output_dir
        self.device = device
        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO] = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_dir / f"paper_ppo_{timestamp}_seed{seed}"

    def validate_args(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("--total-timesteps must be positive")
        if self.rollout_steps < 2:
            raise ValueError("--rollout-steps must be at least 2")
        if self.batch_size < 2 or self.rollout_steps % self.batch_size:
            raise ValueError("--batch-size must divide --rollout-steps")
        if self.parallel_runs <= 0:
            raise ValueError("--parallel-runs must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("--checkpoint-every must be positive")
        if self.resume is not None and not self.resume.is_file():
            raise ValueError(f"Checkpoint not found: {self.resume}")

    def save_model(self, name: str) -> None:
        if self.model is None:
            return
        path = self.run_dir / name
        self.model.save(path)
        print(f"Model saved to {path.with_suffix('.zip')}")

    def safe_save(self, name: str) -> None:
        try:
            self.save_model(name)
        except Exception as error:
            print(f"Could not save model: {error}")

    def make_env(self, tumour_ids: list[int]) -> gym.Env:
        env = gym.make(
            ENVIRONMENT_ID,
            tumors_list=tumour_ids,
            parallel_runs=self.parallel_runs,
            with_time=True,
            dosing_hours=24,
            appointment_delays=False,
        )

        env.unwrapped.np_random = np.random.default_rng(self.seed)
        env.action_space.seed(self.seed)
        return env

    def make_model(self) -> PPO:
        if self.resume is not None:
            print(f"Resuming from {self.resume}")
            return PPO.load(
                self.resume,
                env=self.env,
                device=self.device,
                n_steps=self.rollout_steps,
                batch_size=self.batch_size,
            )

        return PPO(
            "MlpPolicy",
            self.env,
            n_steps=self.rollout_steps,
            batch_size=self.batch_size,
            seed=self.seed,
            verbose=1,
            device=self.device,
        )

    def train(self) -> None:
        self.validate_args()
        self.run_dir.mkdir(parents=True, exist_ok=False)

        rng = np.random.default_rng(self.seed)
        tumour_ids = [int(value) for value in rng.permutation(TUMOUR_IDS)]
        self.env = self.make_env(tumour_ids)

        initial_observation = self.env.reset()
        initial_cells = float(self.env.unwrapped.start_cells)
        if initial_cells <= 0:
            self.env.close()
            raise RuntimeError("EMT6/Ro returned zero cells at reset")

        settings = {
            "environment_id": ENVIRONMENT_ID,
            "environment_source": ENVIRONMENT_SOURCE,
            "total_timesteps": self.total_timesteps,
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "parallel_runs": self.parallel_runs,
            "seed": self.seed,
            "checkpoint_every": self.checkpoint_every,
            "resume": str(self.resume) if self.resume else None,
            "output_dir": str(self.output_dir),
            "device": self.device,
            "selected_tumour_ids": tumour_ids,
            "initial_observation": np.asarray(initial_observation).tolist(),
            "initial_mean_cells": initial_cells,
            "simulations_per_step": len(tumour_ids) * self.parallel_runs,
        }
        (self.run_dir / "settings.json").write_text(
            json.dumps(settings, indent=2, default=str)
        )

        self.model = self.make_model()
        self.model.set_random_seed(self.seed)
        self.model.set_logger(configure(str(self.run_dir), ["stdout", "csv"]))

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_every,
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
        )

        stop_signal: Optional[str] = None
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(signum, _frame):
            nonlocal stop_signal
            stop_signal = signal.Signals(signum).name
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, handle_sigterm)
        try:
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=checkpoint_callback,
                reset_num_timesteps=self.resume is None,
                progress_bar=True,
            )
            self.save_model("ppo_completed")
        except KeyboardInterrupt:
            print(f"Training stopped by {stop_signal or 'Ctrl+C'}")
            self.safe_save("ppo_interrupted")
        except Exception:
            self.safe_save("ppo_error")
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            self.env.close()


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
