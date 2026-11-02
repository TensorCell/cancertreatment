"""Train the CNN Mesh PPO model.

- Starts with a new model and trains it for 50,000 steps.
- Uses a convolutional neural network to read the tumour grid information.
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
from cnn_mesh_policy import CNNMeshExtractor

REPORT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORT_ROOT / "TumorGrowthEnv"))
import tumor_growth

from stable_baselines3.common.callbacks import BaseCallback


TUMOUR_IDS = tuple(range(1, 11))
ENVIRONMENT_ID = "tumor_growth:tumor-growth-cnn-mesh-ppo-v0"
ENVIRONMENT_SOURCE = "tumor_growth_env_cnn_mesh_ppo.py"
OBSERVATION_ATTRIBUTES = (  
    "normalized_tumor_cells",
    "normalized_time",
    "normalized_cumulative_dose",
    "phase_one_hot_mesh (no cell, G1, S, G2, M, D)",
    "mode_one_hot_mesh (no cell, 0, 1, 2, 3)",
    "oxygen_mesh",
)
DIRECTORY_NAME = "cnn_mesh_ppo"

class TrainingOutputCallback(BaseCallback):
    def __init__(self, print_every=100):
        super().__init__()
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.locals["dones"][0] or self.num_timesteps % self.print_every == 0:  
            info = self.locals["infos"][0]

            print(
                f"step={self.num_timesteps}, "
                f"observation={self.locals['new_obs'][0]}, "
                f"action={self.locals['actions'][0]}, "
                f"reward={self.locals['rewards'][0]:.2f}, "
                f"leftover_cells={info.get('leftover_cells')}, "
                f"done={self.locals['dones'][0]}"
            )

        return True
        
class PPOTrainer:
    def __init__(
        self,
        total_timesteps: int = 50_000,
        batch_size: int = 40,
        parallel_runs: int = 10,
        ent_coef = 0.001,
        rollout_steps = 1000,
        seed = 0,
        checkpoint_every: int = 1_000,
        output_dir: Path = REPORT_ROOT / "baselines/results/models/cnn_mesh_ppo",
        device: str = "auto",
    ):
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.parallel_runs = parallel_runs
        self.ent_coef = ent_coef  
        self.seed = seed
        self.checkpoint_every = checkpoint_every
        self.output_dir = output_dir
        self.device = device
        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO] = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_dir / f"{DIRECTORY_NAME}_{timestamp}_seed{seed}"

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
        return PPO(
            "CnnPolicy",
            self.env,
            policy_kwargs={
                "features_extractor_class": CNNMeshExtractor,
                "normalize_images": False,
                "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            },
            n_steps=self.rollout_steps,
            batch_size=self.batch_size,
            ent_coef=self.ent_coef,  
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
        if isinstance(initial_observation, tuple):  
            initial_observation = initial_observation[0]  

        print(f"Current observation space: {self.env.observation_space}")  
        print("Observation attributes:")  
        for name in OBSERVATION_ATTRIBUTES:  
            print(f"  {name}")  
        print("Example initial state:")  
        print(np.array2string(initial_observation, threshold=30))  

        initial_cells = float(self.env.unwrapped.start_cells)
        if initial_cells <= 0:
            self.env.close()
            raise RuntimeError("EMT6/Ro returned zero cells at reset")

        settings = {
            "model_name": "CNN Mesh PPO",
            "environment_id": ENVIRONMENT_ID,
            "environment_source": ENVIRONMENT_SOURCE,
            "features_extractor": "cnn_mesh_policy.CNNMeshExtractor",
            "total_timesteps": self.total_timesteps,
            "rollout_steps": self.rollout_steps,  
            "batch_size": self.batch_size,
            "parallel_runs": self.parallel_runs,
            "cnn_state_fields": ["phase_one_hot", "mode_one_hot", "ox"],
            "ent_coef": self.ent_coef,  
            "seed": self.seed,  
            "checkpoint_every": self.checkpoint_every,
            "resume": None,
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
            output_callback = TrainingOutputCallback(print_every=100)
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=[checkpoint_callback, output_callback],
                reset_num_timesteps=True,
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
