"""Summarize radiotherapy protocols selected by PPO models.

- Evaluates Normalized, Fract, and Mesh PPO checkpoints.
- Shows deterministic action sequences chosen by models.
- Saves summary as a CSV file.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

REPORT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORT_ROOT / "TumorGrowthEnv"))
import tumor_growth


MODEL_CONFIGS = {
    "normalized_25k": {
        "model": "Normalized PPO",
        "training_steps": 25_000,
        "environment_id": "tumor_growth:tumor-growth-normalized-ppo-v0",
        "path": REPORT_ROOT / Path(
            "baselines/results/models/normalized_ppo/normalized_ppo_20260817_203149_seed0/"
            "checkpoints/ppo_25000_steps.zip"
        ),
    },
    "fract_25k": {
        "model": "Fract PPO",
        "training_steps": 25_000,
        "environment_id": "tumor_growth:tumor-growth-fract-ppo-v0",
        "path": REPORT_ROOT / Path(
            "baselines/results/models/fract_ppo/fract_phase_mode_ox_ppo_20260816_164136_seed0/"
            "checkpoints/ppo_25000_steps.zip"
        ),
    },
    "mesh_25k": {
        "model": "Mesh PPO",
        "training_steps": 25_000,
        "environment_id": "tumor_growth:tumor-growth-mesh-ppo-v0",
        "path": REPORT_ROOT / Path(
            "baselines/results/models/mesh_ppo/mesh_phase_mode_ox_ppo_20260817_203656_seed0/"
            "checkpoints/ppo_25000_steps.zip"
        ),
    },
    "normalized_50k": {
        "model": "Normalized PPO",
        "training_steps": 50_000,
        "environment_id": "tumor_growth:tumor-growth-normalized-ppo-v0",
        "path": REPORT_ROOT / Path(
            "baselines/results/models/normalized_ppo/normalized_ppo_20260904_202531_seed0/"
            "checkpoints/ppo_50000_steps.zip"
        ),
    },
    "fract_50k": {
        "model": "Fract PPO",
        "training_steps": 50_000,
        "environment_id": "tumor_growth:tumor-growth-fract-ppo-v0",
        "path": REPORT_ROOT / Path(
            "baselines/results/models/fract_ppo/fract_phase_mode_ox_ppo_20260904_202539_seed0/"
            "checkpoints/ppo_50000_steps.zip"
        ),
    },
}

TUMOUR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EPISODES = 100
PARALLEL_RUNS = 10
DETERMINISTIC = True
EVALUATION_SEED = 0
OUTPUT_PATH = REPORT_ROOT / Path(
    "baselines/results/ppo_protocol_choice_frequencies_"
    "normalized_fract_mesh_25k_and_50k.csv"
)


def format_protocol(protocol: tuple) -> str:
    actions = [
        f"(hour={time_slot}, dose={dose_index * 0.5:.1f} Gy)"
        for time_slot, dose_index in protocol
    ]
    return " -> ".join(actions)


def run_episode(model: PPO, tumour_id: int, environment_id: str) -> tuple:
    env = gym.make(
        environment_id,
        tumors_list=[tumour_id],
        parallel_runs=PARALLEL_RUNS,
        with_time=True,
        dosing_hours=24,
        appointment_delays=False,
    )

    try:
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]

        protocol = []
        done = False

        for _ in range(5):
            action, _ = model.predict(
                observation,
                deterministic=DETERMINISTIC,
            )
            action = np.asarray(action, dtype=np.int64).ravel()
            protocol.append((int(action[0]), int(action[1])))

            result = env.step(action)
            if len(result) == 4:
                observation, _, done, _ = result
            else:
                observation, _, terminated, truncated, _ = result
                done = terminated or truncated

            if done:
                break

        if not done:
            raise RuntimeError("Episode did not finish within five actions")

        return tuple(protocol)
    finally:
        env.close()


def evaluate_model(config_name: str, config: dict) -> list:
    name = config["model"]
    training_steps = config["training_steps"]
    environment_id = config["environment_id"]
    model_path = config["path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nEvaluating {name} at {training_steps // 1000}k: {model_path}")
    model = PPO.load(model_path, device="cpu")
    print(f"Observation size: {model.observation_space.shape[0]}")
    print(f"Policy: {'deterministic' if DETERMINISTIC else 'stochastic'}")

    rng = np.random.default_rng(EVALUATION_SEED)
    tumour_order = []
    protocol_counts = Counter()

    for episode in range(EPISODES):
        if episode % len(TUMOUR_IDS) == 0:
            tumour_order = rng.permutation(TUMOUR_IDS).tolist()

        tumour_id = int(tumour_order[episode % len(TUMOUR_IDS)])
        protocol = run_episode(model, tumour_id, environment_id)
        protocol_counts[protocol] += 1
        print(
            f"Episode {episode + 1:>3}/{EPISODES}, tumour {tumour_id}: "
            f"{format_protocol(protocol)}"
        )

    print(
        f"\n{name} at {training_steps // 1000}k: "
        f"{len(protocol_counts)} unique protocols"
    )
    rows = []

    for rank, (protocol, count) in enumerate(
        protocol_counts.most_common(), start=1
    ):
        percentage = 100.0 * count / EPISODES
        readable_protocol = format_protocol(protocol)
        print(
            f"{rank:>3}. {count:>3}/{EPISODES} "
            f"({percentage:6.2f}%): {readable_protocol}"
        )
        rows.append(
            {
                "configuration": config_name,
                "model": name,
                "training_steps": training_steps,
                "rank": rank,
                "protocol": readable_protocol,
                "action_indices": json.dumps(protocol),
                "count": count,
                "percentage": percentage,
                "episodes": EPISODES,
                "parallel_runs": PARALLEL_RUNS,
                "tumour_ids": len(TUMOUR_IDS),
                "deterministic": DETERMINISTIC,
            }
        )

    return rows


def main() -> None:
    all_rows = []
    for config_name, config in MODEL_CONFIGS.items():
        all_rows.extend(evaluate_model(config_name, config))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved protocol frequencies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
