"""Define the EMT6-Ro environment used by CNN Mesh PPO.

- Builds normalized scalar observations plus one-hot spatial tumour maps.
- Simulates fractionated radiation with optional treatment delays.
- Preserves the training and evaluation dynamics used by the report model.
"""

import emt6ro.simulation as sim
import json
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import os

TICKS_PER_MINUTE = 10.0
TREATMENT_DELAY_MEAN_MINUTES = 34.4
TREATMENT_DELAY_STD_MINUTES = 46.3

class TumorGrowthEnv(gym.Env):
    def _get_cnn_state_fields(self):
        states = self.experiment.state()
        occupancy_stack = np.stack(
            [np.asarray(state.occupancy()) for state in states]
        )
        ox_stack = np.stack([np.asarray(state.ox()) for state in states])
        mode_stack = np.stack([np.asarray(state.mode()) for state in states])
        phase_stack = np.stack([np.asarray(state.phase()) for state in states])

        occupied = occupancy_stack == 1

        mode_counts = np.stack(
            [np.sum((mode_stack == value) & occupied, axis=0) for value in range(4)]
        )
        mode = np.argmax(mode_counts, axis=0).astype(np.int64)
        mode[np.sum(occupied, axis=0) == 0] = -1

        phase_counts = np.stack(
            [np.sum((phase_stack == value) & occupied, axis=0) for value in range(5)]
        )
        phase = np.argmax(phase_counts, axis=0).astype(np.int64)
        phase[np.sum(occupied, axis=0) == 0] = -1

        return {
            "phase_one_hot": np.eye(6, dtype=np.float32)[phase + 1],
            "mode_one_hot": np.eye(5, dtype=np.float32)[mode + 1],
            "ox": np.mean(ox_stack, axis=0).astype(np.float32),
        }

    def __init__(self,
                params_filename: str = None,
                tumors_list = None,
                parallel_runs: int = 10,
                dosing_hours: int = 24,
                treatment_delay: bool = False,
                with_time: bool = False,
                gpu_id=0):
        super(TumorGrowthEnv, self).__init__()

        self.with_time = with_time
        self.treatment_delay = treatment_delay
        self.dosing_hours = dosing_hours

        if not isinstance(treatment_delay, (bool, np.bool_)):
            raise TypeError("treatment_delay must be True or False")
        if dosing_hours < 1 or dosing_hours > 24:
            raise ValueError("dosing_hours must be between 1 and 24")

        if params_filename is None:
            params_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/default-parameters.json")
        with open(params_filename) as params_file:
            params_json = json.load(params_file)
        self.external_oxygen = float(params_json["external_levels"]["ox"])
        if self.external_oxygen <= 0:
            raise ValueError("External oxygen level must be positive")
        if tumors_list is None:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in range(1, 11)]
        else:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in tumors_list]




        params = sim.load_parameters(params_filename)
        tumors = [sim.load_state(tumors, params) for tumors in tumors_list]
        self.experiment = sim.Experiment(params,
                                    tumors,
                                    parallel_runs,
                                    1,
                                    gpu_id=gpu_id)

        self.experiment.run(0)

        self.tumor_cells = self.experiment.get_results()[0]
        self.time = 0
        self.cumulative_dose = 0
        self.dosing_hours = dosing_hours

        self.start_cells = np.mean(self.tumor_cells)
        self.reward = 0

        current_params = self._get_cnn_state_fields()
        observation_low_parts = [np.asarray([0], dtype=np.float32)]
        observation_high_parts = [np.asarray([1], dtype=np.float32)]
        if self.with_time:
            observation_low_parts.append(np.asarray([0], dtype=np.float32))
            observation_high_parts.append(np.asarray([1], dtype=np.float32))
        observation_low_parts.append(np.asarray([0], dtype=np.float32))
        observation_high_parts.append(np.asarray([1], dtype=np.float32))
        for name, upper_bound in (
            ("phase_one_hot", 1),
            ("mode_one_hot", 1),
            ("ox", self.external_oxygen),
        ):
            size = current_params[name].size
            observation_low_parts.append(np.zeros(size, dtype=np.float32))
            observation_high_parts.append(
                np.full(size, upper_bound, dtype=np.float32)
            )
        observation_low = np.concatenate(observation_low_parts)
        observation_high = np.concatenate(observation_high_parts)
        self.observation_space = spaces.Box(
            observation_low, observation_high, dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.dosing_hours, 11])

    def step(self, action):
        done, info = self._step_default(action)
        state = self._get_state()
        return state, self.reward, done, info

    def reset(self):
        self.experiment.reset()

        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()[0]
        self.start_cells = np.mean(self.tumor_cells)
        self.reward = 0
        self.time = 0
        self.cumulative_dose = 0
        state = self._get_state()
        return state

    def _get_state(self):
        cells = np.clip(float(np.mean(self.tumor_cells)) / 1500.0, 0.0, 1.0)
        values = [np.asarray([cells], dtype=np.float32)]
        if self.with_time:
            day = np.clip(self.time / (10 * 24 * 600), 0.0, 1.0)
            values.append(np.asarray([day], dtype=np.float32))
        dose = np.clip(self.cumulative_dose / 10.0, 0.0, 1.0)
        values.append(np.asarray([dose], dtype=np.float32))
        current_params = self._get_cnn_state_fields()
        values.extend(
            np.asarray(current_params[name], dtype=np.float32).ravel()
            for name in ("phase_one_hot", "mode_one_hot", "ox")
        )
        return np.concatenate(values)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if hasattr(self, "action_space"):
            self.action_space.seed(seed)
        return [seed]

    def get_treatment_delay(self):
        if not self.treatment_delay:
            return 0
        delay_ticks = self.np_random.normal(
            loc=TREATMENT_DELAY_MEAN_MINUTES * TICKS_PER_MINUTE,
            scale=TREATMENT_DELAY_STD_MINUTES * TICKS_PER_MINUTE,
        )
        while delay_ticks < 0:
            delay_ticks = self.np_random.normal(
                loc=TREATMENT_DELAY_MEAN_MINUTES * TICKS_PER_MINUTE,
                scale=TREATMENT_DELAY_STD_MINUTES * TICKS_PER_MINUTE,
            )
        return int(round(delay_ticks))

    def get_num_cells(self):
        return float(np.mean(self.tumor_cells))

    def _step_default(self, action):
        (delay, dose) = action

        dose = 0.5 * dose

        dose = min(dose, 10 - self.cumulative_dose)

        self.cumulative_dose += dose

        treatment_delay = self.get_treatment_delay()

        translated_action = (self.time + min(delay * 600 + treatment_delay, self.dosing_hours * 600 - 1), dose)
        self.experiment.add_irradiations([[translated_action]])

        dosing_hour = (translated_action[0] / 600) % 24
        dosing_day = (translated_action[0]) // (24 * 600)

        self.experiment.run(24 * 600)
        self.time += 24 * 600
        self.tumor_cells = self.experiment.get_results()[0]
        cells_after_treatment_step = float(np.mean(self.tumor_cells))
        treatment_step_end_day = self.time / (24 * 600)

        done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10 or np.all(self.tumor_cells == 0))

        if done:
            leftover_time = 10 * 24 * 600 - self.time
            self.experiment.run(leftover_time)
            self.tumor_cells = self.experiment.get_results()[0]
            self.time = 10 * 24 * 600
            self._update_reward(final_step=True)
        else:
            self._update_reward()
        info = {"delay [h]": delay , "dose [Gy]": dose,
                "treatment_delay [h]": treatment_delay / 600.0,
                "day [d]": dosing_day,
                "hour [h]": dosing_hour,
                "cumulative_dose": self.cumulative_dose,
                "cells_after_treatment_step": cells_after_treatment_step,
                "treatment_step_end_day": treatment_step_end_day,
                "evaluation_day": self.time / (24 * 600),
                "leftover_cells": np.mean(self.tumor_cells), "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _update_reward(self, final_step=False):
        if final_step:
            self.reward = -float(np.mean(self.tumor_cells)) / 1500.0
        else:
            self.reward = 0.0
