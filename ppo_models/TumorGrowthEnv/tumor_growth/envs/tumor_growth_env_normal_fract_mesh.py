"""Define the shared EMT6-Ro environment for three PPO observation profiles.

- ``normalized`` returns the three normalized scalar state variables.
- ``fract`` adds phase, mode, and hypoxia fractions; ``mesh`` adds raw maps.
- Radiation timing and optional appointment delays use the report's original logic.
"""

import emt6ro.simulation as sim
import json
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import os
from typing import Optional

PPO_ADDITIONAL_STATE = {
    "normalized": (),
    "fract": (
        "phase_fractions",
        "mode_fractions",
        "hypoxia_fractions",
    ),
    "mesh": (
        "phase",
        "mode",
        "ox",
    ),
}

class TumorGrowthEnv(gym.Env):
    def show_all_params(self):
        states = self.experiment.state()  
        occupancy_stack = np.stack(
            [np.asarray(state.occupancy()) for state in states]
        )  
        irradiation_stack = np.stack(
            [np.asarray(state.irradiation()) for state in states]
        )  
        ox_stack = np.stack([np.asarray(state.ox()) for state in states])  
        mode_stack = np.stack([np.asarray(state.mode()) for state in states])  
        phase_stack = np.stack([np.asarray(state.phase()) for state in states])  

        occupied = occupancy_stack == 1  
        irradiation_stack = np.where(occupied, irradiation_stack, 0.0)  

        occupancy_counts = np.stack(
            [np.sum(occupancy_stack == value, axis=0) for value in range(3)]
        )  
        occupancy = np.argmax(occupancy_counts, axis=0).astype(np.float32)  

        mode_counts = np.stack(
            [np.sum((mode_stack == value) & occupied, axis=0) for value in range(4)]
        )  
        mode = np.argmax(mode_counts, axis=0).astype(np.float32)  
        mode[np.sum(occupied, axis=0) == 0] = -1  

        phase_counts = np.stack(
            [np.sum((phase_stack == value) & occupied, axis=0) for value in range(5)]
        )  
        phase = np.argmax(phase_counts, axis=0).astype(np.float32)  
        phase[np.sum(occupied, axis=0) == 0] = -1  

        living_cells = int(np.sum(occupied))  
        if living_cells > 0:  
            mode_fractions = np.bincount(
                mode_stack[occupied].astype(np.int64), minlength=4
            ).astype(np.float32) / living_cells  
            phase_fractions = np.bincount(
                phase_stack[occupied].astype(np.int64), minlength=5
            ).astype(np.float32) / living_cells  
        else:  
            mode_fractions = np.zeros(4, dtype=np.float32)  
            phase_fractions = np.zeros(5, dtype=np.float32)  

        occupied_oxygen = ox_stack[occupied]
        if occupied_oxygen.size:
            hypoxia_fractions = np.asarray([  
                np.mean(occupied_oxygen > 2.25),  
                np.mean((occupied_oxygen > 1.35) & (occupied_oxygen <= 2.25)),  
                np.mean(occupied_oxygen <= 1.35),  
            ], dtype=np.float32)  
        else:
            hypoxia_fractions = np.zeros(3, dtype=np.float32)  

        return {
            "occupancy": occupancy,
            "irradiation": np.mean(irradiation_stack, axis=0),  
            "ox": np.mean(ox_stack, axis=0),  
            "mode": mode,
            "phase": phase,
            "mode_fractions": mode_fractions,  
            "phase_fractions": phase_fractions,  
            "hypoxia_fractions": hypoxia_fractions,  
        }

    def __init__(self,
                reward_type = None, 
                mode = None,  
                params_filename: str = None,
                tumors_list = None,  
                
                parallel_runs: int = 10,  
                promotion: int = 100,  
                dosing_hours: int = 24,
                appointment_delays: bool = False,
                with_time: bool = False,
                ppo_type: str = "normalized",
                additional_state: Optional[tuple] = None,
                gpu_id=0):  
        super(TumorGrowthEnv, self).__init__()

        available_state_fields = (
            "occupancy", "irradiation", "ox", "mode", "phase",
            "mode_fractions", "phase_fractions", "hypoxia_fractions",
        )  
        MODES = [None, '2doses']
        REW = [None, 'kill_prob']
        if not mode in MODES:
            raise ValueError(f"Mode must be one of {MODES}!!")
        if not reward_type in REW:
            raise ValueError(f"Reward must be one of {REW}!!")
        if ppo_type not in PPO_ADDITIONAL_STATE:
            raise ValueError(
                f"ppo_type must be one of {tuple(PPO_ADDITIONAL_STATE)}"
            )
        profile_state = PPO_ADDITIONAL_STATE[ppo_type]
        if additional_state is None:
            additional_state = profile_state
        else:
            additional_state = tuple(additional_state)
            if additional_state != profile_state:
                raise ValueError(
                    f"additional_state for ppo_type={ppo_type!r} must be "
                    f"{profile_state}, got {additional_state}"
                )
        self.mode = mode
        self.with_time = with_time
        self.reward_type = reward_type
        self.appointment_delays = appointment_delays
        self.dosing_hours = dosing_hours
        self.ppo_type = ppo_type
        self.additional_state = additional_state

        unknown_state_fields = set(additional_state) - set(available_state_fields)
        if unknown_state_fields:
            raise ValueError(f"Unknown additional_state fields: {sorted(unknown_state_fields)}")
        

        if not isinstance(appointment_delays, (bool, np.bool_)):
            raise TypeError("appointment_delays must be True or False")
        if dosing_hours < 1 or dosing_hours > 24:
            raise ValueError("dosing_hours must be between 1 and 24")

        self.open_hour = False
        if dosing_hours <= 14:
            self.open_hour = 7
        elif dosing_hours <= 17:
            self.open_hour = 6
        elif dosing_hours == 18:
            self.open_hour = 5
        else:
            self.open_hour = 24 - self.dosing_hours

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
        self.promotion = promotion
        

        observation_low_parts = [np.asarray([0], dtype=np.float32)]  
        observation_high_parts = [np.asarray([1], dtype=np.float32)]  
        if self.with_time:  
            observation_low_parts.append(np.asarray([0], dtype=np.float32))  
            observation_high_parts.append(np.asarray([1], dtype=np.float32))  
        observation_low_parts.append(np.asarray([0], dtype=np.float32))  
        observation_high_parts.append(np.asarray([1], dtype=np.float32))  
        if self.additional_state:  
            current_params = self.show_all_params()  
            field_bounds = {  
                "occupancy": (0, 2),  
                "mode": (-1, 3),  
                "phase": (-1, 4),  
                "mode_fractions": (0, 1),  
                "phase_fractions": (0, 1),  
                "hypoxia_fractions": (0, 1),  
                "ox": (0, self.external_oxygen),  
            }  
            for name in self.additional_state:  
                size = current_params[name].size  
                low = -np.inf
                high = np.inf
                if name in field_bounds:
                    low, high = field_bounds[name]
                observation_low_parts.append(np.full(size, low, dtype=np.float32))  
                observation_high_parts.append(np.full(size, high, dtype=np.float32))  
        observation_low = np.concatenate(observation_low_parts)  
        observation_high = np.concatenate(observation_high_parts)  
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)  
        
        if self.mode == '2doses':
            self.action_space = spaces.MultiDiscrete([self.dosing_hours, self.dosing_hours, 
                                                      11,  
                                                      3])  
        else:
            self.action_space = spaces.MultiDiscrete([self.dosing_hours,  
                                                 11])  

    def step(self, action):
        if self.mode is None:
            done, info = self._step_default(action)
        elif self.mode == '2doses':
            done, info = self._step_2doses(action)
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
        if self.additional_state:  
            current_params = self.show_all_params()  
            values.extend(np.asarray(current_params[name], dtype=np.float32).ravel() for name in self.additional_state)  
        return np.concatenate(values)  

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if hasattr(self, "action_space"):
            self.action_space.seed(seed)
        return [seed]

    def get_app_delay(self):
        if not self.appointment_delays:
            return 0
        delay_ticks = self.np_random.normal(loc=250.0, scale=620.0)
        while delay_ticks < 0:
            delay_ticks = self.np_random.normal(loc=250.0, scale=620.0)
        return int(round(delay_ticks))

    def get_num_cells(self):
        return float(np.mean(self.tumor_cells))

    def _step_default(self, action):
        (delay, dose) = action

        dose = 0.5 * dose
        
        dose = min(dose, 10 - self.cumulative_dose)

        self.cumulative_dose += dose

        app_delay = self.get_app_delay()

        translated_action = (self.time + min(delay * 600 + app_delay, self.dosing_hours * 600 - 1), dose)
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
            self._update_reward(promotion=True)
        else:
            self._update_reward()
        info = {"delay [h]": delay , "dose [Gy]": dose,
                "app_delay [h]": app_delay / 600.0,
                "day [d]": dosing_day,
                "hour [h]": dosing_hour,
                "cumulative_dose": self.cumulative_dose,
                "cells_after_treatment_step": cells_after_treatment_step,  
                "treatment_step_end_day": treatment_step_end_day,  
                "evaluation_day": self.time / (24 * 600),  
                "leftover_cells": np.mean(self.tumor_cells), "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info    

    def _step_2doses(self, action):
        (delay1, delay2, dose, division) = action

        dose = 0.5 * dose
        dose = min(dose, 10 - self.cumulative_dose)

        self.cumulative_dose += dose

        dose1 = dose * division / 4
        dose2 = dose - dose1

        app_delay1 = self.get_app_delay()
        app_delay2 = self.get_app_delay()

        translated_action = [(self.time + min(delay1 * 600 + app_delay1, self.dosing_hours * 600 - 1), dose1), (self.time + min(delay2 * 600 + app_delay2, self.dosing_hours * 600 - 1), dose2)]
        self.experiment.add_irradiations([translated_action])  

        dosing_hour_1 = (translated_action[0][0] / 600) % 24
        dosing_hour_2 = (translated_action[1][0] / 600) % 24
        dosing_day = (translated_action[0][0]) // (24 * 600)

        self.experiment.run(24 * 600)  
        self.time += 24 * 600
        self.tumor_cells = self.experiment.get_results()[0]
        
        done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10 or np.all(self.tumor_cells == 0))
        
        if done:
            leftover_time = 10 * 24 * 600 - self.time
            self.experiment.run(leftover_time)
            self.tumor_cells = self.experiment.get_results()[0]
            self.time = 10 * 24 * 600
            self._update_reward(promotion=True)
        else:
            self._update_reward()

        info = {"delay1": delay1, "delay2": delay2, "dose1": dose1, "dose2": dose2,
                "app_delay1 [h]": app_delay1 / 600.0, "app_delay2 [h]": app_delay2 / 600.0,
                "day [d]": dosing_day, "hour1 [h]": dosing_hour_1, "hour2 [h]": dosing_hour_2,
                "cumulative_dose": self.cumulative_dose, "leftover_cells": np.mean(self.tumor_cells),
                "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _update_reward(self, promotion=False):  
        if promotion:  
            self.reward = -float(np.mean(self.tumor_cells)) / 1500.0  
        else:  
            self.reward = 0.0  
