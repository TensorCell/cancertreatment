"""Define the legacy EMT6-Ro environment used by Paper PPO.

- Reproduces the discrete observation space and original reward formulation.
- Simulates the report's radiation actions with optional appointment delays.
- Remains separate from the normalized PPO environment profiles.
"""

import emt6ro.simulation as sim
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import os

class TumorGrowthEnv(gym.Env):
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
                gpu_id=0):  
        super(TumorGrowthEnv, self).__init__()

        MODES = [None, '2doses']
        REW = [None, 'kill_prob']
        if not mode in MODES:
            raise ValueError(f"Mode must be one of {MODES}!!")
        if not reward_type in REW:
            raise ValueError(f"Reward must be one of {REW}!!")
        self.mode = mode
        self.with_time = with_time
        self.reward_type = reward_type
        self.appointment_delays = appointment_delays
        self.dosing_hours = dosing_hours

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
        

        if with_time:
            self.observation_space = spaces.MultiDiscrete([1500]+[10])  
        else:
            self.observation_space = spaces.MultiDiscrete([1500])  
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
        if self.with_time:
            day = self.time / 24 / 600  
            state = np.array([int(np.mean(self.tumor_cells))]+[day])
        else:
            state = np.array([int(np.mean(self.tumor_cells))])
        return state, self.reward, done, info

    def reset(self):
        self.experiment.reset()
        
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()[0]
        self.start_cells = np.mean(self.tumor_cells)
        self.reward = 0
        self.time = 0
        self.cumulative_dose = 0
        if self.with_time:
            state = np.array([int(np.mean(self.tumor_cells))]+[0])
        else:
            state = np.array([int(np.mean(self.tumor_cells))])
        return state

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
        if self.reward_type is None:
            self.reward = self.start_cells - np.mean(self.tumor_cells) if promotion==False \
                else self.promotion * (self.start_cells - np.mean(self.tumor_cells))
        elif self.reward_type == 'kill_prob':
            self.reward = np.sum(self.tumor_cells.flatten() == 0) if promotion == False \
                else self.promotion * np.sum(self.tumor_cells.flatten() == 0)
