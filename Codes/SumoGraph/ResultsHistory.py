from Codes.configuration import episode_time, traffic_light_period, Result_Path, load_object
import pandas as pd
import numpy as np
import os

class ResultsHistory:

    def __init__(self, is_Resumption):
        self.action_history = load_object("action_history", Result_Path) if is_Resumption and os.path.exists(os.path.join(Result_Path, "action_history.pkl")) else {}
        self.action_per_step = load_object("action_per_step", Result_Path) if is_Resumption and os.path.exists(os.path.join(Result_Path, "action_per_step.pkl")) else {}

        if not (is_Resumption and os.path.exists(os.path.join(Result_Path, "action_history.pkl"))):
            self.waiting_time_history_per_episode, self.waiting_time_history = {}, {}
            self.std_waiting_time_history_per_episode, self.std_waiting_time_history = {}, {}
            self.reward_history_per_episode, self.reward_history = {}, {}
            self.density_history_per_episode, self.density_history = {}, {}
        else:
            self.waiting_time_history_per_episode, self.waiting_time_history = load_object("waiting_time_history_per_episode", Result_Path), load_object("waiting_time_history", Result_Path)
            self.std_waiting_time_history_per_episode, self.std_waiting_time_history = load_object("std_waiting_time_history_per_episode", Result_Path), load_object("std_waiting_time_history", Result_Path)
            self.reward_history_per_episode, self.reward_history = load_object("reward_history_per_episode", Result_Path), load_object("reward_history", Result_Path)
            self.density_history_per_episode, self.density_history = load_object("density_history_per_episode", Result_Path), load_object("density_history", Result_Path) 
        # self.action_history_per_episode = {}
        
    def Actions_Save(self, action, edges):
        for i, edge in enumerate(edges):
            if not edge in self.action_history:self.action_history[edge] = [action[i]]
            else:self.action_history[edge].append(action[i])

    def Make_Results_Per_episode(self, methode_name):
        number_of_steps_per_episode = episode_time//traffic_light_period
        for key in self.reward_history.keys():
            for i in range(0,len(self.reward_history[key]),number_of_steps_per_episode):
                if not key in self.reward_history_per_episode.keys():
                    self.reward_history_per_episode[key] = [np.average(self.reward_history[key][i:i+number_of_steps_per_episode])]
                    self.density_history_per_episode[key] = [np.average(self.density_history[key][i:i+number_of_steps_per_episode])]
                    self.waiting_time_history_per_episode[key] = [np.average(self.waiting_time_history[key][i:i+number_of_steps_per_episode])]
                    self.std_waiting_time_history_per_episode[key] = [np.average(self.std_waiting_time_history[key][i:i+number_of_steps_per_episode])]
                else:
                    self.reward_history_per_episode[key].append(np.average(self.reward_history[key][i:i+number_of_steps_per_episode]))
                    self.density_history_per_episode[key].append(np.average(self.density_history[key][i:i+number_of_steps_per_episode]))
                    self.waiting_time_history_per_episode[key].append(np.average(self.waiting_time_history[key][i:i+number_of_steps_per_episode]))
                    self.std_waiting_time_history_per_episode[key].append(np.average(self.std_waiting_time_history[key][i:i+number_of_steps_per_episode]))

            df = pd.DataFrame()
            df['Reward'], df['Waiting Time'] = self.reward_history_per_episode[key], self.waiting_time_history_per_episode[key]
            df['STD_Waiting Time'], df['Density'] = self.std_waiting_time_history_per_episode[key], self.density_history_per_episode[key]
            save_path = os.path.join(Result_Path, str(methode_name) + ' Results')
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, "Numerical Results Per Episode " + key + ".csv"), index=False)

    def Save_Results_as_CSV(self, methode_name):
        for key in self.reward_history.keys():
            df = pd.DataFrame()
            df['Reward'], df['Waiting Time'] = self.reward_history[key], self.waiting_time_history[key]
            df['STD_Waiting Time'], df['Density'] = self.std_waiting_time_history[key], self.density_history[key]
            save_path = os.path.join(Result_Path, str(methode_name) + ' Results')
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, "Numerical Results Per Step " + key + ".csv"), index=False)