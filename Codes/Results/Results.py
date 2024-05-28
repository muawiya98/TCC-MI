# from Codes.Visualization.KalmanVisualization import KalmanVisualization
from Codes.Visualization.ResultsVisualization import ResultsVisualization
from Codes.configuration import Methods, Result_Path, save_object, load_object
import pandas as pd
import numpy as np
import pickle
import math
import os

class Results:
    
    def __init__(self, logger):
        self.logger = logger
        # self.kalman_visualization = KalmanVisualization()
        self.Results_visualization = ResultsVisualization()
    
    
    def calculate_rmse(self,y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        squared_errors = (y_true - y_pred) ** 2
        mean_squared_error = np.mean(squared_errors)
        rmse = math.sqrt(mean_squared_error)
        return rmse
    
    # def Kalman_Results(self, methode_name, methods_names = ['Ground Truth', 'Kalman Filter', 'Measurement',
    #                                                         'Smooth Kalman Filter', 'Particle Filter', 'Smooth Particle Filter']):
    #     save_path = os.path.join(Result_Path, str(methode_name)+' Results')
    #     os.makedirs(save_path, exist_ok=True)
    #     results = self.graph.models_history.get_history()
    #     Kalman_rmse, Measurement_rmse, Edge, Kalman_Smooth_rmse, Particle_Filter, \
    #     Particle_soomth_Filter = [], [], [], [], [], []
    #     RMSE_df = pd.DataFrame()
    #     for key in results.keys():
    #         df = pd.DataFrame()
    #         Edge.append("ID :" + key)
    #         Kalman_rmse.append(self.calculate_rmse(results[key][0], results[key][1]))
    #         Measurement_rmse.append(self.calculate_rmse(results[key][0], results[key][2]))
    #         # Kalman_Smooth_rmse.append(self.calculate_rmse(results[key][0], results[key][3]))
    #         # Particle_Filter.append(self.calculate_rmse(results[key][0], results[key][4]))
    #         # Particle_soomth_Filter.append(self.calculate_rmse(results[key][0], results[key][5]))

    #         df[methods_names[0]] = results[key][0] # Ground_Truth
    #         df[methods_names[2]] = results[key][2] # Measurement
    #         df[methods_names[1]] = results[key][1] # Kalman_Filter
    #         # df[methods_names[3]] = results[key][3] # Smooth_Kalman_Filter
    #         # df[methods_names[4]] = results[key][4] # Particle_Filter
    #         # df[methods_names[5]] = results[key][5]# Smooth_Particle_Filter
    #         methods_names1 = ['Ground Truth', 'Measurement', 'Kalman Filter'] #, 'Smooth Kalman Filter'] ,results[key][3]]
    #         self.kalman_visualization.vehicles_plot(key+" Kalman",[results[key][0],results[key][2],results[key][1]], name_folder=str(methode_name)+' Results', methods_name=methods_names1)
    #         # methods_names2 = ['Ground Truth', 'Measurement', 'Particle Filter', 'Smooth Particle Filter']
    #         # self.kalman_visualization.vehicles_plot(key+" Particle", [results[key][0],results[key][2],results[key][4],results[key][5]], name_folder=str(methode_name)+' Results', methods_name=methods_names2)
    #         for i, _ in enumerate(results[key]):
    #             if not results[key][i] == []:
    #                 results[key][i] = np.array(results[key][i])/max(results[key][i])
    #             else:results[key][i] = [0]

    #         edge_info_for_debug = [results[key][0]]
    #         methods_names_ = ['Ground Truth']
    #         for edge_id in self.graph.results_history.action_history.keys():
    #             if key == edge_id:
    #                 actions = self.graph.results_history.action_history[key]
    #                 edge_info_for_debug.append(np.array(actions)/10)
    #                 df["Actions"] = actions
    #                 methods_names_.append('Action for Edge ' + key)
    #             try:
    #                 Incomming_edges = self.graph.Incomming_edges[key]
    #                 for ed in Incomming_edges:
    #                     if ed == edge_id:
    #                         actions = self.graph.results_history.action_history[ed]
    #                         edge_info_for_debug.append(np.array(actions)/10)
    #                         methods_names_.append('Action for Edge ' + ed)
    #                         df["Actions for Edge "+ed] = actions
    #             except:pass
    #         df.to_csv(os.path.join(save_path, key + ".csv"), index=False)
    #         self.kalman_visualization.vehicles_plot(key+"_with_action", edge_info_for_debug, name_folder= str(methode_name)+' Results', methods_name=methods_names_,step=1)

    #     RMSE_df["Edge ID"] = Edge
    #     RMSE_df["Measurement RMSE"] = Measurement_rmse
    #     RMSE_df["Kalman RMSE"] = Kalman_rmse
    #     # RMSE_df["Smooth_Kalman RMSE"] = Kalman_Smooth_rmse
    #     # RMSE_df["Particle_Filter RMSE"] = Particle_Filter
    #     # RMSE_df["Smooth_Particle_Filter RMSE"] = Particle_soomth_Filter
    #     RMSE_df.to_csv(os.path.join(save_path, "RMSE.csv"), index=False)

    def RL_Results(self, methode_name):
        save_path = os.path.join(Result_Path, str(methode_name)+' Results')
        os.makedirs(save_path, exist_ok=True)
        junction_ids = list(self.logger.reward_history.keys())
        for junction_id in junction_ids:
            self.Results_visualization.Results_plot(junction_id, self.logger.waiting_time_history_per_episode[junction_id], "AVG Waiting Time", "Episode", "AVG Waiting Time", str(methode_name)+' Results', 1)
            self.Results_visualization.Results_plot(junction_id, self.logger.std_waiting_time_history_per_episode[junction_id], "STD Waiting Time", "Episode", "STD Waiting Time", str(methode_name)+' Results', 1)
            self.Results_visualization.Results_plot(junction_id, self.logger.reward_history_per_episode[junction_id], "Reward", "Episode", "Reward", str(methode_name)+' Results', 1)

    def Prepare_All_Results(self, methode_name):
        # if not methode_name is Methods.Random:
        #     self.Kalman_Results(methode_name)
        self.logger.Save_Results_as_CSV(methode_name)
        self.logger.Make_Results_Per_episode(methode_name)
        self.RL_Results(methode_name)