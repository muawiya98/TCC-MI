from Codes.configuration import traffic_light_period, WINDOW_SIZE, \
    episode_time, Weighting_Factor
import numpy as np
from math import log
class Reward:
    def __init__(self, information, graph, Agent_ids):
        self.information = information
        self.graph = graph
        self.previous_waiting_time = {key: 0 for key in Agent_ids}

    def Save(self, agent_id, waiting_time, std_waiting_time, reward):
        density = self.information.Average_Density_Vehicles(agent_id)
        if not agent_id in self.graph.results_history.waiting_time_history.keys():
            self.graph.results_history.waiting_time_history[agent_id] = [waiting_time]
            self.graph.results_history.std_waiting_time_history[agent_id] = [std_waiting_time]
            self.graph.results_history.reward_history[agent_id] = [reward]
            self.graph.results_history.density_history[agent_id] = [density]
        else:
            self.graph.results_history.waiting_time_history[agent_id].append(waiting_time)
            self.graph.results_history.std_waiting_time_history[agent_id].append(std_waiting_time)
            self.graph.results_history.reward_history[agent_id].append(reward)
            self.graph.results_history.density_history[agent_id].append(density)
        return self.graph

    # def Short_Term_Reward(self, Agent_id, vehicles=None):
    #     if not Agent_id in self.graph.results_history.waiting_time_history.keys():
    #         p_waiting_time = self.previous_waiting_time[Agent_id]
    #     else:
    #         p_waiting_time = self.graph.results_history.waiting_time_history[Agent_id][-1]
    #     edges = self.graph.Junction_controlledEdge[Agent_id]
    #     waiting_time, std_waiting_time = self.information.Reward_Info(edges)
    #     reward = p_waiting_time - waiting_time
    #     self.previous_waiting_time[Agent_id] = waiting_time
    #     if not Agent_id in self.graph.results_history.waiting_time_history.keys():
    #         self.graph.results_history.waiting_time_history[Agent_id] = [waiting_time]
    #     else:
    #         self.graph.results_history.waiting_time_history[Agent_id].append(waiting_time)
    #     return np.tanh(reward)
    # def Long_Term_Reward(self, Agent_id, vehicles=None):
    #     # waiting_time = self.information.Average_Waiting_Time_Vehicles(Agent_id, vehicles)
    #     # self.graph.waiting_time_history[Agent_id].append(waiting_time)
    #     reward = np.average(self.graph.results_history.waiting_time_history[Agent_id][-WINDOW_SIZE:])
    #     # self.graph.set_waiting_time_history(Agent_id)
    #     return reward
    # def Reward_Function_(self, Agent_id, step, vehicles=None):
    #     # if step + traffic_light_period == (episode_time * ((step + traffic_light_period) // episode_time)):
    #     reward_1 = self.Short_Term_Reward(Agent_id, vehicles)
    #     if (step % (episode_time+1)) // traffic_light_period >= WINDOW_SIZE:
    #         reward_2 = self.Long_Term_Reward(Agent_id, vehicles)
    #         reward = (Weighting_Factor * reward_1) + ((1-Weighting_Factor) * reward_2)
    #     else:reward = reward_1
    #     density = self.information.Average_Density_Vehicles(Agent_id)
    #     if not Agent_id in self.graph.results_history.reward_history.keys():
    #         self.graph.results_history.reward_history[Agent_id] = [reward]
    #         self.graph.results_history.accumulative_reward_history[Agent_id] = [reward]
    #         self.graph.results_history.density_history[Agent_id] = [density]
    #     else:
    #         self.graph.results_history.reward_history[Agent_id].append(reward)
    #         self.graph.results_history.accumulative_reward_history[Agent_id].append(sum(self.graph.results_history.reward_history[Agent_id]))
    #         self.graph.results_history.density_history[Agent_id].append(density)
    #     return reward

    def Reward_Function(self, agent_id):
        edges = self.graph.Junction_controlledEdge[agent_id]
        waiting_time, std_waiting_time = self.information.Reward_Info(edges)
        reward = log((Weighting_Factor*std_waiting_time) + ((1-Weighting_Factor)*(0.9**waiting_time)), 0.5)
        self.Save(agent_id, waiting_time, std_waiting_time, reward)
        return reward, self.graph