from Codes.configuration import Network_Path, Methods, traffic_light_period, \
    generation_period, episode_time, Simulation_Time, TEST_STAGE, Result_Path
from Codes.TrafficLightController.TrafficLightsControler import TrafficLightsController
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from Codes.ObjectsController.SumoController import SumoObjectController
from Codes.InformationProvider.InformationGeter import Infromation
from warnings import simplefilter,filterwarnings
from Codes.Results.Results import Results
from Codes.RL_Algo.Reward import Reward
from Codes.SumoGraph.Graph import Graph
from Codes.RL_Algo.Agent import Agent
import matplotlib.pyplot as plt # type: ignore
from numpy import random
from traci import trafficlight
import traci
import os
import gc
filterwarnings('ignore')
filterwarnings(action='once')
simplefilter('ignore', FutureWarning)

class GarbageCollectorCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

class Controller:
    def __init__(self, intersection):
        self.number_of_agent = len(intersection)
        self.tls_controller = TrafficLightsController(intersection)
        self.Agent_ids = intersection
        self.graph = Graph(intersection)
        self.results = Results(self.graph)
        self.SumoObject = SumoObjectController(self.graph.incomming_edges, self.graph.outcomming_edges)
        self.information = Infromation(self.Agent_ids)
        self.reward = Reward(self.information, self.graph, self.Agent_ids)
        self.Agents = []
    def Create_Agents(self, method_name):
        callbacks_lists, model_info_paths, model_names = [], [], []
        for Agent_id in self.Agent_ids:
            self.Agents.append(Agent(Agent_id, self.graph, self.reward))
            callbacks_list, model_info_path, model_name = self.Make_Callbacks(method_name, Agent_id)
            callbacks_lists.append(callbacks_list)
            model_info_paths.append(model_info_path)
            model_names.append(model_name)
        return callbacks_lists, model_info_paths, model_names

    def Save_Start_State(self):
        path_start_state = Network_Path
        path_0 = path_start_state.split('.')[0]
        path_start_state = path_0+'_start_state.xml'
        traci.simulation.saveState(path_start_state)
    def Load_Start_State(self):
        path_start_state = Network_Path
        path_0 = path_start_state.split('.')[0]
        path_start_state = path_0+'_start_state.xml'
        traci.simulation.loadState(path_start_state)
    def Rest_Sumo(self):
        self.Load_Start_State()
    def Maping_Between_agents_junctions(self, actions):
        self.tls_controller.send_actions_tls(actions)
        self.tls_controller.check_tls_cmds()
    def Save_Actions_For_Edge(self):
        for i, Agent_id in enumerate(self.Agent_ids):
            edges = self.graph.Junction_controlledEdge[Agent_id]
            list_action = []
            for edge_id in edges:
                lanes = self.graph.Edge_lane[edge_id]
                controlled_lanes = trafficlight.getControlledLanes(Agent_id)
                lane_index = list(controlled_lanes).index(lanes[len(lanes) // 2])
                edge_state = trafficlight.getRedYellowGreenState(Agent_id)[lane_index]
                edge_state = self.graph.lane_state[edge_state]
                list_action.append(edge_state)
            self.graph.results_history.Actions_Save(list_action, edges)

    def Make_Callbacks(self, method_name, Agent_id):
        model_name = str(method_name) + "_model" + Agent_id
        save_path = os.path.join(Result_Path, str(method_name)+' Results')
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, model_name + '.best.hdf5')
        history_logger = CSVLogger(os.path.join(save_path, model_name + '_history.csv'), separator=",", append=True)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', patience=5)
        callbacks_list = [GarbageCollectorCallback(), checkpoint, es, history_logger]
        return callbacks_list, save_path, model_name

    def Communication_With_Environment(self, method_name, step, callbacks_lists, model_info_paths, model_names, Scenario_Type):
        Actions_dic = {}
        for i, Agent_id in enumerate(self.Agent_ids):
            if method_name is Methods.Random:
                action = random.choice([1, 2, 3, 4, 5, 6])
                Actions_dic[Agent_id] = action
                self.reward.Reward_Function(Agent_id, step)
            else:
                # if Agent_id == 'J0':
                Actions_dic[Agent_id] = self.Agents[i].get_action(method_name, step, callbacks_lists[i],
                                                                  model_info_paths[i], model_names[i], Scenario_Type)
                # else:
                #     action = random.choice([1, 2, 3, 4, 5, 6])
                #     Actions_dic[Agent_id] = action
        self.Maping_Between_agents_junctions(Actions_dic)
        self.Save_Actions_For_Edge()

    def Run(self, methode_name):
        os.makedirs(Result_Path, exist_ok=True)
        step_generation, step, sub_episode_number, episode_number = 0, 0, 0, 0
        print(methode_name)
        callbacks_lists, model_info_paths, model_names = self.Create_Agents(methode_name)
        Scenario_Type = "train"
        while step < Simulation_Time:
            traci.simulationStep()
            if episode_number>=TEST_STAGE:Scenario_Type = "test"
            if step == 0: self.Save_Start_State()
            if step % traffic_light_period == 0: self.Communication_With_Environment(methode_name, step, callbacks_lists, model_info_paths, model_names, Scenario_Type)
            if step_generation % generation_period == 0:
                self.SumoObject.generate_object(sub_episode_number)
                sub_episode_number += 1
                step_generation = 0
            if (step+generation_period) % episode_time == 0 and step != 0:
                sub_episode_number = 0
                episode_number += 1
                self.Rest_Sumo()
            step_generation += 1
            step += 1
        self.results.Prepare_All_Results(methode_name)
