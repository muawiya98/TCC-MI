from Codes.configuration import traffic_light_period, Result_Path, save_object, load_object, number_of_episode_for_save
from Codes.MathematicalModels.Models import Models
from keras.models import load_model
from keras.layers import LeakyReLU
from Codes.RLUnit.DQN_Algorithm.DQN import DQN
from Codes.RLUnit.QLearning_Algorithm.QLearning import Qlearning
from collections import deque
import numpy as np
import os

import random
import numpy as np

random.seed(42)
np.random.seed(42)

import tensorflow as tf
@tf.function(experimental_relax_shapes=True)
def predict_function(model, input_tensor):
    return model(input_tensor)

class Agent:
    def __init__(self,agent_id, vstate, batch_size=32, max_len_queue=10000): # graph, reward, 
        self.dqnAlgo, self.action, self.reward, self.next_state, self.state = None, None, None, None, None
        self.max_len_queue = max_len_queue
        self.assign = False
        self.vstate = vstate
        self.QLearning = Qlearning()
        # self.graph = graph
        # self.Reward = reward
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_len_queue)
        # self.kalman_model = Models(self.graph, self.agent_id)
        self.loaded = False

    def rest_memory(self):
        self.memory = deque(maxlen=self.max_len_queue)
    
    def remember(self):
        if not self.state is None:
            self.memory.append((self.state, self.action, self.reward, self.next_state))


    def get_state(self):
        stat = self.vstate.state_function()
        maximum = max(stat)
        minimum = min(stat)
        threshold = ((maximum-minimum)/2)+minimum
        new_state = []
        for i, n in enumerate(stat):
            if n>=threshold:new_state.append(1)
            else:new_state.append(0)
        return np.array(new_state)
    
    
    def DQN_action(self, method, step, model_info_path, model_name, scenario_type, episode_number, is_Resumption):
        save_path_ = os.path.join(Result_Path, str(method)+' Results')
        # self.kalman_model.Run_Models(method)
        self.next_state = self.get_state() # self.vstate.state_function() # 
        if scenario_type=="train":
            if not self.assign:
                self.dqnAlgo = DQN(batch_size=self.batch_size, input_shape=self.next_state.shape[0])
                self.assign = True
            self.action = self.dqnAlgo.fit(self.next_state, model_info_path, model_name, episode_number, is_Resumption, save_path_)
        else:
            model = load_model(filepath=os.path.join(model_info_path, model_name + ".h5"))
            
            input_data = np.array(self.next_state.reshape(-1, self.next_state.size))
            input_tensor = tf.convert_to_tensor(input_data)
            pred = predict_function(model, input_tensor)
            self.action = np.argmax(pred[0])+1
        self.reward = self.vstate.reward_function() # self.Reward.Reward_Function(self.agent_id) # step , self.graph
        
        if is_Resumption and (not self.loaded):
            self.memory = load_object("memory "+str(self.agent_id), save_path_)
            self.loaded = True
        if scenario_type=="train":
            if step > traffic_light_period: self.replay_bufer(model_info_path, model_name, episode_number, is_Resumption, save_path_)
        self.state = self.next_state

        if episode_number%number_of_episode_for_save==0 and episode_number!=0:
            save_object(self.memory, "memory "+str(self.agent_id), Result_Path)
            save_object(self.memory, "memory "+str(self.agent_id), save_path_)
        return self.action # , self.graph

    def QLearning_action(self, step, Scenario_Type, method_name):
        self.next_state = self.get_state()
        if Scenario_Type=="train":q_value, self.action = self.QLearning.fit(self.next_state)
        else:q_value, self.action = self.QLearning.value_function(self.next_state)
        # self.Save(q_value, w_p)
        self.reward = self.vstate.reward_function()
        # if Scenario_Type=="train":
        if step > traffic_light_period:
        self.state = self.next_state
            bgu[' vrjo/OKi']
        return self.action
    
    # method, step, model_info_path, model_name, scenario_type, episode_number, is_Resumption
    def get_action(self, method, step, model_info_path, model_name, scenario_type, episode_number, is_Resumption):
        # return self.QLearning_action(step, scenario_type, method)
        return self.DQN_action(method, step, model_info_path, model_name, scenario_type, episode_number, is_Resumption)
    
    
    def replay_bufer(self, model_info_path, model_name, episode_number, is_Resumption, save_path_):
        self.remember()
        if len(self.memory) > self.batch_size:
            self.dqnAlgo.replay(self.memory, model_info_path, model_name, episode_number, is_Resumption, save_path_)