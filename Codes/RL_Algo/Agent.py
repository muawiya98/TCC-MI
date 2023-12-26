from Codes.configuration import traffic_light_period
from Codes.MathematicalModels.Models import Models
from keras.models import load_model
from keras.layers import LeakyReLU
from Codes.RL_Algo.DQN import DQN
from collections import deque
import numpy as np
import os
class Agent:
    def __init__(self,agent_id, graph, reward, batch_size=32, max_len_queue=10000):
        self.dqnAlgo, self.action, self.reward, self.next_state, self.state = None, None, None, None, None
        self.assign = False
        self.graph = graph
        self.Reward = reward
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_len_queue)
        self.kalman_model = Models(self.graph, self.agent_id)

    def remember(self):
        self.memory.append((self.state, self.action, self.reward, self.next_state))

    def get_action(self, method, step, callbacks_list, model_info_path, model_name, Scenario_Type):
        self.kalman_model.Run_Models(method)
        self.next_state = self.graph.get_RL_State(self.agent_id)
        if Scenario_Type=="train":
            if not self.assign:
                self.dqnAlgo = DQN(batch_size=self.batch_size, input_shape=self.next_state.shape[0])
                self.assign = True
            self.action = self.dqnAlgo.fit(self.next_state)
        else:
            custom_objects = {'LeakyReLU': LeakyReLU}
            model = load_model(filepath=os.path.join(model_info_path, model_name + ".h5"), custom_objects=custom_objects)
            self.action = np.argmax(model.predict(self.next_state.reshape(-1, self.next_state.size), verbose=0)[0])+1
        self.reward = self.Reward.Reward_Function(self.agent_id, step)
        if Scenario_Type=="train":
            if step > traffic_light_period: self.replay_bufer(callbacks_list, model_info_path, model_name)
        self.state = self.next_state
        return self.action

    def replay_bufer(self, callbacks_list, model_info_path, model_name):
        self.remember()
        if len(self.memory) > self.batch_size:
            self.dqnAlgo.replay(self.memory, callbacks_list, model_info_path, model_name)