from Codes.configuration import NUMBER_OF_ACTION, load_object, save_object, number_of_episode_for_save, Result_Path
from Codes.RLUnit.RLAlgorithm import RLAlgorithm
from keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.activations import relu
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
import os

class DQN(RLAlgorithm):
    def __init__(self, input_shape, epochs=10, neurons_num:list = [32, 16], lr=0.01, gamma=0.9,epsilon=0.99, decay=0.00001, min_epsilon=0.01, batch_size=32):
        super().__init__(epochs, lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DQN"
        self.batch_size = batch_size
        self.neurons_num_list = neurons_num
        self.input_shape = input_shape
        self.model = self.create_model() 
        self.rewards = []
        self.loaded=False
    
    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))
        for n in self.neurons_num_list:
            model.add(Dense(n, activation=relu))
        model.add(Dense(NUMBER_OF_ACTION, name="Actions")) # sigmoid #  activation="softmax",
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model
    
    def replay(self, memory, model_info_path, model_name, episode_number, is_Resumption, save_path_):
        x_batch, y_batch = [], []
        if (is_Resumption) and (not self.loaded):
            # if os.path.exists(os.path.join(model_info_path, model_name + ".h5")):
            self.model = load_model(filepath=os.path.join(model_info_path, model_name + ".h5"))
            print("Load model")
            self.loaded=True
            # self.epsilon = load_object("epsilon_"+model_name, save_path_)
        mini_batch = random.sample(memory, min(len(memory), self.batch_size))
        for state, action, reward, next_state in mini_batch:
            if state is None:continue

            y_target = self.pred(state)
            y_target[0][action-1] = reward
            y_target[0][action-1] += self.gamma * np.max(self.Q_next(next_state))            

            # y_target = self.pred(state)
            # y_target[0][0] = reward + self.gamma * self.Q_next(next_state)[0]

            x_batch.append(state.reshape(-1, state.size))
            y_batch.append(y_target[0].reshape(-1, y_target[0].size))

        self.model.fit(np.squeeze(x_batch), np.squeeze(y_batch), epochs=self.epochs, batch_size=len(x_batch), validation_split=0.10, verbose=0)

        # if episode_number%number_of_episode_for_save==0 and episode_number!=0:
        #     self.model.save(os.path.join(model_info_path, model_name + ".h5"))
        #     print(f"the model saved {episode_number}")
        # self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
    
    def Q_next(self, state):
        return self.pred(state)[0]
    
    def fit(self,state, model_info_path, model_name, episode_number, is_Resumption, save_path_):
        return self.policy(state, model_info_path, model_name, episode_number, is_Resumption, save_path_)
    
    def policy(self, state, model_info_path, model_name, episode_number, is_Resumption, save_path_):
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            return self.value_function(state)
        if is_Resumption:
            self.epsilon = load_object("epsilon_"+model_name, save_path_)
        if self.epsilon >= self.min_epsilon:
            self.epsilon -= self.epsilon * self.decay
        if episode_number%number_of_episode_for_save==0 and episode_number!=0:
            self.model.save(os.path.join(model_info_path, model_name + ".h5"))
            save_object(self.epsilon, "epsilon_"+model_name, Result_Path)
            save_object(self.epsilon, "epsilon_"+model_name, save_path_)
        return np.random.choice([1, 2])#, 3, 4, 5, 6])
    
    def pred(self, state):
        x = self.model.predict(state.reshape(-1, state.size) , verbose=0) # 
        return x
    
    def value_function(self, state):
        return np.argmax(self.pred(state)[0])+1