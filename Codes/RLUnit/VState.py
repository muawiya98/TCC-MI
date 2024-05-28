from Codes.configuration import Weighting_Factor
from math import log
import numpy as np
import random
random.seed(42)
np.random.seed(42)

class State:
    
    def __init__(self, id, logger):
        self.id = id
        self.logger = logger
        self.number_of_edge = 4
        self.vehicle_list = np.zeros(self.number_of_edge)
        self.waiting_time_list = [[],] * self.number_of_edge
        self.high_range = (1, 4)
        self.low_range = (0, 4)
        self.throughput = 30
    
    def reset(self):
        self.vehicle_list = np.zeros(self.number_of_edge)
        self.waiting_time_list = [[],] * self.number_of_edge   

    def check_for_increas(self, index, value):
        if self.vehicle_list[index]+value<(100+value):
            return self.vehicle_list[index]+value
        else:return 100

    def generate_vehicles(self, sub_episode_number):
        for i in range(self.number_of_edge):
            # if sub_episode_number==0:
            self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # if sub_episode_number==1:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            
            # elif sub_episode_number==2:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            
            # elif sub_episode_number==3:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # elif sub_episode_number==4:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==5:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # elif sub_episode_number==6:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # elif sub_episode_number==7:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==8:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # elif sub_episode_number==9:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==10:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==11:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.low_range[0], self.low_range[1]))
            # elif sub_episode_number==12:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==13:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==14:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.low_range[0], self.low_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))
            # elif sub_episode_number==15:
            #     self.vehicle_list[0] = self.check_for_increas(0, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[1] = self.check_for_increas(1, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[2] = self.check_for_increas(2, random.randint(self.high_range[0], self.high_range[1]))
            #     self.vehicle_list[3] = self.check_for_increas(3, random.randint(self.high_range[0], self.high_range[1]))

    def update_vehicles_charactaristice(self, action):
        if action==1:
            self.waiting_time_list[0] = self.waiting_time_list[0][self.throughput:]
            self.waiting_time_list[2] = self.waiting_time_list[2][self.throughput:]
            if self.vehicle_list[0]>=self.throughput:self.vehicle_list[0]-=self.throughput
            else:self.vehicle_list[0]=0
            if self.vehicle_list[2]>=self.throughput:self.vehicle_list[2]-=self.throughput
            else:self.vehicle_list[2]=0
        elif action==2:
            self.waiting_time_list[1] = self.waiting_time_list[1][self.throughput:]
            self.waiting_time_list[3] = self.waiting_time_list[3][self.throughput:]
            if self.vehicle_list[1]>=self.throughput:self.vehicle_list[1]-=self.throughput
            else:self.vehicle_list[1]=0
            if self.vehicle_list[3]>=self.throughput:self.vehicle_list[3]-=self.throughput
            else:self.vehicle_list[3]=0
        # elif action==3:
        #     self.waiting_time_list[0] = self.waiting_time_list[0][self.throughput:]
        #     if self.vehicle_list[0]>=self.throughput:self.vehicle_list[0]-=self.throughput
        #     else:self.vehicle_list[0]=0
        # elif action==4:
        #     self.waiting_time_list[1] = self.waiting_time_list[1][self.throughput:]
        #     if self.vehicle_list[1]>=self.throughput:self.vehicle_list[1]-=self.throughput
        #     else:self.vehicle_list[1]=0
        # elif action==5:
        #     self.waiting_time_list[2] = self.waiting_time_list[2][self.throughput:]
        #     if self.vehicle_list[2]>=self.throughput:self.vehicle_list[2]-=self.throughput
        #     else:self.vehicle_list[2]=0
        # elif action==6:
        #     self.waiting_time_list[3] = self.waiting_time_list[3][self.throughput:]
        #     if self.vehicle_list[3]>=self.throughput:self.vehicle_list[3]-=self.throughput
        #     else:self.vehicle_list[3]=0
        
    def generate_update_waiting_time(self):
        for i, number_of_vehicle in enumerate(self.vehicle_list):
            for j in range(int(number_of_vehicle)):
                if self.waiting_time_list[i]==[] or j>=len(self.waiting_time_list[i]):
                    self.waiting_time_list[i].append(30)
                else:
                    self.waiting_time_list[i][j]+=30

    def reward_info(self):
        avg_waiting_time, std_waiting_time = [], []
        for w_t in self.waiting_time_list:
            if w_t!=[]:
                avg_waiting_time.append(np.average(w_t))
                std_waiting_time.append(np.std(w_t))
            else:
                avg_waiting_time.append(0)
                std_waiting_time.append(0)             
        return np.average(avg_waiting_time), np.std(std_waiting_time)

    def state_function(self):
        avg_waiting_time = []
        for w_t in self.waiting_time_list:
            if w_t!=[]:
                avg_waiting_time.append(np.average(w_t))
            else:avg_waiting_time.append(0)
        return np.array(avg_waiting_time)

    def save(self, waiting_time, std_waiting_time, reward):
        if not self.id in self.logger.waiting_time_history.keys():
            self.logger.waiting_time_history[self.id] = [waiting_time]
            self.logger.std_waiting_time_history[self.id] = [std_waiting_time]
            self.logger.reward_history[self.id] = [reward]
        else:
            self.logger.waiting_time_history[self.id].append(waiting_time)
            self.logger.std_waiting_time_history[self.id].append(std_waiting_time)
            self.logger.reward_history[self.id].append(reward)

    
    def reward_function(self):
        waiting_time, std_waiting_time = self.reward_info()
        reward = log((Weighting_Factor*std_waiting_time) + ((1-Weighting_Factor)*(0.9**waiting_time)), 0.5)
        self.save(waiting_time, std_waiting_time, reward)
        return reward