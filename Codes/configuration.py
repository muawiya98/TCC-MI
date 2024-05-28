from enum import Enum
import pickle
import os

import random
import numpy as np

random.seed(42)
np.random.seed(42)

root_path = '/content/drive/MyDrive/Colab_Notebooks/Muawiya/TCC_MI/TCC-MI/'
# root_path = os.path.abspath(".")
# 
# ================= Simulation Settings =================
Network_Path = os.path.join(root_path, "Networks", "environment 1.1", "environment.sumocfg") # 2.2, 3.3

# Result_Path = os.path.join(os.path.abspath("."), "Results")
Result_Path = os.path.join(root_path, 'Results')

Simulation_Time = 9000 #13200 #26400 #396000# 792000

TEST_STAGE = 70

# ================= Traffic Light Settings =================
traffic_light_period = 30

Yellow_period = 5

Green_red_period = 25
# ================= Object Settings =================
generation_period = 90 # 30 #450

Vehicle_characteristics = {
    'length': 3,
    'min_cap': 0.5
}

HIGH_NUMBER_OF_VEHICLE = 18

LOW_NUMBER_OF_VEHICLE = 3

# ================= RL Settings =================
episode_time = 90 # 120 # 480 #7200 # 450 # 1440 # 

NUMBER_OF_ACTION = 2 # 6

WINDOW_SIZE = 15

Weighting_Factor = 0.6

number_of_episode_for_save = 5

# ================= Mathematical Models Settings =================
particle_variance = 5

Q = 15

R1 = Q

R2 = 1/Q

R3 = 2*Q

class Methods(Enum):
    Random = 'Random'
    Fixed = 'Fixed'
    Kalman_R1 = 'With Kalman R1'
    Kalman_R2 = 'With Kalman R2'
    Kalman_R3 = 'With Kalman R3'
    Traditional_R1 = 'Traditional RL R1'
    Traditional_R2 = 'Traditional RL R2'
    Traditional_R3 = 'Traditional RL R3'

# ================= Shared Functions =================
def save_object(obj, filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()

def load_object(filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object