from enum import Enum

# ================= Simulation Settings =================
# Network_Path = "Networks\\environment 3.3\\environment.sumocfg"
# Network_Path = "Networks\\environment 2.2\\environment.sumocfg"
Network_Path = "Networks\\environment 1.1\\environment.sumocfg"

Simulation_Time = 45000 # 144000 # 720000

TEST_STAGE = 90

# ================= Traffic Light Settings =================
traffic_light_period = 30

Yellow_period = 5

Green_red_period = 25

# ================= Object Settings =================
generation_period = 30 # 90 # 450

Vehicle_characteristics = {
    'length': 3,
    'min_cap': 0.5
}

HIGH_NUMBER_OF_VEHICLE = 10

LOW_NUMBER_OF_VEHICLE = 2

# ================= RL Settings =================
episode_time = 450 # 1440 # 7200

NUMBER_OF_ACTION = 6

WINDOW_SIZE = 15

W_Short_term = 0.6

# ================= Mathematical Models Settings =================
particle_variance = 5

Q = 15

R1 = Q

R2 = 1/Q

R3 = 2*Q

class Methods(Enum):
    Random = 'Random'
    Kalman_R1 = 'With Kalman R1'
    Kalman_R2 = 'With Kalman R2'
    Kalman_R3 = 'With Kalman R3'
    Traditional_R1 = 'Traditional RL R1'
    Traditional_R2 = 'Traditional RL R2'
    Traditional_R3 = 'Traditional RL R3'