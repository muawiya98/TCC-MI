from Codes.RLUnit.RLAlgorithm import RLAlgorithm
from Codes.configuration import NUMBER_OF_ACTION
import random
import numpy as np

random.seed(42)
np.random.seed(42)

class Qlearning(RLAlgorithm):

    def __init__(self, lr=0.01, gamma=0.9, epsilon=0.99, decay=0.000002, min_epsilon=0.0001):
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, decay=decay, min_epsilon=min_epsilon)
        self.algorithm_name = "Q-learning"
        self.q_table = {}
        self.epsilon_list = []

    def replay(self, state, next_state, action, reward):
        action -=1
        if not tuple(state) in self.q_table.keys():
            self.q_table[tuple(state)] = np.zeros(NUMBER_OF_ACTION)
        if not tuple(next_state) in self.q_table.keys():
            self.q_table[tuple(next_state)] = np.zeros(NUMBER_OF_ACTION)
        # Q(s, a) = (1-α)*Q(s, a) + α * (r + γ * max[Q(s', a')])
        self.q_table[tuple(state)][action] = ((1 - self.lr) * self.q_table[tuple(state)][action]) + \
                                             (self.lr * (reward + self.gamma * max(self.q_table[tuple(next_state)])))

    def policy(self, state):
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            return self.value_function(state)
        if self.epsilon >= self.min_epsilon:
            self.epsilon -= self.epsilon * self.decay
        # self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        self.epsilon_list.append(self.epsilon)
        if not tuple(state) in self.q_table.keys():
            self.q_table[tuple(state)] = np.zeros(NUMBER_OF_ACTION)
        return self.q_table[tuple(state)], np.random.choice([*range(1,NUMBER_OF_ACTION+1)])

    def value_function(self, state):
        if not tuple(state) in self.q_table.keys():
            self.q_table[tuple(state)] = np.zeros(NUMBER_OF_ACTION)
        return self.q_table[tuple(state)], np.argmax(self.q_table[tuple(state)])+1

    def fit(self, state):
        return self.policy(state)

    def print_q_table(self, q_table):
        print("------------------- Q LEARNING TABLE ------------------\n",
              q_table, "\n-------------------------------------------------------")