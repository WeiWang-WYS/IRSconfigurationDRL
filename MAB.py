"""
Multi-arm bandit
"""

import numpy as np
import random

class MAB:
    def __init__(self, BanditNum):
        self.epsilon = 0.01  # exploration rate
        self.Q_MAB = np.zeros(BanditNum)
        self.Count_MAB = np.zeros(BanditNum)
        self.ActNum = BanditNum

    def act_sel(self):
        if np.random.rand()<self.epsilon:
            act_index = random.randrange(self.ActNum)
        else:
            act_index = np.argmax(self.Q_MAB)
        return act_index

    def Q_update(self, act_index, Reward):
        self.Q_MAB[act_index] =  (self.Q_MAB[act_index]*self.Count_MAB[act_index]  + Reward)/(self.Count_MAB[act_index]+1)
        self.Count_MAB[act_index] = self.Count_MAB[act_index] + 1