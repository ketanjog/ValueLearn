"""
A linear gridworld. A random s* is picked to have reward 1
All other states are set to reward 0.
"""

import torch
from ValueLearn.envs.base_env import BaseEnv

class LinearGridworld(BaseEnv):
    def __init__(self, T, length):
        super().__init__(T)
        self.length = length

        self.reset()

        self.rewards = []
        self.cum_rewards = [0]


    def step(self, action):
        assert action in [-1, 1]

        self.current_state += action

        reward = self.gridworld[self.current_state]

        return reward

    def update(self, reward):
        """
        Records the rewards
        """
        self.rewards.append(reward)
        self.cum_rewards.append(reward + self.cum_rewards[-1])

    def next_actions(self):
        """
        Provides the set of available actions to the agent
        """
        if self.current_state == 0:
            return [1]

        elif self.current_state == self.length - 1:
            return [-1]

        else:
            return [-1,1]

    def reset(self):
        self.gridworld = torch.zeros(self.length)
        self.rewarding_state = torch.randint(0, self.length, (1,)).item()
        self.gridworld[self.rewarding_state] = 1

        self.starting_state = torch.randint(0, self.length, (1,)).item()
        self.current_state = self.starting_state
        


        


