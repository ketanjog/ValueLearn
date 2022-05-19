"""
A linear gridworld. A random s* is picked to have reward 1
All other states are set to reward 0.
"""

import torch
from ValueLearn.envs.base_env import BaseEnv

class LinearGridworld(BaseEnv):
    def __init__(self, T, state_space, algo, episode_length=10):
        super().__init__(T, episode_length)
        self.state_space = state_space
        self.algo = algo

        self.reset()


    def step(self, action):
        assert action in [-1, 1]

        self.current_state += action

        reward = self.gridworld[self.current_state]

        return reward

    def update(self):
        """
        Nothing yet
        """
        pass

    def next_actions(self):
        """
        Provides the set of available actions to the agent
        """
        if self.current_state == 0:
            return [1]

        elif self.current_state == self.state_space - 1:
            return [-1]

        else:
            return [-1,1]

    def reset(self):
        self.gridworld = torch.zeros(self.state_space)
        self.rewarding_state = torch.randint(0, self.state_space, (1,)).item()
        self.gridworld[self.rewarding_state] = 1

        self.starting_state = torch.randint(0, self.state_space, (1,)).item()
        self.current_state = self.starting_state

    def start_new_episode(self):
        self.starting_state = torch.randint(0, self.state_space, (1,)).item()
        self.current_state = self.starting_state


    def get_context(self):
        return self.current_state
        


        


