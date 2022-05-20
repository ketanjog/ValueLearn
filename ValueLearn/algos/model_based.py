"""
A barebones implementation of Model Based RL. Given a policy, 
tries to estimate the transition matrix T and reward function R to
learn value function V
"""

from ValueLearn.algos.base_algo import BaseAlgo
import torch

class ModelBased(BaseAlgo):
    def __init__(self, state_space, gamma, run_id=None):
        super().__init__(run_id)
        self.state_space = state_space
        self.set_policy()
        
        self.T = torch.zeros((self.state_space, self.state_space))
        self.reward = torch.zeros(self.state_space)
        self.V = torch.zeros(self.state_space)
        self.gamma = gamma

        self.reset()
        

    def choose_action(self):
        """
        returns action as chosen by preset policy
        """
    
        return self.policy[self.current_state].item()
        
    
    
    def set_policy(self):
        """
        Create a random deterministic policy
        """
        self.policy = torch.randint(0, 2, (self.state_space,))
        self.policy[self.policy == 0] = -1

    def update(self, context, reward):
        """
        Iteratively updates V. 
        Updates R(s) (expected (deterministic) reward in state s)
        Updates T(s,s') (Nonnormalised Probability of transition)
        """
        self.previous_state = self.current_state
        self.current_state = context
            

        # Update T
        if self.previous_state is not None:
            self.T[self.previous_state][self.current_state] += 1

            # Normalise probability matrix T
            self.T = torch.nn.functional.normalize(self.T, p=1)

        # v = R + gamma * Tv
        self.V = self.reward + self.gamma * torch.matmul(self.T, self.V)
        
        # Update R
        self.reward[self.current_state] = reward
    
        

    def get_value_function(self):
        return self.V

    def reset(self):
        self.current_state = None
        self.previous_state = None
        self.previous_reward = None
        
        
            

        

