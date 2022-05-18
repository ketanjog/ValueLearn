"""
A barebones implementation of Model Based RL. Given a policy, 
tries to estimate the transition matrix T and reward function R to
learn value function V
"""

from ValueLearn.algos.base_algo import BaseAlgo
import torch

class ModelBased(BaseAlgo):
    def __init__(self, state_space, run_id=None):
        super().__init__(run_id)
        self.state_space = state_space
        self.set_policy()
        
        self.T = torch.zeros((self.state_space, self.state_space))
        self.reward = torch.zeros(self.state_space)
        self.V = torch.zeros(self.state_space)
        
        self.current_state = None
        self.previous_state = None
        self.previous_reward = None

    def choose_action(self, context, actions):
        """
        returns action as chosen by preset policy
        """
        self.previous_state = self.current_state
        self.current_state = context
        
        if len(actions) == 1:
            return actions[0]
        else:
            return self.policy[context].item()
        
    
    
    def set_policy(self):
        self.policy = torch.randint(0, 2, (self.state_space,))
        self.policy[self.policy == 0] = -1

    def update(self, reward):
        """
        Updates R(s) (expected (deterministic) reward in state s)
        Updates T(s,s') (Nonnormalised Probability of transition)
        """

        #Update V, only V(curr) and V(prev) changes.
        if self.previous_state is not None:
            
            # If this is the first time the state is visited, denom will be 0 -> skip update
            if torch.sum(self.T[self.previous_state]) != 0:
                self.V[self.previous_state] += (self.V[self.current_state] - torch.dot(self.V, self.T[self.previous_state])) / ((torch.sum(self.T[self.previous_state]))*(torch.sum(self.T[self.previous_state] + 1)))
            
            self.V[self.current_state] += self.previous_reward - self.reward[self.current_state]


            

        # Update T
        if self.previous_state is not None:
            self.T[self.previous_state][self.current_state] += 1
        
        # Update R
        if self.previous_reward is not None:
            self.reward[self.current_state] = self.previous_reward

        # Save new reward. Will update R(s) for resultant state in 
        # the next iteration
        self.previous_reward = reward

    def get_value_function(self):
        return self.V
        
        
            

        

