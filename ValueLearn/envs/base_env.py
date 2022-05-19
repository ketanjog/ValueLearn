"""
Base Class for the Environments
"""
from tqdm import tqdm


class BaseEnv:
    def __init__(self, T, episode_length):
        """
        Initializes the environment
        """
        self.name = "BaseEnv"
        self.T = T
        self.episode_length = episode_length

    def update(self):
        """
        Updates the environments internal state
        """
        raise NotImplementedError

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        raise NotImplementedError

    def start_new_episode(self):
        """
        Resets the agent into new starting state.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environment
        """
        raise NotImplementedError

    def train(self):
        """
        Trains the algorithm
        """
        pbar = tqdm(total=self.T)
        for _ in range(self.T):
            actions = self.next_actions()
            context = self.get_context()
            action = self.algo.choose_action(context, actions)
            r = self.step(action)
            self.algo.update(r)
            l_inf_loss = self.get_l_inf_loss(self.algo.get_value_function())
            self.update(l_inf_loss)

            if _ % self.episode_length == 0:
                self.start_new_episode()
                self.algo.reset()
            
            # print the reward
            # pbar.set_description(f"L1 loss: {self.l_inf_loss[-1]:.2f}")
            pbar.update(1)