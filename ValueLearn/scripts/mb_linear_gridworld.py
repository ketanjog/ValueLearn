from ValueLearn.algos.model_based import ModelBased
from ValueLearn.envs.linear_gridworld import LinearGridworld

# Constants for the Environment and Algorithm
T = 10000
state_space = 10
episode_length = 20
gamma = 0.8


algo = ModelBased(state_space, gamma)
env = LinearGridworld(T, state_space, algo,episode_length)

env.train()


print("Reward State: " + str(env.rewarding_state))
print("Value Function:\n" + str(algo.V))

