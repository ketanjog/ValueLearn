from ValueLearn.algos.model_based import ModelBased
from ValueLearn.envs.linear_gridworld import LinearGridworld

# Constants for the Environment and Algorithm
T = 10000
state_space = 10
episode_length = 20



algo = ModelBased(state_space)
env = LinearGridworld(T, state_space, algo,episode_length)

env.train()

#print(env.gridworld)
print(algo.V)
print(algo.reward)
