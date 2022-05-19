from ValueLearn.algos.model_based import ModelBased
from ValueLearn.envs.linear_gridworld import LinearGridworld

# Constants for the Environment and Algorithm
T = 1000
state_space = 10



algo = ModelBased(state_space)
env = LinearGridworld(T, state_space, algo)

env.train()

#print(env.gridworld)
print(algo.V)
print(algo.reward)
print("-----")
print(algo.tflag)
print(algo.vflag)
print(algo.rflag)