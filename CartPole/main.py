import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from Q_Learning import QL


env = gym.make("CartPole-v1", render_mode="human")
(state, _) = env.reset()
env.render()

# parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]

# parameters for Q-Learning
alpha = 0.1
gamma = 1
epsilon = 0.2
n_episodes = 15000

# Create an Object
Q1 = QL(env, alpha, gamma, epsilon, n_episodes, numberOfBins, lowerBounds, upperBounds)
Q1.simulateEpisodes()
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

# Plot the convergence & Save it
plt.figure(figsize=(12,5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')

# Close the Environment
env1.close()
np.sum(obtainedRewardsOptimal)

# Simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()