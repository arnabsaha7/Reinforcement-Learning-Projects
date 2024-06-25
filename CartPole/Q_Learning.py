import gymnasium as gym
import numpy as np
import random
import time


class QL():
    def __init__(self, env, alpha, gamma, epsilon, n_episodes, numberOfBins, lowerBounds, upperBounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.n_actions = self.env.action_space.n
        self.sumRewardsEpisode = []
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.n_actions))

    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngularVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], poleAngularVelocityBin) - 1, 0)

        return tuple ([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def chooseAction(self, state, index):
        # first 500 episodes --> completely random actions to have enough explorations
        if index < 500:
            return np.random.choice(self.n_actions)
        randomNumber = np.random.random() 
        # after 7000 episodes --> slowly decrease the epsilon parameter
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        for indexEpisode in range(self.n_episodes):
            rewardsEpisodes = []
            (stateS, _) = self.env.reset()
            stateS = list(stateS)
            print("Simulating episode {}". format(indexEpisode))

            terminalState = False
            while not terminalState:
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.chooseAction(stateS, indexEpisode)
                (stateSprime, reward, terminalState, _, _) = self.env.step(actionA)
                rewardsEpisodes.append(reward)
                stateSprime = list(stateSprime)
                stateSprimeIndex = self.returnIndexState(stateSprime)
                QmaxPrime = np.max(self.Qmatrix[stateSprimeIndex])

                if not terminalState:
                    error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    error = reward - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
                stateS = stateSprime
            print("Sum of rewards {}.".format(np.sum(rewardsEpisodes)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisodes))

    def simulateLearnedStratergy(self):
        env1 = gym.make("CartPole-v1", render_mode="human")
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        # obtained rewards at every timestep
        obtainedRewards = []
        for timeIndex in range(timeSteps):
            print(timeIndex)
            actionInState = np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)] == np.max(self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInState)
            obtainedRewards.append(reward)
            time.sleep(0.03)
            if terminated:
                time.sleep(1)
                break
        return obtainedRewards, env1

    def simulateRandomStrategy(self):
        env2 = gym.make("CartPole-v1", render_mode="human")
        (currentState, _) = env2.reset()
        env2.render()
        episodeNumber = 100
        timeSteps = 1000
        sumRewardsEpisodes = []
        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode = []
            initial_state = env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if terminated:
                    break
                sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
            return sumRewardsEpisodes, env2 

