import numpy as np

class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.qtable = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        
    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate
    
    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)
        else:
            return np.argmax(self.qtable[state])
        
    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.qtable[next_state])
        current_q_value = self.qtable[state][action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.qtable[next_state][best_next_action] - current_q_value)
        self.qtable[state][action] = new_q_value
