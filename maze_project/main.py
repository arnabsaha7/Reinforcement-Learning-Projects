import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from maze_env import Maze, maze_layout
from agent import QLearningAgent
from matplotlib import animation

maze = Maze(maze_layout, (1, 1), (9, 18))

actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Define the Reward System
goal_reward = 100
wall_penalty = -10
step_penalty = -1

def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start
    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]

    while not is_done:
        action = agent.get_action(current_state, current_episode)
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])
        
        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[0]][next_state[1]] == 1:
            reward = wall_penalty
            next_state = current_state
        elif next_state == maze.goal:
            path.append(next_state)
            reward = goal_reward
            is_done = True
        else:
            path.append(next_state)
            reward = step_penalty
        
        episode_reward += reward
        episode_step += 1
        
        if train:
            agent.update_q_table(current_state, action, next_state, reward)
        
        current_state = next_state

    return episode_reward, episode_step, path


def test_agent(agent, maze, num_episodes=1):
    episode_reward, episode_step, path = finish_episode(agent, maze, num_episodes, train=False)
    print("Learned Path: ")
    for row, col in path:
        print(f"({row}, {col})", end=' ')
    print("Goal!")
    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)

    plot_final_path(path, maze, episode_step, episode_reward)
    return episode_step, episode_reward

def train_agent(agent, maze, num_episodes=100):
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)  # Append the episode step count directly

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')
    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, max(episode_steps) + 10)  # Adjust y-axis limit dynamically
    plt.title('Steps per Episode')
    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")
    
    plt.tight_layout()
    plt.show()

    # Find and return the final path with the highest reward
    best_episode = np.argmax(episode_rewards)
    _, _, final_path = finish_episode(agent, maze, best_episode, train=False)
    
    return final_path, episode_steps, episode_rewards

agent = QLearningAgent(maze)
final_path, episode_steps, episode_rewards = train_agent(agent, maze, num_episodes=100)

def plot_final_path(final_path, maze):
    plt.figure(figsize=(5, 5))
    plt.imshow(maze.maze, cmap='gray')
    plt.text(maze.start[1], maze.start[0], 'S', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal[1], maze.goal[0], 'G', ha='center', va='center', color='green', fontsize=20)

    # Plot arrows for the final path
    for i in range(len(final_path) - 1):
        current_state = final_path[i]
        next_state = final_path[i + 1]
        plt.arrow(current_state[1], current_state[0], next_state[1] - current_state[1], next_state[0] - current_state[0],
                  head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.title('Final Path Found by Agent')
    plt.show()

plot_final_path(final_path, maze)




