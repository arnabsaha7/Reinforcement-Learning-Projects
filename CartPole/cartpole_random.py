import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

def RandomGame():
    for ep in range(1000000):
        env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            print(t, next_state, reward, done, info, action)
            if done:
                break

RandomGame()