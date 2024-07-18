import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent


ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_EP = 5000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

for i in range(NUM_EP):
    done = False
    state, _ = env.reset()

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, trunc, info = env.step(action)
        agent.store_in_memory(state, action, reward, next_state, done)
        agent.learn()
        state = next_state

env.close()