import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class skipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        score = 0
        done = False
        for i in range(self._skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            score += reward
            if done:
                break
        return next_state, score, done, trunc, info
    
def apply_wrappers(env):
    env = skipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env