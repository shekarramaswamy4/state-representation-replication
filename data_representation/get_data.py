import gym
import time
# import numpy as np

from atariari.benchmark.wrapper import AtariARIWrapper
env = gym.make('Breakout-v0')
observation = env.reset()
print(observation.shape)
print(env.observation_space)
print(env.action_space)
for i in range(1000):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample())  # take a random action
    print(info)
    print(env.unwrapped.ale.getRAM())
    break
    if done:
        print(i)
        break
    # time.sleep(1)
env.close()
