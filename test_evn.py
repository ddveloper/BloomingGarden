import gym
import gym_bloomingGarden
import numpy as np

env = gym.make("bloomingGarden-v0")
env.reset()
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    ns, rw, done, _ = env.step(action)
