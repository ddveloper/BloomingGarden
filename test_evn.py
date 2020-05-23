import gym
import gym_bloomingGarden
import numpy as np

env = gym.make("bloomingGarden-v0")
env.reset()
done = False
movable = (0, 0, 0, 1)
while not done:
    env.render()
    action_rand = env.action_space.sample()
    action = (movable[0], movable[1], action_rand[2], action_rand[3])
    ns, rw, done, mv = env.step(action)
    movable = mv.pop()