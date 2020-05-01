import gym
from gym import wrappers
import time

if __name__ == '__main__':
    env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy')
    env = wrappers.Monitor(env, 'test')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        # print(env.action_space.n)
        env.render()
        time.sleep(0.01)
        if done:
            obs = env.reset()
