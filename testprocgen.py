import gym
from gym import wrappers
import time

if __name__ == '__main__':
    """
    Actions: 
        0-2: left
        3-5: do nothing
        6-8: right
        9: shoot
        10-14: do nothing
    """
    env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy', start_level=1, num_levels=1)
    # env = wrappers.Monitor(env, 'test')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        # obs, rew, done, info = env.step(1)
        # print(env.action_space.n)
        env.render()
        time.sleep(0.01)
        if done:
            obs = env.reset()


