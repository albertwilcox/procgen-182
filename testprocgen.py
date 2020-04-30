import gym
env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy')
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        break
