import tensorflow as tf
import argparse, random, time
import gym
import numpy as np


def get_env(args):
    if args.env == 'procgen:procgen-fruitbot-v0':
        env = gym.make(args.env, distribution_mode='easy')
    else:
        env = gym.make(args.env)
    env.seed(args.seed)
    return env


def test_model(env, model, num_tests, fruitbot, render):
    for i in range(num_tests):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if fruitbot:
                obs = obs / 255.0
            outputs = model(np.expand_dims(obs, axis=0).astype(np.float32))
            a = np.argmax(outputs)
            if fruitbot:
                a = a * 3
            obs, reward, done, info = env.step(a)

            if render:
                env.render()
                time.sleep(0.01)

            total_reward += reward
        print("======================================")
        print("Total reward for test %d: %f" % (i, total_reward))
        print("======================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_tests', type=int, default=4e6)
    parser.add_argument('--load_loc', type=str, default=None)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    assert args.env in ['procgen:procgen-fruitbot-v0', 'CartPole-v0']
    fruitbot = args.env == 'procgen:procgen-fruitbot-v0'
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))

    env = get_env(args)
    model = tf.keras.models.load_model(args.load_loc, compile=False)
    test_model(env, model, args.num_tests, fruitbot, args.render)
