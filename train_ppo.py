import os, random, time, argparse, gym, sys
import logz
import procgen
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import gym
from gym import wrappers

import ppo
from utils import *
from schedulers import *
    
def fruitbot_policy_model(input_shape: tuple, num_actions: int) -> tf.keras.Model:
    """
    Returns a keras model for Q learning
    """
    conv1 = tf.keras.layers.Conv2D(32, (9, 9), 2, data_format='channels_last', input_shape=input_shape)
    activation = tf.keras.layers.LeakyReLU(alpha=0.1)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, data_format='channels_last')
    activation2 = tf.keras.layers.LeakyReLU(alpha=0.1)
    conv3 = tf.keras.layers.Conv2D(64, (2, 2), 1, data_format='channels_last')
    activation3 = tf.keras.layers.LeakyReLU(alpha=0.1)
    # conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4, activation='relu',
    #                               data_format='channels_last', input_shape=input_shape)
    # conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', data_format='channels_last')
    # conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', data_format='channels_last')

    flatten = tf.keras.layers.Flatten()

    fc1 = tf.keras.layers.Dense(512)
    # fc1 = tf.keras.layers.Dense(512, activation='relu')
    activation4 = tf.keras.layers.LeakyReLU(alpha=0.1)
    fc2 = tf.keras.layers.Dense(num_actions, activation='softmax')

    return tf.keras.Sequential([conv1, activation, conv2, activation2, conv3, activation3, flatten, fc1, activation4, fc2], name='fruitbot_policy')


def fruitbot_value_model(input_shape: tuple) -> tf.keras.Model:
    """
    Returns a keras model for Q learning
    """
    conv1 = tf.keras.layers.Conv2D(32, (9, 9), 2, data_format='channels_last', input_shape=input_shape)
    activation = tf.keras.layers.LeakyReLU(alpha=0.1)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, data_format='channels_last')
    activation2 = tf.keras.layers.LeakyReLU(alpha=0.1)
    conv3 = tf.keras.layers.Conv2D(64, (2, 2), 1, data_format='channels_last')
    activation3 = tf.keras.layers.LeakyReLU(alpha=0.1)
    # conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4, activation='relu',
    #                               data_format='channels_last', input_shape=input_shape)
    # conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', data_format='channels_last')
    # conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', data_format='channels_last')

    flatten = tf.keras.layers.Flatten()

    # fc1 = tf.keras.layers.Dense(512, activation='relu')
    fc1 = tf.keras.layers.Dense(512)
    activation4 = tf.keras.layers.LeakyReLU(alpha=0.1)
    fc2 = tf.keras.layers.Dense(1, activation='softmax')

    return tf.keras.Sequential([conv1, activation, conv2, activation2, conv3, activation3, flatten, fc1, activation4, fc2], name='fruitbot_value')


def cartpole_policy_model(input_shape: tuple, num_actions: int) -> tf.keras.Model:
    """
    For CartPole we'll use a smaller network.
    """
    fc1 = tf.keras.layers.Dense(32, activation='tanh', input_shape=input_shape)
    fc2 = tf.keras.layers.Dense(32, activation='tanh')
    fc3 = tf.keras.layers.Dense(num_actions, activation='softmax')

    return tf.keras.Sequential([fc1, fc2, fc3])


def cartpole_value_model(input_shape: tuple) -> tf.keras.Model:
    """
    For CartPole we'll use a smaller network.
    """
    fc1 = tf.keras.layers.Dense(32, activation='tanh', input_shape=input_shape)
    fc2 = tf.keras.layers.Dense(32, activation='tanh')
    fc3 = tf.keras.layers.Dense(1)

    return tf.keras.Sequential([fc1, fc2, fc3])


def learn(env, args):
    if args.env == 'procgen:procgen-fruitbot-v0':
        policy_func_constructor = fruitbot_policy_model
        value_func_constructor = fruitbot_value_model

        ppo.learn(
            env,
            policy_func_constructor,
            value_func_constructor,
            args.logdir,
            epochs=args.epochs,
            load_from=args.load_from,
            save_freq=args.save_freq,
            fruitbot=True
        )
        env.close()
    elif args.env == 'CartPole-v0':
        policy_func_constructor = cartpole_policy_model
        value_func_constructor = cartpole_value_model

        ppo.learn(
            env,
            policy_func_constructor,
            value_func_constructor,
            args.logdir,
            epochs=args.epochs,
            load_from=args.load_from,
            save_freq=args.save_freq,
            fruitbot=False
        )
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--load_from', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    args = parser.parse_args()

    assert args.env in ['procgen:procgen-fruitbot-v0', 'CartPole-v0']
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))
    exp_name = 'ppo'

    if not(os.path.exists('data_ppo')):
        os.makedirs('data_ppo')
    logdir = exp_name + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data_ppo', logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir

    env = get_env(args)
    learn(env, args)
