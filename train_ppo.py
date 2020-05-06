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


class ResBlock(tf.keras.models.Model):

    class InnerBlock(tf.keras.models.Model):

        def __init__(self, channels):
            super(ResBlock.InnerBlock, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')
            self.conv2 = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')

        def call(self, x):
            z = tf.keras.activations.relu(x)
            z = self.conv1(z)
            z = tf.keras.activations.relu(z)
            z = self.conv2(z)
            return x + z

    def __init__(self, channels, input_shape=None):
        super(ResBlock, self).__init__()
        if input_shape:
            self.conv = tf.keras.layers.Conv2D(channels, (3, 3), padding='same', input_shape=input_shape)
        else:
            self.conv = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        self.block1 = self.InnerBlock(channels)
        self.block2 = self.InnerBlock(channels)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class ImpalaModel(tf.keras.models.Model):

    def __init__(self, policy, input_shape, num_actions=10):

        # self.rb1 = ResBlock(16, input_shape)
        # self.rb2 = ResBlock(32)
        # self.rb3 = ResBlock(32)
        # self.relu = tf.keras.layers.ReLU()

        flatten = tf.keras.layers.Flatten()
        fc1 = tf.keras.layers.Dense(256, activation='relu')
        if policy:
            fc2 = tf.keras.layers.Dense(num_actions, activation='softmax')
        else:
            fc2 = tf.keras.layers.Dense(1)

        inputs = tf.keras.Input(shape=input_shape)
        x = self.res_block(inputs, 16)
        x = self.res_block(x, 32)
        x = self.res_block(x, 32)
        x = tf.keras.activations.relu(x)
        x = flatten(x)
        x = fc1(x)
        x = fc2(x)

        super(ImpalaModel, self).__init__(inputs=inputs, outputs=x)

    @staticmethod
    def res_block(inputs, channels):

        def inner_block(x):
            relu = tf.keras.activations.relu
            conv1 = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')
            conv2 = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')
            return conv2(relu(conv1(relu(x)))) + x

        conv = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')
        pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        x = conv(inputs)
        x = pool(x)
        x = inner_block(x)
        x = inner_block(x)
        return x

    # def call(self, x):
    #     x = self.rb1(x)
    #     x = self.rb2(x)
    #     x = self.rb3(x)
    #     x = self.relu(x)
    #     x = self.flatten(x)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return x


def fb_p_impala(input_shape: tuple, num_actions:int) -> tf.keras.Model:
    # rb1 = ResBlock(16, input_shape)
    # rb2 = ResBlock(32)
    # rb3 = ResBlock(32)
    # relu = tf.keras.layers.ReLU()
    # flatten = tf.keras.layers.Flatten()
    # fc1 = tf.keras.layers.Dense(256, activation='relu')
    # fc2 = tf.keras.layers.Dense(num_actions, activation='softmax')
    # return tf.keras.models.Sequential([rb1, rb2, rb3, relu, flatten, fc1, fc2])
    return ImpalaModel(True, input_shape, num_actions)


def fb_v_impala(input_shape: tuple) -> tf.keras.Model:
    # rb1 = ResBlock(16, input_shape)
    # rb2 = ResBlock(32)
    # rb3 = ResBlock(32)
    # relu = tf.keras.layers.ReLU()
    # flatten = tf.keras.layers.Flatten()
    # fc1 = tf.keras.layers.Dense(256, activation='relu')
    # fc2 = tf.keras.layers.Dense(1)
    # return tf.keras.models.Sequential([rb1, rb2, rb3, relu, flatten, fc1, fc2])
    return ImpalaModel(False, input_shape)


def fruitbot_policy_model(input_shape: tuple, num_actions: int) -> tf.keras.Model:
    """
    Returns a keras model for Q learning
    """
    conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4, activation='relu',
                                   data_format='channels_last', input_shape=input_shape)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', data_format='channels_last')
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', data_format='channels_last')

    flatten = tf.keras.layers.Flatten()

    fc1 = tf.keras.layers.Dense(512, activation='relu')
    fc2 = tf.keras.layers.Dense(num_actions, activation='softmax')

    return tf.keras.Sequential([conv1, conv2, conv3, flatten, fc1, fc2], name='fruitbot_policy')


def fruitbot_value_model(input_shape: tuple) -> tf.keras.Model:
    """
    Returns a keras model for Q learning
    """
    conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4, activation='relu',
                                   data_format='channels_last', input_shape=input_shape)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', data_format='channels_last')
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', data_format='channels_last')

    flatten = tf.keras.layers.Flatten()

    fc1 = tf.keras.layers.Dense(512, activation='relu')
    fc2 = tf.keras.layers.Dense(1, activation='softmax')

    return tf.keras.Sequential([conv1, conv2, conv3, flatten, fc1, fc2], name='fruitbot_value')


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
        # policy_func_constructor = fb_p_impala
        # value_func_constructor = fb_v_impala
        policy_func_constructor = fruitbot_policy_model
        value_func_constructor = fruitbot_value_model

        ppo.learn(
            env,
            policy_func_constructor,
            value_func_constructor,
            args.logdir,
            epochs=args.epochs,
            load_policy_from=args.load_policy_from,
            load_value_from=args.load_value_from,
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
            entropy_coef=0,
            epochs=args.epochs,
            # load_from=args.load_from,
            save_freq=args.save_freq,
            fruitbot=False
        )
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--load_policy_from', type=str, default=None)
    parser.add_argument('--load_value_from', type=str, default=None)
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
