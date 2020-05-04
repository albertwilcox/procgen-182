import os, random, time, argparse, gym, sys
import logz
import procgen
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import dqn
import gym
from gym import wrappers
from utils import *
from schedulers import *


def fruitbot_model(input_shape: tuple, num_actions: int, dist_param=0) -> tf.keras.Model:
    """
    Returns a keras model for Q learning
    """
    conv1 = tf.keras.layers.Conv2D(32, (8, 8), 4, activation='relu',
                                   data_format='channels_last', input_shape=input_shape)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', data_format='channels_last')
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', data_format='channels_last')

    flatten = tf.keras.layers.Flatten()

    fc1 = tf.keras.layers.Dense(512)
    if dist_param:
        fc2 = tf.keras.layers.Dense((num_actions, dist_param))
    else:
        fc2 = tf.keras.layers.Dense(num_actions)

    return tf.keras.Sequential([conv1, conv2, conv3, flatten, fc1, fc2])


def cartpole_model(input_shape: tuple, num_actions: int, dist_param=0) -> tf.keras.Model:
    """
    For CartPole we'll use a smaller network.
    """
    fc1 = tf.keras.layers.Dense(32, activation='tanh', input_shape=input_shape)
    fc2 = tf.keras.layers.Dense(32, activation='tanh')
    fc3 = tf.keras.layers.Dense(num_actions)

    return tf.keras.Sequential([fc1, fc2, fc3])


def learn(env, args):
    if args.env == 'procgen:procgen-fruitbot-v0':
        optimizer = {
            'type': tf.keras.optimizers.Adam,
            'learning_rate': 2.5e-4,
            'grad_norm_clipping': 10
        }

        limit = max(int(args.num_steps/2), 2e6)

        exploration_schedule = PiecewiseSchedule([
                (0,     1.00),
                (1e6,   0.10),
                (limit, 0.01),
            ], outside_value=0.01
        )

        q_model_constructor = fruitbot_model

        dqn.learn(
            env=env,
            q_model_constructor=q_model_constructor,
            optimizer_params=optimizer,
            exploration=exploration_schedule,
            replay_buffer_size=100000,
            batch_size=512,
            gamma=0.99,
            learning_starts=50000,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            multistep_len=3 if args.multistep else 1,
            double_q=args.double_q,
            logdir=args.logdir,
            max_steps=args.num_steps,
            fruitbot=True,
            load_from=args.load_from,
            save_every=args.save_freq
        )
        env.close()
    elif args.env == 'CartPole-v0':
        optimizer = {
            'type': tf.keras.optimizers.Adam,
            'learning_rate': 5e-4,
            'grad_norm_clipping': 10
        }

        limit = max(int(args.num_steps / 2), 2e6)

        exploration_schedule = PiecewiseSchedule([
            (0, 1.00),
            (5e4, 0.10),
            (1e5, 0.02),
        ], outside_value=0.02
        )

        q_model_constructor = cartpole_model

        dqn.learn(
            env=env,
            q_model_constructor=q_model_constructor,
            optimizer_params=optimizer,
            exploration=exploration_schedule,
            replay_buffer_size=10000,
            batch_size=100,
            gamma=0.99,
            learning_starts=1000,
            learning_freq=4,
            frame_history_len=1,
            target_update_freq=500,
            multistep_len=3 if args.multistep else 1,
            double_q=args.double_q,
            logdir=args.logdir,
            max_steps=args.num_steps,
            fruitbot=False,
            load_from=args.load_from,
            save_every=args.save_freq
        )
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=4e6)
    parser.add_argument('--double_q', action='store_true', default=False)
    parser.add_argument('--load_from', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--multistep', action='store_true', default=False)
    args = parser.parse_args()

    assert args.env in ['procgen:procgen-fruitbot-v0', 'CartPole-v0']
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))
    exp_name = 'dqn'
    if args.double_q:
        exp_name = 'double-dqn'

    if not(os.path.exists('data_dqn')):
        os.makedirs('data_dqn')
    logdir = exp_name + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data_dqn', logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir

    env = get_env(args)
    learn(env, args)
