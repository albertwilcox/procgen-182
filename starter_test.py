import tensorflow as tf
from baselines.common.policies import build_policy
from baselines.ppo2 import ppo2
from starter_models import *
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.ppo2.model import Model
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
import os, time, datetime
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import wrappers

# import tensorflow.python.util.deprecation as deprecation
from tensorflow_core.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

LOG_DIR = '/tmp/procgen/test'

ent_coef = .01
nsteps = 256
vf_coef = 0.5


def test_model(venv, model, num_per_model, num_envs):
    """
    Runs a vectorized environment with model until it has num_data episodes of data
    """
    obs = venv.reset()
    ep_rewards = []
    done_cts = [0]*num_envs
    while len(ep_rewards) < num_per_model:
        actions, values, states, neglogpacs = model.step(obs)

        obs, rewards, dones, infos = venv.step(actions)

        for i in range(num_envs):
            if (done_cts[i] >= num_per_model // num_envs):
                continue
            info = infos[i]
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                done_cts[i] += 1
                ep_rewards.append(maybeepinfo['r'])
                # print(len(ep_rewards))
    #print(done_cts)
    return ep_rewards[:num_per_model]


def make_venv(name, start_level, num_levels=None, num_envs=16, mode='easy'):
    venv = ProcgenEnv(num_envs=num_envs, env_name=name, num_levels=num_levels,
                      start_level=start_level, distribution_mode=mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    return venv


def plot_reward_training(train_files, test_files, save_dir, title='', x_mult=1):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if type(train_files) == str:
        train_files = [train_files]
    if type(test_files) == str:
        test_files = [test_files]

    train_data = np.array([np.loadtxt(f, delimiter=',') for f in train_files])
    test_data = np.array([np.loadtxt(f, delimiter=',') for f in test_files])

    plt.figure()
    plt.title(title)
    plt.ylabel('Avg. Reward')
    plt.xlabel('Training timesteps')
    x = np.arange(train_data.shape[1]) * x_mult
    y_train = np.mean(train_data, axis=(0, 2)).squeeze()
    y_test = np.mean(test_data, axis=(0, 2)).squeeze()
    plt.plot(x, y_train, label='Training Levels', c='g')
    plt.plot(x, y_test, label='Testing Levels', c='b')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'graph.pdf'))
    plt.savefig(os.path.join(save_dir, 'graph_im.png'))


def plot_baselines(train_files, test_files, save_dir, x_mult):
    """
    The files must come in 2D arrays, where each subarray contains a list of training trajectories to average over
    for a given number of training levels
    """
    assert len(train_files) == 4 and len(test_files) == 4, 'this function is not capable of operating outside of its very narrow range'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    titles = ['50 Training Levels', '100 Training Levels', '250 Training Levels', '500 Training Levels']

    train_data = [np.array([np.loadtxt(f, delimiter=',') for f in files]) for files in train_files]
    test_data = [np.array([np.loadtxt(f, delimiter=',') for f in files]) for files in test_files]

    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 5))
    fig.suptitle("Baselines")
    # fig.supylabel('Avg. Reward')
    # fig.supxlabel('Training timesteps')
    # fig.set()
    # fig.add_subplot(111, frameon=False)
    # plt.xlabel('Timesteps')
    # plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    for i, (dat_train, dat_test) in list(enumerate(zip(train_data, test_data))):
        x = np.arange(dat_train.shape[1]) * x_mult
        y_train = np.mean(dat_train, axis=(0, 2)).squeeze()
        y_test = np.mean(dat_test, axis=(0, 2)).squeeze()
        axs[i].plot(x, y_train, label='Training Levels', c='g')
        axs[i].plot(x, y_test, label='Testing Levels', c='b')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Timesteps')
    axs[0].set_ylabel('Average Reward')
    axs[3].legend()
    # axs[0].set_xlabel('Timesteps')

    # fig.legend()
    plt.savefig(os.path.join(save_dir, 'graph.pdf'))
    plt.savefig(os.path.join(save_dir, 'graph_im.png'))


def video_checkpoint(start_file, checkpoints, dest):
    env = gym.make('procgen:procgen-fruitbot-v0', start_level=1000, distribution_mode='easy')
    # env = wrappers.Monitor(env, dest)

    network_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = build_policy(env, network_fn, clip_vf=True)
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=0,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=0)

    checkpoints.sort()
    for cp in checkpoints:
        model.load(os.path.join(start_file, cp))
        env_w = gym.make('procgen:procgen-fruitbot-v0', start_level=1000, distribution_mode='easy')
        env_w = wrappers.Monitor(env_w, os.path.join(dest, cp), force=True)

        for i in range(5):
            obs = env_w.reset()
            while True:
                action, _, __, ___ = model.step(obs)

                obs, _, done, ___ = env_w.step(action)

                if done:
                    break


def visualize_layers(file, dest):
    env = gym.make('procgen:procgen-fruitbot-v0', start_level=1000, distribution_mode='easy')
    # env = wrappers.Monitor(env, dest)

    network_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = build_policy(env, network_fn, clip_vf=True)
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=0,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=0)

    model.load(file)

    first_layer = model.sess.run(tf.trainable_variables()[0])
    filters = [first_layer[:, :, :, idx] for idx in range(16)]

    fig, ax = plt.subplots(4, 4)
    fig.suptitle('Visualizing Convolutional Layer Filters')
    for i, filter in list(enumerate(filters)):
        a = i // 4
        b = i % 4
        ax[a, b].imshow(filter / np.max(filter))
    plt.show()


def make_videos(file):
    important = [
                    '00001',
                    '00010',
                    '00020',
                    '00030',
                    '00050',
                    '00100',
                    '00500',
                    '01000',
                    '01500',
                    '02000',
                    '02500',
                    '03000',
                ]
    dest = os.path.join('videos', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    video_checkpoint(file, important, dest)


def main():
    parser = argparse.ArgumentParser(description='Process procgen testing arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_levels_train', type=int, default=100)
    parser.add_argument('--start_level_train', type=int, default=0)
    parser.add_argument('--num_levels_test', type=int, default=50000)
    parser.add_argument('--start_level_test', type=int, default=500)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_per_model', type=int, default=128)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_stop', type=int, default=60)
    parser.add_argument('--figure_title', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='tmp/procgen/checkpoints')
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--ppo', action='store_true', default=False)

    args = parser.parse_args()
    
    logger.info("creating environment")

    venv_train = make_venv(args.env_name, args.start_level_train, args.num_levels_train, args.num_envs)
    venv_test = make_venv(args.env_name, args.start_level_test, args.num_levels_test, args.num_envs)

    network_fn = lambda x: build_impala_batchnorm(x, depths=[16, 32, 32], emb_size=256)
    ob_space = venv_train.observation_space
    ac_space = venv_train.action_space

    policy = build_policy(venv_train, network_fn, clip_vf=True)
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=args.num_envs, nbatch_train=0,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=0)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    model_files = os.listdir(args.model_dir)
    model_files.sort()
    model_files = [os.path.join(args.model_dir, file) for file in model_files]
    train_data = []
    test_data = []
    eval_freq = args.eval_freq
    for i, file in list(enumerate(model_files)):
        if i % eval_freq == 0 and i <= args.eval_stop:
            print(file)

            model.load(file)

            rew_train = test_model(venv_train, model, args.num_per_model, args.num_envs)
            rew_test = test_model(venv_test, model, args.num_per_model, args.num_envs)

            train_data.append(rew_train)
            test_data.append(rew_test)

            venv_train = make_venv(args.env_name, args.start_level_train, args.num_levels_train, args.num_envs)
            venv_test = make_venv(args.env_name, args.start_level_test, args.num_levels_test, args.num_envs)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    now = datetime.datetime.now()
    save_dir = os.path.join(args.save_dir, now.strftime("%m-%d-%H-%M-%S"))
    os.mkdir(save_dir)

    np.savetxt(os.path.join(save_dir, 'train_data.csv'), train_data, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'test_data.csv'), test_data, delimiter=',')

    plt.figure()
    plt.title(args.figure_title)
    plt.ylabel('Avg. Reward')
    plt.xlabel('Training timesteps')
    x = np.arange(len(train_data)) * 10 * eval_freq * 163_840
    y_train = np.mean(train_data, axis=1).squeeze()
    y_test = np.mean(test_data, axis=1).squeeze()
    plt.plot(x, y_train, label='Training Levels', c='g')
    plt.plot(x, y_test, label='Testing Levels', c='b')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'graph.pdf'))
    plt.savefig(os.path.join(save_dir, 'graph_im.png'))


if __name__ == '__main__':
    # main()
    # make_videos('checkpoints_250')
    # visualize_layers('checkpoints_250/03050', 'vis_test')
    # plot_reward_training(['outputs/base_500/train_data.csv', 'outputs/base_500_2/train_data.csv'],
    #                      ['outputs/base_500/test_data.csv', 'outputs/base_500_2/test_data.csv'],
    #                      'graphs/500', title='500 Training Levels', x_mult=163_840)
    # plot_reward_training(['outputs/base_100_1_16env/train_data.csv', 'outputs/base_100_2_16env/train_data.csv'],
    #                      ['outputs/base_100_1_16env/test_data.csv', 'outputs/base_100_2_16env/test_data.csv'],
    #                      'graphs/100_16env', title='100 Training Levels, 16 Environments', x_mult=163_840)
    plot_baselines([['outputs/base_50/train_data.csv'], ['outputs/base_100_1/train_data.csv'],
                    ['outputs/base_250/train_data.csv'], ['outputs/base_500/train_data.csv', 'outputs/base_500_2/test_data.csv']],
                   [['outputs/base_50/test_data.csv'], ['outputs/base_100_1/test_data.csv'],
                    ['outputs/base_250/test_data.csv'], ['outputs/base_500/test_data.csv', 'outputs/base_500_2/test_data.csv']],
                   'graphs/baselines', x_mult=163_840)
