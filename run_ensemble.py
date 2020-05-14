import tensorflow as tf
from baselines.common.policies import *
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

# import tensorflow.python.util.deprecation as deprecation
from tensorflow_core.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

LOG_DIR = '/tmp/procgen/test'
  
def most_frequent(l): 
    gamma = 0.95 # make later models less important in decision
    weights = [gamma**i for i in range(len(l))]
    counts = np.apply_along_axis(lambda x: np.bincount(x, weights=weights, minlength=len(l[0])), axis=1, arr=np.array(l).T)
    return np.argmax(counts,axis=1)

# In case of ties, priority goes to the earliest model in the list.
def make_ensemble(models):
    def step(obs):
        if (len(models) == 1):
            return models[0].step(obs)[0]
        votes = []
        for m in models:
            votes.append(m.step(obs)[0])
        return most_frequent(votes)
    return step
            
def test_ensemble(venv, model, per_env, num_envs):
    """
    Runs a vectorized environment with model until it has num_data episodes of data
    """
    obs = venv.reset()
    ep_rewards = []
    done_cts = [0]*num_envs
    while len(ep_rewards) < num_envs*per_env:
        actions = model(obs)

        obs, rewards, dones, infos = venv.step(actions)

        for i in range(num_envs):
            if (done_cts[i] >= per_env):
                continue
            info = infos[i]
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                done_cts[i] += 1
                ep_rewards.append(maybeepinfo['r'])
                # print(len(ep_rewards))
    #print(done_cts)
    return ep_rewards[:num_envs*per_env]

def make_venv(name, start_level, num_levels=None, num_envs=16, mode='easy'):
    venv = ProcgenEnv(num_envs=num_envs, env_name=name, num_levels=num_levels,
                      start_level=start_level, distribution_mode=mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    return venv

import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

'''
Copied in from baselines so that we could load multiple
'''
class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()


def main():
    ent_coef = .01
    nsteps = 256
    vf_coef = 0.5

    parser = argparse.ArgumentParser(description='Process procgen testing arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_levels_test', type=int, default=98000)
    parser.add_argument('--start_level_test', type=int, default=2000)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_per_env', type=int, default=8)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_models', type=int, default=10)
    parser.add_argument('--model_step', type=int, default=4)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--ppo', action='store_true', default=False)

    args = parser.parse_args()
    
    logger.info("creating environment")

    venv_test = make_venv(args.env_name, args.start_level_test, args.num_levels_test, args.num_envs)

    network_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
    ob_space = venv_test.observation_space
    ac_space = venv_test.action_space

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()
    
    model_files = os.listdir(args.model_dir)
    model_files.sort()
    model_files = [os.path.join(args.model_dir, file) for file in model_files]
    model_files = model_files[:-(args.num_models*args.model_step+1):-args.model_step]
    
    models = []
    
    for i,mf in enumerate(model_files):
        policy = build_policy(venv_test, network_fn, clip_vf=True)
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=args.num_envs, nbatch_train=0,
                      nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=0, mpi_rank_weight=i)
        model.load(mf)
        models.append(model)
        
    ensemble = make_ensemble(models)
    print(len(models), "models loaded.")
    test_data = []
    

    rew_test = test_ensemble(venv_test, ensemble, args.num_per_env, args.num_envs)
    print("Test acc: ", np.mean(rew_test))

    test_data.append(rew_test)

    test_data = np.array(test_data)

    now = datetime.datetime.now()
    # save_dir = os.path.join(args.save_dir, now.strftime("%m-%d-%H-%M-%S"))
    # os.mkdir(save_dir)

    # np.savetxt(os.path.join(save_dir, 'test_data.csv'), test_data, delimiter=',')



if __name__ == '__main__':
    main()
