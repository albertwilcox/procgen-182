import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
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
import os

LOG_DIR = '/tmp/procgen/test'

def test_model(i, venv, n_steps, model, num_tests, ppo, render):
    obs = env.reset()
    done = False
    total_reward = 0
    # last_obs = [np.zeros(env.observation_space.shape) for _ in range(4)]
    info = None
    for j in range(n_steps):
        outputs = model(np.expand_dims(obs, axis=0).astype(np.float32)).numpy().squeeze()

        if ppo:
            a = np.random.choice(np.arange(outputs.size), p=outputs, axis=1)
        else:
            a = np.argmax(outputs, axis=1)

        obs, reward, done, info = venv.step(a)

        if render:
            venv.render()
            time.sleep(0.01)

        total_reward += venv.get_original_reward() / num_envs
    '''
    print("======================================")
    print("Total reward for test %d: %f" % (i, total_reward))
    print("======================================")
    '''
    print(i, info)

def main():
    num_envs = 64
    learning_rate = 0
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 512
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    timesteps_per_proc = 30000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen testing arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    #parser.add_argument('--num_levels', type=int, default=100)
    parser.add_argument('--start_level', type=int, default=500)
    parser.add_argument('--model_dir', type=str, default='/tmp/procgen/checkpoints')

    args = parser.parse_args()
    
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    network = build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    
    # sort by numerical order, so our results are in order
    sort_key = lambda x: int(''.join(ch for ch in x if ch.isdigit()))
    
    model_files = [f for f in os.listdir(args.model_dir) if os.path.isfile(os.path.join(mypath, f))]
    model_files.sort(key=sort_key)
    i = 0
    for m in model_files:
        if i % 2 != 0:
            continue
        i += 1
        logger.configure(dir=LOG_DIR, format_strs=format_strs)
        



if __name__ == '__main__':
    main()
