"""Utility functions that are useful for implementing DQN.
"""
import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
import os


def huber_loss(x, delta=1.0):
    """https://en.wikipedia.org/wiki/Huber_loss
    """
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables create ops corresponding to
    their exponential averages.

    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.

    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in `var_list`
    while ensure the norm of the gradients for each variable is clipped to
    `clip_val`.
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)


def get_env(args):
    if args.env == 'procgen:procgen-fruitbot-v0':
        env = gym.make(args.env, distribution_mode='easy')
    else:
        env = gym.make(args.env)

    set_global_seeds(args.seed)
    env.seed(args.seed)
    expt_dir = os.path.join(args.logdir, "gym")
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=False)
    return env


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(i)
    np.random.seed(i)
    random.seed(i)


