"""Utility functions that are useful for implementing DQN.
"""
import gym
import numpy as np
import random


def huber_loss(x, delta=1.0):
    import tensorflow as tf
    """https://en.wikipedia.org/wiki/Huber_loss
    """
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def compute_exponential_averages(variables, decay):
    import tensorflow as tf
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
    import tensorflow as tf
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


# ------------------------------------------------------------------------------
# SCHEDULES
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# REPLAY BUFFER
# ------------------------------------------------------------------------------


