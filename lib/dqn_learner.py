import uuid
import time
import pickle
import os
import sys
import itertools
import numpy as np
import random
from lib.prioritized_replay_buffer import *
from collections import namedtuple
from lib.utils import *
from lib.replay_buffer import ReplayBuffer
import ray


@ray.remote(num_gpus=0.35, num_cpus=0.2)
class DqnLearner(object):
    def __init__(self,
                 q_model_constructor,
                 optimizer_params,
                 batch_size,
                 gamma,
                 target_update_freq,
                 input_shape,
                 num_actions,
                 multistep_len=1,
                 double_q=True,
                 logdir=None,
                 fruitbot=False,
                 dist_param=0,
                 dist_v_min=0,
                 dist_v_max=0,
                 load_from=None,
                 save_every=None):
        import tensorflow as tf
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using `q_func`.
        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        q_model_constructor: function
            Constructor of the model to use for computing the q function. It should accept the
            following named arguments:
                input_shape: tuple
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
        optimizer_params: dict
            Specifying the constructor and kwargs, as well as learning rate
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        double_q: bool
            If True, use double Q-learning to compute target values. Otherwise, vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        logdir: str
            Where we save the results for plotting later.
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

        # Training params
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        # Model params
        self.double_q = double_q
        self.gamma = gamma
        self.fruitbot = fruitbot

        # Distributional DQN specific parameters
        if dist_param:
            self.dist_param = dist_param
            self.dist_v_min = dist_v_min
            self.dist_v_max = dist_v_max
            self.dist_delta = (self.dist_v_max - self.dist_v_min) / self.dist_param

        # Misc Params
        self.save_freq = save_every
        self.multistep_len = multistep_len

        '''
        if fruitbot:
            img_h, img_w, img_c = input_shape
            input_shape = (img_h, img_w, img_c)
        else:
            input_shape = input_shape
        '''

        self.num_actions = num_actions

        if load_from:
            self.q_model = tf.keras.models.load_model(load_from)
        else:
            self.q_model = q_model_constructor(input_shape, self.num_actions)
        self.q_model_target = tf.keras.models.clone_model(self.q_model)
        self.q_model_target.set_weights(self.q_model.get_weights())
        print(self.q_model.summary())

        opt = optimizer_params.get('type', tf.keras.optimizers.Adam)
        lr = optimizer_params.get('learning_rate', 1e-4)
        grad_norm_clipping = optimizer_params.get('grad_norm_clipping', 10)
        self.optimizer = opt(learning_rate=lr, clipnorm=grad_norm_clipping)

        self.initialized = False
        self.num_param_updates = 0

        @tf.function
        def _error(obs_t, obs_tp1, actions_t, rew_t, done_mask_t, weights, eps=1e-4):
            import tensorflow as tf
            if self.fruitbot:
                obs_t = tf.cast(obs_t, tf.float32) / 255.0
                obs_tp1 = tf.cast(obs_tp1, tf.float32) / 255.0
            else:
                obs_t = tf.cast(obs_t, tf.float32)
                obs_tp1 = tf.cast(obs_tp1, tf.float32)

            q = self.q_model(obs_t)
            action_q = tf.gather_nd(q, tf.expand_dims(actions_t, 1), batch_dims=1)

            q_target = self.q_model_target(obs_tp1)
            if self.double_q:
                q_future = self.q_model(obs_tp1)
                target_actions = tf.math.argmax(q_future, axis=1)
                target_action_q = tf.gather_nd(q_target, tf.expand_dims(target_actions, 1), batch_dims=1)
            else:
                target_action_q = tf.reduce_max(q_target, axis=1)

            target = rew_t + (1.0 - tf.cast(done_mask_t,tf.float32)) * (self.gamma ** self.multistep_len) * target_action_q
            tf.stop_gradient(target)

            total_error = tf.reduce_mean(huber_loss(target - action_q))

            # Don't apply huber loss
            td_error = target - action_q + eps
            # total_error *= weights

            return total_error, td_error

        @tf.function
        def _dist_error(obs_t, obs_tp1, actions_t, rew_t, done_mask_t, weights):
            import tensorflow as tf
            """
            Error for a distributional DQN
            """
            if self.fruitbot:
                obs_t = tf.cast(obs_t, tf.float32) / 255.0
                obs_tp1 = tf.cast(obs_tp1, tf.float32) / 255.0
            else:
                obs_t = tf.cast(obs_t, tf.float32)
                obs_tp1 = tf.cast(obs_tp1, tf.float32)

            z_dist = self.q_model(obs_t)  # shape (batch, num_actions, dist_param)
            action_z_dist = tf.gather_nd(z_dist, tf.expand_dims(actions_t, 1), batch_dims=1)  # shape (batch, dist_param)

            z_dist_target = self.q_model_target(obs_tp1)
            vals = tf.range(self.dist_param) * self.dist_delta + self.dist_v_min  # shape (dist_param,)
            if self.double_q:
                pass
            else:
                q = tf.reduce_sum(z_dist_target * vals, axis=2)  # shape (batch, num_actions)
                actions = tf.math.argmax(q, axis=1)  # shape (batch,)
                action_z_dist_target = tf.gather_nd(z_dist_target, tf.expand_dims(actions, 1),
                                                    batch_dims=1)  # shape (batch, dist_param)

            # Project t+1 predictions for comparison
            # projections shape: (batch, dist_param)
            projections = tf.broadcast_to(vals * self.gamma, (self.batch_size, self.dist_param)) \
                          + tf.transpose(tf.broadcast_to(rew_t, (self.dist_param, self.batch_size)))
            projections = projections - self.dist_v_min
            projections = tf.math.minimum(projections, self.dist_v_max)
            projections = tf.math.maximum(projections, self.dist_v_min)
            projections = projections / self.dist_delta
            targets = tf.zeros((self.batch_size, self.dist_param))

        @tf.function
        def _optimizer_update(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch, weights):
            import tensorflow as tf
            with tf.GradientTape() as tape:
                error, td_error = self.error(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch, weights)
            grad = tape.gradient(error, self.q_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.q_model.trainable_variables))
            return error, td_error

        self.error = _error
        self.dist_error = _dist_error
        self.optimizer_update = _optimizer_update

    def get_weights(self):
        import tensorflow as tf
        return self.q_model.get_weights()

    def update_model(self, batch, indices, weights):
        import tensorflow as tf
        """
        Update with parameters recv'd from replay buffer
        """
        obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch = batch
        error, td_error = self.optimizer_update(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch, weights)
        new_priorities = np.abs(td_error)

        if self.save_freq and self.num_param_updates % self.save_freq == 0:
            self.save(str(self.num_param_updates))

        if self.num_param_updates % self.target_update_freq == 0:
            self.q_model_target.set_weights(self.q_model.get_weights())

        self.num_param_updates += 1

        return indices, new_priorities, self.num_param_updates

    def get_num_param_updates(self):
        return self.num_param_updates

    def save(self, strr):
        #self.q_model.save(os.path.join(self.logdir, 'model_%s.h5' % strr))
        pass
