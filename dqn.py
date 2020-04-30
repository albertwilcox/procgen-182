import uuid
import time
import pickle
import os
import sys
import gym.spaces
import itertools
import numpy as np
import random
import logz
import tensorflow as tf
from collections import namedtuple
from dqn_utils import *
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


class QLearner(object):

    def __init__(self,
                 env,
                 q_model,
                 optimizer_spec,
                 exploration,
                 replay_buffer_size,
                 batch_size,
                 gamma,
                 learning_starts,
                 learning_freq,
                 frame_history_len,
                 target_update_freq,
                 grad_norm_clipping,
                 double_q=True,
                 logdir=None,
                 max_steps=2e8):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using `q_func`.
        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_model: function
            Model to use for computing the q function. It should accept the
            following named arguments:
                img_in: tf.Tensor
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
                scope: str
                    scope in which all the model related variables
                    should be created
                reuse: bool
                    whether previously created variables should be reused.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        exploration: Schedule
            schedule for probability of chosing random action.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        double_q: bool
            If True, use double Q-learning to compute target values. Otherwise, vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        logdir: str
            Where we save the results for plotting later.
        max_steps: int
            Maximum number of training steps. The number of *frames* is 4x this
            quantity (modulo the initial random no-op steps).
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        self.max_steps = int(max_steps)
        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.exploration = exploration
        self.double_q = double_q
        self.gamma = gamma
        self.env = env

        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)

        self.num_actions = self.env.action_space.n

        self.q_model = q_model
        self.q_model_target = None # TODO: duplicate this model somehow

        self.optimizer = None # TODO: initialize a TF optimizer

        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        self.replay_buffer_idx = None

        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.std_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.log_every_n_steps = 10000
        self.start_time = time.time()
        self.last_obs = self.env.reset()
        self.t = 0


    def error(self, obs_t, obs_tp1, actions_t, rew_t, done_mask_t):
        q = self.q_model(obs_t)
        # TODO: this is the way I did this in my HW, but there might be a more efficient way. Do any of you have something better?
        action_q = tf.linalg.diag_part(tf.gather(q, actions_t, axis=1))

        q_target = self.q_model_target(obs_tp1)
        if self.double_q:
            target_actions = tf.math.argmax(q, axis=1)
            target_action_q = tf.linalg.diag_part(tf.gather(q_target, target_actions, axis=1))
        else:
            target_action_q = tf.reduce_max(q_target, axis=1)

        target = rew_t + (1 - done_mask_t) * self.gamma * target_action_q
        total_error = tf.reduce_mean(huber_loss(target - action_q))

        return total_error

    def step_env(self):
        """
        Step the env and store the transition.
        """

        # Find an action with epsilon exploration
        ep = self.exploration.value(self.t)
        if not self.model_initialized or np.random.random() < ep:
            a = np.random.randint(self.num_actions)
        else:
            last_obs = self.replay_buffer.encode_recent_observation()
            outputs = self.q_model(last_obs)
            a = np.argmax(outputs)

        # Step environment with that action, reset if `done==True`
        obs, reward, done, info = self.env.step(a)
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        self.replay_buffer.store_effect(self.replay_buffer_idx, a, reward, done)
        if done:
            obs = self.env.reset()

        self.last_obs = obs

    def update_model(self):
        """
        Perform experience replay and train the network.

        Only done if the replay buffer has enough samples
        """
        if (self.t > self.learning_starts and
                self.t % self.learning_freq == 0 and
                self.replay_buffer.can_sample(self.batch_size)):

            # TODO: Pretty sure the following is unnecesary, but that should be verified
            # if not self.model_initialized:
            #     self.session.run(tf.global_variables_initializer())
            #     self.session.run(self.update_target_fn)
            #     self.model_initialized = True

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask_batch = self.replay_buffer.sample(self.batch_size)

            with tf.GradientTape() as tape:
                error = self.error(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch)
            grad = tape.gradient(error, self.q_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.q_model.trainable_variables))

            if self.num_param_updates % self.target_update_freq == 0:
                # TODO: figure out how to update target model
                # Code from hw for reference:
                # update_target_fn = []
                # for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                #                            sorted(target_q_func_vars, key=lambda v: v.name)):
                #     update_target_fn.append(var_target.assign(var))
                # self.update_target_fn = tf.group(*update_target_fn)
                pass

            self.num_param_updates += 1

        self.t += 1

    def log_progress(self):
        # TODO: DO we use this or use our own thing?
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
            self.std_episode_reward = np.std(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = \
                max(self.best_mean_episode_reward, self.mean_episode_reward)

        # See the `log.txt` file for where these statistics are stored.
        if self.t % self.log_every_n_steps == 0:
            lr = self.optimizer_spec.lr_schedule.value(self.t)
            hours = (time.time() - self.start_time) / (60. * 60.)
            logz.log_tabular("Steps", self.t)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes", len(episode_rewards))
            logz.log_tabular("Exploration_Epsilon", self.exploration.value(self.t))
            logz.log_tabular("Adam_Learning_Rate", lr)
            logz.log_tabular("Elapsed_Time_Hours", hours)
            logz.dump_tabular()


def learn(*args, **kwargs):
    alg = QLearner(*args, **kwargs)
    while True:
        alg.step_env()
        # The environment should have been advanced one step (and reset if done
        # was true), and `self.last_obs` should point to new latest observation
        alg.update_model()
        alg.log_progress()
        if alg.t > alg.max_steps:
            print("\nt = {} exceeds max_steps = {}".format(alg.t, alg.max_steps))
            sys.exit()
