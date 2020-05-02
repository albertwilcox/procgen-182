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
from replay_buffer import *
from collections import namedtuple
from dqn_utils import *
from replay_buffer import ReplayBuffer


class QLearner(object):

    def __init__(self,
                 env,
                 q_model_constructor,
                 optimizer_params,
                 exploration,
                 replay_buffer_size,
                 batch_size,
                 gamma,
                 learning_starts,
                 learning_freq,
                 frame_history_len,
                 target_update_freq,
                 double_q=True,
                 logdir=None,
                 max_steps=2e8,
                 fruitbot=False,
                 dist_param=0,
                 dist_v_min=0,
                 dist_v_max=0,
                 load_from=None,
                 save_every=None,):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using `q_func`.
        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_model_constructor: function
            Constructor of the model to use for computing the q function. It should accept the
            following named arguments:
                input_shape: tuple
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
        optimizer_params: dict
            Specifying the constructor and kwargs, as well as learning rate
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
        double_q: bool
            If True, use double Q-learning to compute target values. Otherwise, vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        logdir: str
            Where we save the results for plotting later.
        max_steps: int
            Maximum number of training steps. The number of *frames* is 4x this
            quantity (modulo the initial random no-op steps).
        """
        # We're not using gym anymore, for performance concerns
        # assert type(env.observation_space) == gym.spaces.Box
        # assert type(env.action_space) == gym.spaces.Discrete

        # Training params
        self.max_steps = int(max_steps)
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.exploration = exploration
        self.env = env

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
        self.logdir = logdir

        if fruitbot:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        else:
            input_shape = self.env.observation_space.shape

        if fruitbot:
            self.num_actions = 4
        else:
            self.num_actions = self.env.action_space.n

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

        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        self.replay_buffer_idx = None

        self.initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.std_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.log_every_n_steps = 10000 if fruitbot else 1000
        self.start_time = time.time()
        if self.fruitbot:
            self.last_obs = self.env.reset() / 255.0
        else:
            self.last_obs = self.env.reset()
        self.t = 0

    @tf.function
    def error(self, obs_t, obs_tp1, actions_t, rew_t, done_mask_t):
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

        target = rew_t + (1 - done_mask_t) * self.gamma * target_action_q
        tf.stop_gradient(target)

        total_error = tf.reduce_mean(huber_loss(target - action_q))

        return total_error

    @tf.function
    def dist_error(self, obs_t, obs_tp1, actions_t, rew_t, done_mask_t):
        """
        Error for a distributional DQN
        """
        if self.fruitbot:
            obs_t = tf.cast(obs_t, tf.float32)/255.0
            obs_tp1 = tf.cast(obs_tp1, tf.float32)/255.0
        else:
            obs_t = tf.cast(obs_t, tf.float32)
            obs_tp1 = tf.cast(obs_tp1, tf.float32)

        z_dist = self.q_model(obs_t) # shape (batch, num_actions, dist_param)
        action_z_dist = tf.gather_nd(z_dist, tf.expand_dims(actions_t, 1), batch_dims=1) # shape (batch, dist_param)

        z_dist_target = self.q_model_target(obs_tp1)
        vals = tf.range(self.dist_param) * self.dist_delta + self.dist_v_min  # shape (dist_param,)
        if self.double_q:
            pass
        else:
            q = tf.reduce_sum(z_dist_target * vals, axis=2) # shape (batch, num_actions)
            actions = tf.math.argmax(q, axis=1) # shape (batch,)
            action_z_dist_target = tf.gather_nd(z_dist_target, tf.expand_dims(actions, 1), batch_dims=1) # shape (batch, dist_param)

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
    def optimizer_update(self, obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch):
        with tf.GradientTape() as tape:
            error = self.error(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch)
        grad = tape.gradient(error, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.q_model.trainable_variables))

    def step_env(self):
        """
        Step the env and store the transition.
        """

        # Find an action with epsilon exploration
        ep = self.exploration.value(self.t)
        # if not self.model_initialized or np.random.random() < ep:
        if np.random.random() < ep:
            a = np.random.randint(self.num_actions)
        else:
            last_obs = self.replay_buffer.encode_recent_observation()
            outputs = self.q_model(np.expand_dims(last_obs, axis=0).astype(np.float32))
            a = np.argmax(outputs)

        # Step environment with that action, reset if `done==True`
        if self.fruitbot:
            # This is just to convert one of the four actions returned by the Q model to
            # one of the 15 actions the env recognizes
            a_env = a * 3
        else:
            a_env = a
        obs, reward, done, info = self.env.step(a_env)

        # Reward engineering:
        if self.fruitbot:
            if done:
                reward = -3.0
            else:
                reward += 0.01

        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        self.replay_buffer.store_effect(self.replay_buffer_idx, a, reward, done)
        if done:
            obs = self.env.reset()
            obs = obs # why is this line here lol

        self.last_obs = obs

    def update_model(self):
        """
        Perform experience replay and train the network.

        Only done if the replay buffer has enough samples
        """
        if (self.t > self.learning_starts and
                self.t % self.learning_freq == 0 and
                self.replay_buffer.can_sample(self.batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask_batch = self.replay_buffer.sample(self.batch_size)

            self.optimizer_update(obs_batch, next_obs_batch, act_batch, rew_batch, done_mask_batch)
            
            if self.num_param_updates % self.target_update_freq == 0:
                self.q_model_target.set_weights(self.q_model.get_weights())

            if self.save_freq and self.num_param_updates % self.save_freq == 0:
                self.save(str(self.num_param_updates))

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
            # lr = self.optimizer_spec.lr_schedule.value(self.t)
            hours = (time.time() - self.start_time) / (60. * 60.)
            logz.log_tabular("Steps", self.t)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes", len(episode_rewards))
            logz.log_tabular("Exploration_Epsilon", self.exploration.value(self.t))
            # logz.log_tabular("Adam_Learning_Rate", lr)
            logz.log_tabular("Elapsed_Time_Hours", hours)
            logz.dump_tabular()

    def save(self, strr):
        self.q_model.save(os.path.join(self.logdir, 'model_%s.h5' % strr))


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
            alg.save('final')
            sys.exit()
