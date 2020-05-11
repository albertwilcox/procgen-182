import uuid
import time
import pickle
import os
import sys
import gym.spaces
import itertools
import numpy as np
import random
import lib.logz
from lib.replay_buffer import *
from collections import namedtuple
from lib.utils import *
import ray


@ray.remote(num_gpus=0.1, num_cpus=0.1)
class DqnActor(object):
    def __init__(self,
                 worker_id,
                 env,
                 q_model_constructor,
                 exploration,
                 gamma,
                 multistep_len=1,
                 logdir=None,
                 fruitbot=False,
                 dist_param=0,
                 dist_v_min=0,
                 dist_v_max=0,
                 load_from=None):
        import tensorflow as tf
        import procgen
        import gym
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
        exploration: Schedule
            schedule for probability of choosing random action.
        logdir: str
            Where we save the results for plotting later.
        """
        #env = gym.make("procgen:procgen-fruitbot-v0", distribution_mode='easy')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

        # Training params
        self.exploration = exploration
        self.env = env

        # Model params
        self.fruitbot = fruitbot
        self.gamma = gamma

        # Distributional DQN specific parameters
        if dist_param:
            self.dist_param = dist_param
            self.dist_v_min = dist_v_min
            self.dist_v_max = dist_v_max
            self.dist_delta = (self.dist_v_max - self.dist_v_min) / self.dist_param

        # Misc Params
        self.logdir = logdir
        self.multistep_len = multistep_len

        # calculate multistep reward
        self.obs_history = [[]]*multistep_len

        # Async stuff
        self.trans_buffer = []
        self.worker_id = worker_id

        if fruitbot:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, img_c)
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

    def receive_weights(self, new_weights):
        """
        Asynchronously receive new model parameters from the learner
        """
        self.q_model.set_weights(new_weights)

    def store_recent_transition(self, last_obs, action, reward, done):
        self.obs_history.pop(0)
        self.obs_history.append([last_obs, action, reward, done])

    def get_recent_transition(self):
        """
        Returns obs, action, reward, done
        If this returns none, do not store in replay buffer
        """
        if not len(self.obs_history[0]):
            return None
        else:
            # incident observation and action
            obs, action = self.obs_history[0][:2]
            obs_tpn = self.obs_history[-1][0]

            # resultant reward and done mask over next n steps
            idxes = range(self.multistep_len)
            done = any([self.obs_history[i][3] for i in idxes])
            first_done = np.argmax([self.obs_history[i][3] for i in idxes])
            if not done:
                rew = sum([self.obs_history[i][2] * (self.gamma**i) for i in idxes])
            else:
                rew = sum([self.obs_history[i][2] * (self.gamma ** i) for i in range(first_done+1)])

            return obs, obs_tpn, action, rew, done

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
            last_obs = self.last_obs
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
                reward = -10.0
            else:
                reward += 0.1

        # Store the most recent transition, then load the encoded historical
        # transition to place into the replay buffer
        self.store_recent_transition(self.last_obs, a, reward, done)
        transition = self.get_recent_transition()
        if transition:
            self.trans_buffer.append(transition)

        if self.t % self.log_every_n_steps == 0:
            self.log_progress()
        self.t += 1

        if done:
            obs = self.env.reset()

        self.last_obs = obs

    def fill_buffer(self, num_transitions):
        """
        Queue an operation to fill the transition buffer, essentially by calling step_env n times.
        (Note that it may be called more than n times as not every step env will create a transition)
        """
        self.clear_buffer()
        while len(self.trans_buffer) < num_transitions:
            self.step_env()
        return self.return_buffer()

    def return_buffer(self):
        ret = tuple(self.trans_buffer)
        return ret

    def clear_buffer(self):
        self.trans_buffer.clear()

    def log_progress(self):
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
            logz.log_tabular("Agent", self.worker_id)
            logz.log_tabular("Steps", self.t)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes", len(episode_rewards))
            logz.log_tabular("Exploration_Epsilon", self.exploration.value(self.t))
            # logz.log_tabular("Adam_Learning_Rate", lr)
            logz.log_tabular("Elapsed_Time_Hours", hours)
            logz.dump_tabular()

    """
    Since we'll have multiple actors, it's easier to save in the learner class
    
    def save(self, strr):
        self.q_model.save(os.path.join(self.logdir, 'model_%s.h5' % strr))
    """


def learn(*args, **kwargs):
    alg = DqnActor(*args, **kwargs)

    while True:
        alg.step_env()
        # The environment should have been advanced one step (and reset if done
        # was true), and `self.last_obs` should point to new latest observation
        if alg.t > alg.max_steps:
            print("\nt = {} exceeds max_steps = {}".format(alg.t, alg.max_steps))
            alg.save('final')
            sys.exit()
