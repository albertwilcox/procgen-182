import tensorflow as tf
import os, time
from utils import *
import logz


class PPOLearner(object):

    def __init__(self,
                 env,
                 policy_func_constructor,
                 value_func_constructor,
                 logdir,
                 optimizer_params={},
                 gamma=0.999,
                 clip_epsilon=0.2,
                 policy_lr=3e-4,
                 value_lr=1e-3,
                 steps_per_epoch=4000,
                 epochs=50,
                 policy_iters=80,
                 value_iters=80,
                 adv_lambda=0.95,
                 save_freq=5,
                 fruitbot=True,
                 load_from=None
                 ):
        """
        Initialize a PPO learner

        Parameters:
        -----------
        env: gym.Env
            gym environment to train on
        policy_func_constructor: func
        value_func_constructor: func
        optimizer_params: dict
        logdir: str
        steps_per_epoch: int
            Lower bound on number of state action pairs for each epoch of training
        epochs: int
            How many epochs to train on
        gamma: float
            reward-to-go discount factor
        clip_epsilon: float
            epsilon value used for clipping change in advantage function
        policy_lr: float
            learning rate for policy model
        value_lr: float
            learning rate for value model
        policy_iters: int
            How many iterations per epoch to perform for the policy model
        value_iters: int
            How many iterations per epoch to perform for the value model
        adv_lambd: int
            lambda value to be used to compute advantage predictions based on value predictions
        save_freq: int
            How many epochs between each save
        fruitbot: boolean
            whether we are training on the fruitbot environment
        load_from: str
            directory from which to load models
        """

        self.env = env

        self.steps_per_epoch = steps_per_epoch
        self.eopchs = epochs
        self.policy_iters = policy_iters
        self.value_iters = value_iters

        self.clip_epsilon = clip_epsilon
        self.adv_lambda = adv_lambda
        self.gamma = gamma

        # Misc Params
        self.save_freq = save_freq
        self.logdir = logdir

        input_shape = self.env.observation_space.shape

        if fruitbot:
            self.num_actions = 4
        else:
            self.num_actions = self.env.action_space.n

        if load_from:
            self.policy_func = tf.keras.models.load_model(os.path.join(load_from, 'policy.h5'), compile=False)
            self.value_func = tf.keras.models.load_model(os.path.join(load_from, 'value.h5'), compile=False)
        else:
            self.policy_func = policy_func_constructor(input_shape, self.num_actions)
            self.value_func = value_func_constructor(input_shape)

        print(self.policy_func.summary())
        print(self.value_func.summary())

        opt = optimizer_params.get('type', tf.keras.optimizers.Adam)
        grad_norm_clipping = optimizer_params.get('grad_norm_clipping', 10)
        self.policy_optimizer = opt(learning_rate=policy_lr, clipnorm=grad_norm_clipping)
        self.value_optimizer = opt(learning_rate=value_lr, clipnorm=grad_norm_clipping)

        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.std_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.log_every_n_steps = 10000 if fruitbot else 1000
        self.start_time = time.time()
        self.t = 0

    def sample_trajectories(self, env):
        """
        Collect paths until we have enough timesteps, as determined by the
        length of all paths collected in this batch.
        """

        def sample_trajectory():
            obs = env.reset()
            observations, actions, rewards = [], [], []
            steps = 0
            while True:
                observations.append(obs)
                action_probs = self.policy_func(np.expand_dims(obs, axis=0))
                action_probs = action_probs[0]
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                obs, rew, done, _ = env.step(action)
                actions.append(action)
                rewards.append(rew)
                steps += 1
                if done:
                    break
            path = {"observation": np.array(observations, dtype=np.float32),
                    "reward": np.array(rewards, dtype=np.float32),
                    "action": np.array(actions, dtype=np.int8)}
            return path

        def pathlength(path):
            return len(path["reward"])

        timesteps_this_batch = 0
        paths = []
        while True:
            path = sample_trajectory()
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.steps_per_epoch:
                break
        return paths, timesteps_this_batch

    def reward_to_go(self, paths):
        for path in paths:
            rewards = path['reward']
            t = len(rewards)
            ran = np.arange(t)
            exponents = np.triu(np.tile(ran, t).reshape((t, t)) - ran.reshape((t, 1)))
            multipliers = self.gamma ** exponents
            path['reward_to_go'] = np.sum(rewards * multipliers, axis=1)

    def policy_loss(self, paths, moving_policy_func, total_timesteps):
        loss = 0
        for path in paths:
            actions = path['action']
            states = path['observation']
            rewards = path['reward']

            probs_moving = moving_policy_func(states)
            probs_stationary = self.policy_func(states)
            quotients = probs_moving / probs_stationary
            quotient = tf.gather_nd(quotients, tf.expand_dims(actions, 1), batch_dims=1)

            adv = self.advantage(states, rewards)
            clipper = self.clip(quotient, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            val = tf.math.minimum(quotient * adv, clipper * adv)
            loss -= val / total_timesteps
        return loss

    def advantage(self, states, rewards):
        values = self.value_func(states)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        # TODO: finish implementing this with the same crazy trick used for reward to go, but in tensorflow
        return 0

    # TODO: fitting the value estimator with standard regression on reward-to-go

    # TODO: update function for policy net

    @tf.function
    def clip(self, x, a, b):
        x = tf.math.minimum(x, b)
        x = tf.math.maximum(x, a)
        return x

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
            logz.log_tabular("Steps", self.t)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes", len(episode_rewards))
            logz.log_tabular("Exploration_Epsilon", self.exploration.value(self.t))
            # logz.log_tabular("Adam_Learning_Rate", lr)
            logz.log_tabular("Elapsed_Time_Hours", hours)
            logz.dump_tabular()

def learn(*args, **kwargs):
    alg = PPOLearner(*args, **kwargs)


