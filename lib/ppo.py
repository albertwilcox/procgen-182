import tensorflow as tf
import os, time
from lib.utils import *
import lib.logz as logz


class PPOLearner(object):

    def __init__(self,
                 env,
                 model_constructor,
                 logdir,
                 optimizer_params={},
                 gamma=0.999,
                 adv_lambda=0.95,
                 entropy_coef=0.01,
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 latent_dim=256,
                 lr=5e-4,
                 steps_per_update=4000,
                 updates=50,
                 epochs_per_update=4,
                 batch_size=1000,
                 save_freq=5,
                 fruitbot=True,
                 load_policy_from=None,
                 load_value_from=None
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
        self.fruitbot = fruitbot

        self.steps_per_epoch = steps_per_update
        self.updates = updates
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        self.clip_epsilon = clip_epsilon
        self.adv_lambda = adv_lambda
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Misc Params
        self.save_freq = save_freq
        self.logdir = logdir

        input_shape = self.env.observation_space.shape

        if fruitbot and False:
            self.num_actions = 4
        else:
            self.num_actions = self.env.action_space.n

        if load_policy_from and load_value_from:
            # self.policy_func = tf.keras.models.load_model(load_policy_from, compile=False)
            # self.value_func = tf.keras.models.load_model(load_value_from, compile=False)
            pass
        else:
            self.model = model_constructor(input_shape, latent_dim)
            self.policy_head = tf.keras.models.Sequential([tf.keras.layers.Dense(self.num_actions, activation='softmax')])
            self.value_head = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
            self.model.build((None,) + input_shape)

        print(self.model.summary())

        # TODO: decide if this should be the way to optimize, I think it should
        opt = optimizer_params.get('type', tf.keras.optimizers.Adam)
        grad_norm_clipping = optimizer_params.get('grad_norm_clipping', 10)
        self.optimzer = opt(learning_rate=lr, clipnorm=grad_norm_clipping)
        # self.policy_optimizer = opt(learning_rate=policy_lr, clipnorm=grad_norm_clipping)
        # self.value_optimizer = opt(learning_rate=value_lr, clipnorm=grad_norm_clipping)

        self.mean_episode_reward = -float('nan')
        self.std_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.log_every_n_epochs = 1
        self.start_time = time.time()
        self.updates_done = 0

    def sample_trajectories(self):
        """
        Collect paths until we have enough timesteps, as determined by the
        length of all paths collected in this batch.
        """

        def sample_trajectory():
            obs = self.env.reset()
            observations, actions, rewards, probs, values = [], [], [], [], []
            steps = 0
            while True:
                if self.fruitbot:
                    obs = obs / 127.5 - 1

                latent = self.model(np.expand_dims(obs, axis=0))
                value = self.value_head(latent)
                action_probs = self.policy_head(latent)
                action_probs = action_probs[0]
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs.numpy())

                observations.append(obs)
                values.append(value)

                if self.fruitbot and False:
                    a_env = action * 3
                else:
                    a_env = action

                obs, rew, done, _ = self.env.step(a_env)

                # Reward engineering:
                if self.fruitbot:
                    mode = 0
                    if mode == 0:
                        if done and rew < 10.0:
                            rew = -10.0
                        else:
                            rew += 0.1
                    if mode == 1:
                        if done:
                            rew = -10
                        else:
                            rew = 1

                actions.append(action)
                probs.append(action_probs)
                rewards.append(rew)
                steps += 1
                if done:
                    break
            traj = {"state": np.array(observations, dtype=np.float32),
                    "reward": np.array(rewards, dtype=np.float32),
                    "action": np.array(actions, dtype=np.int32),
                    "probs": np.array(probs, dtype=np.float32),
                    'values': np.array(values, dtype=np.float32).squeeze(),
                    't': len(actions)}
            return traj

        timesteps_this_batch = 0
        trajectories = []
        while True:
            print('    sampling')
            traj = sample_trajectory()
            trajectories.append(traj)
            timesteps_this_batch += traj['t']
            if timesteps_this_batch > self.steps_per_epoch:
                break
        return trajectories, timesteps_this_batch

    def reward_to_go(self, trajectories):
        """
        Calculate reward to go for each state based on a list of trajectories

        Returns a list of ndarrays of reward to go for each trajectory
        """
        out = []
        for traj in trajectories:
            rewards = traj['reward']
            t = len(rewards)
            ran = np.arange(t)
            exponents = np.tile(ran, t).reshape((t, t)) - ran.reshape((t, 1))
            multipliers = np.triu(self.gamma ** exponents)
            out.append(np.sum(rewards * multipliers, axis=1))
        return out

    def advantages(self, trajectories):
        """
        Compute advantage updates as detailed on page 5 of https://arxiv.org/pdf/1707.06347.pdf

        Return a list of ndarrays holding advantages at each state
        """
        out = []
        for traj in trajectories:
            values = traj['values']
            deltas = traj['reward'][:-1] + self.gamma * values[1:] - values[:-1]
            t = traj['t'] - 1
            ran = tf.range(t)
            broadcasted = tf.broadcast_to(ran, (t, t))
            exponents = broadcasted - tf.transpose(broadcasted)
            multipliers = tf.linalg.band_part((self.gamma * self.adv_lambda) ** tf.cast(exponents, tf.float32), 0, -1)
            out.append(tf.reduce_sum(multipliers * deltas, axis=1))
            out.append([0])
        return out

    # def update_value(self, dataset):
    #
    #     @tf.function
    #     def value_step(states, reward_to_go):
    #         with tf.GradientTape() as tape:
    #             pred = self.value_func(states)
    #             loss = tf.keras.losses.MSE(reward_to_go, pred)
    #         gradients = tape.gradient(loss, self.value_func.trainable_variables)
    #         self.value_optimizer.apply_gradients(zip(gradients, self.value_func.trainable_variables))
    #
    #     it = 0
    #     while it < self.value_iters:
    #         for batch in dataset:
    #             it += 1
    #             if it > self.value_iters:
    #                 break
    #
    #             value_step(*batch)
    #
    # def update_policy(self, dataset):
    #
    #
    #
    #     # @tf.function
    #     # def policy_step(states_b, actions_b, advantages_b):
    #     #     with tf.GradientTape() as tape:
    #     #         loss = policy_loss(states_b, actions_b, advantages_b)
    #     #     gradients = tape.gradient(loss, self.policy_func.trainable_variables)
    #     #     self.policy_optimizer.apply_gradients(zip(gradients, self.policy_func.trainable_variables))
    #
    #     it = 0
    #     while it < self.policy_iters:
    #         for batch in dataset:
    #             it += 1
    #             if it > self.value_iters:
    #                 break
    #
    #             # policy_step(*batch)

    # @tf.function
    def clip(self, x: tf.Tensor, a, b):
        x = tf.math.minimum(x, b)
        x = tf.math.maximum(x, a)
        return x

    # @tf.function
    def entropy(self, probs):
        return -1 * tf.reduce_sum(tf.math.log(probs + 1e-8) * probs, axis=-1)

    # @tf.function
    def value_loss(self, latents, reward_to_go):
        pred = self.value_head(latents)
        loss = tf.keras.losses.MSE(reward_to_go, pred)
        return loss

    # @tf.function
    def policy_loss(self, latents, actions_b, advantages_b, probs_b):
        """
        Return policy function loss for a batch of size self.policy_batch_size
        """
        probs_moving = self.policy_head(latents)

        quotients = probs_moving / (probs_b + 1e-8)
        quotient = tf.gather_nd(quotients, tf.expand_dims(actions_b, 1), batch_dims=1)

        entropy = self.entropy(probs_moving)

        clipper = self.clip(quotient, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        val = tf.math.minimum(quotient * advantages_b, clipper * advantages_b) - self.entropy_coef * entropy
        return -1 * tf.reduce_mean(val)

    @tf.function
    def loss(self, states_b, actions_b, advantages_b, reward_to_go_b, probs_b):
        latents = self.model(states_b)
        value_loss = self.value_loss(latents, reward_to_go_b)
        policy_loss = self.policy_loss(latents, actions_b, advantages_b, probs_b)
        loss = policy_loss + self.value_coef * value_loss
        return loss

    def update(self, trajectories, total_timesteps):
        print('  Preprocessing')
        reward_to_go = np.concatenate(self.reward_to_go(trajectories)).astype(np.float32)
        advantages = np.concatenate(self.advantages(trajectories)).astype(np.float32)
        states = np.concatenate([traj['state'] for traj in trajectories])
        actions = np.concatenate([traj['action'] for traj in trajectories])
        probs = np.concatenate([traj['probs'] for traj in trajectories])

        ds_size = total_timesteps // self.batch_size * self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, advantages, reward_to_go, probs))
        dataset = dataset.shuffle(ds_size).batch(self.batch_size)

        print('  Optimizing')
        for _ in range(self.epochs_per_update):
            for batch in dataset:
                with tf.GradientTape(persistent=True) as tape:
                    loss = self.loss(*batch)

                model_grads = tape.gradient(loss, self.model.trainable_variables)
                policy_head_grads = tape.gradient(loss, self.policy_head.trainable_variables)
                value_head_grads = tape.gradient(loss, self.value_head.trainable_variables)

                del tape

                self.optimzer.apply_gradients(zip(model_grads, self.model.trainable_variables))
                self.optimzer.apply_gradients(zip(policy_head_grads, self.policy_head.trainable_variables))
                self.optimzer.apply_gradients(zip(value_head_grads, self.value_head.trainable_variables))

        self.updates_done += 1
        if self.save_freq and self.updates_done % self.save_freq == 0:
            self.save(str(self.updates_done))

    def log_progress(self):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
            self.std_episode_reward = np.std(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = \
                max(self.best_mean_episode_reward, self.mean_episode_reward)

        # See the `log.txt` file for where these statistics are stored.
        if self.updates_done % self.log_every_n_epochs == 0:
            # lr = self.optimizer_spec.lr_schedule.value(self.t)
            hours = (time.time() - self.start_time) / (60. * 60.)
            logz.log_tabular("Epoch", self.updates_done)
            logz.log_tabular("Avg_Last_100_Episodes", self.mean_episode_reward)
            logz.log_tabular("Std_Last_100_Episodes", self.std_episode_reward)
            logz.log_tabular("Best_Avg_100_Episodes", self.best_mean_episode_reward)
            logz.log_tabular("Num_Episodes", len(episode_rewards))
            # logz.log_tabular("Exploration_Epsilon", self.exploration.value(self.t))
            # logz.log_tabular("Adam_Learning_Rate", lr)
            logz.log_tabular("Elapsed_Time_Hours", hours)
            logz.dump_tabular()

    def save(self, strr):
        self.model.save(os.path.join(self.logdir, 'model_%s' % strr))
        self.policy_head.save(os.path.join(self.logdir, 'policy_%s' % strr))
        self.value_head.save(os.path.join(self.logdir, 'value_%s' % strr))


def learn(*args, **kwargs):
    alg = PPOLearner(*args, **kwargs)
    # alg.save('start')

    for _ in range(alg.updates):
        print('  Gathering trajectories')
        trajectories, total_timesteps = alg.sample_trajectories()
        alg.update(trajectories, total_timesteps)
        alg.log_progress()

    alg.save('final')
