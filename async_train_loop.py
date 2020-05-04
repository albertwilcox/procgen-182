from time import sleep

from dqn_actor import *
from dqn_learner import *
from prioritized_replay_buffer import *
import copy
import ray
from gym import wrappers


@ray.remote(num_gpus=0)
class ParameterServer(object):
    """
    Keeps track of the most recent parameters, and serves them when requested.
    We could also keep a list of the n most recent model parameters and serve a random one each time.
    However, this sort of functionality is probably already covered well enough by replay buffers.
    """
    def __init__(self):
        self.params = None
        self.update_step = 0

    def update_params(self, new_params):
        self.params = new_params

    def get_params(self):
        return self.params


def get_env(env_name, seed, logdir, ind):
    if env_name == 'procgen:procgen-fruitbot-v0':
        env = gym.make(env_name, distribution_mode='easy')
    else:
        env = gym.make(env_name)

    env.seed(seed)
    expt_dir = os.path.join(logdir, "gym-agent-" + str(ind))
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=False)
    return env


def learn(
        num_actors,
        env,
        q_model_constructor,
        optimizer_params,
        exploration,
        replay_buffer_size,
        alpha,
        beta,
        batch_size,
        gamma,
        learning_starts,
        learning_freq,
        frame_history_len,
        target_update_freq,
        multistep_len,
        double_q,
        logdir,
        max_steps,
        fruitbot,
        load_from,
        save_every):

    try:
        ray.init()

        param_server = ParameterServer.remote()
        buffer = PrioritizedDqnReplayBuffer.remote(
            replay_buffer_size,
            alpha,
            beta,
            multistep_len,
            fruitbot
        )

        envs = [get_env(env, 0, logdir, i) for i in range(num_actors)]
        input_shape = envs[0].observation_space.shape

        if fruitbot:
            num_actions = 4
        else:
            num_actions = envs[0].action_space.n

        learner = DqnLearner.remote(
            q_model_constructor=q_model_constructor,
            optimizer_params=optimizer_params,
            batch_size=batch_size,
            gamma=gamma,
            target_update_freq=target_update_freq,
            input_shape=input_shape,
            num_actions=num_actions,
            multistep_len=1,
            double_q=double_q,
            logdir=None,
            fruitbot=fruitbot,
            dist_param=0,
            dist_v_min=0,
            dist_v_max=0,
            load_from=load_from,
            save_every=save_every
        )

        actors = [
            DqnActor.remote(
                worker_id=i,
                env=envs[i],
                q_model_constructor=q_model_constructor,
                exploration=copy.deepcopy(exploration),  # This is a schedule
                gamma=gamma,
                multistep_len=multistep_len,
                logdir=logdir,
                fruitbot=fruitbot,
                dist_param=0,
                dist_v_min=0,
                dist_v_max=0,
                load_from=load_from
            ) for i in range(num_actors)
        ]

        _, _ = ray.wait([actor.fill_buffer.remote(10000) for actor in actors], num_returns=num_actors)

        #sleep(6)

        for actor in actors:
            buffer.add_experiences.remote(ray.get(actor.return_buffer.remote()))

        sleep(1)

        actor_buffers = {}
        for actor in actors:
            actor_buffers[actor.fill_buffer.remote(250)] = actor

        num_parameter_updates = ray.get(learner.get_num_param_updates.remote())

        while num_parameter_updates < max_steps:
            # gather actor experience to centralized buffer
            # Do not wait for the actors if they're not ready
            bufs = ray.get(list(actor_buffers))
            for i in range(num_actors):
                buffer.add_experiences.remote(bufs[i])

                # sync learner policy with global
                learner_params = ray.get(learner.get_weights.remote())
                # sync actor policy with global policy
                actors[i].receive_weights.remote(learner_params)

                actor_buffers[actors[i].fill_buffer.remote(250)] = actors[i]
            for i in range(250):
                # Do a parameter update
                batch, indices, weights = ray.get(
                    buffer.sample.remote(batch_size)
                )
                indices, new_priorities, num_parameter_updates = ray.get(
                    learner.update_model.remote(batch, indices, weights)
                )
                buffer.add_priorities.remote(indices, new_priorities)

            if num_parameter_updates % 100 == 0:
                print(num_parameter_updates)
            if num_parameter_updates % 1000000 == 0:
                break

    finally:
        pass


