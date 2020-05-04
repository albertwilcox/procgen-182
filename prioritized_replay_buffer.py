import numpy as np
import ray
from segment_tree import *


@ray.remote(num_gpus=0, num_cpus=0.05)
class PrioritizedDqnReplayBuffer(object):
    """
    Adapted from https://github.com/cyoon1729/Distributed-Reinforcement-Learning/blob/master/Apex-DQN/myutils.py
    """
    def __init__(self, size, alpha, beta, multistep_len, fruitbot=False, multi_actor=True):
        """
        alpha represents a prioritization factor when updating priorities
        0 = uniform, 1 = full prioritization
        """
        self.size = size
        self.num_in_buffer = 0
        self.alpha = alpha
        # TODO: Make beta a schedule?
        self.beta = beta
        self.multistep_len = multistep_len
        self.fruitbot = fruitbot

        self.obs = None
        self.obs_tpn = None
        self.action = None
        self.reward = None
        self.done = None
        self.total_transitions_ever = 0

        self.next_idx = 0

        self.multi_actor = multi_actor

        """
        Build segment trees of smallest power of two for importance sampling states
        """
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self.it_sum = SumSegmentTree(it_capacity)
        self.it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.0

    def store_transition(self, obs, obs_tpn, action, reward, done):
        if self.obs is None:
            self.obs = np.empty([self.size] + list(obs.shape), dtype=np.uint8 if self.fruitbot else np.float32)
            if self.multi_actor:
                self.obs_tpn = np.empty([self.size] + list(obs.shape), dtype=np.uint8 if self.fruitbot else np.float32)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = obs
        if self.multi_actor:
            self.obs_tpn[self.next_idx] = obs_tpn
        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.done[self.next_idx] = done

        self.it_sum[self.next_idx] = self.max_priority ** self.alpha
        self.it_min[self.next_idx] = self.max_priority ** self.alpha

        self.next_idx = (self.next_idx+1) % self.size
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)
        self.total_transitions_ever += 1

    def get_num_in_buffer(self):
        return self.num_in_buffer

    def _sample_proportional(self, batch_size):
        """
        Get sample before alpha factor is taken into account
        """
        total_weight = self.it_sum.sum(0, self.num_in_buffer)
        cdf = np.random.random(size=batch_size)
        idxes = self.it_sum.find_prefixsum_idx(cdf * total_weight)
        return idxes

    def _encode_sample(self, idx):
        """
        returns obs_t, obs_tpn, act_t, rew_t, done_mask_t.
        This function is here in case any special processing needs to be done.
        """
        if self.multi_actor:
            return self.obs[idx], self.obs_tpn[idx], self.action[idx], self.reward[idx], self.done[idx]
        else:
            idx_tpn = (idx + self.multistep_len) % self.num_in_buffer
            return self.obs[idx], self.obs[idx_tpn], self.action[idx], self.reward[idx], self.done[idx]

    def sample(self, batch_size):
        """
        This function is from
        https://github.com/cyoon1729/Distributed-Reinforcement-Learning/blob/master/Apex-DQN/myutils.py#L175

        beta is the prioritization factor taken when sampling
        it's an exponent, usually 0.5 (sqrt).
        0 - uniform, 1 - full importance sampling
        """

        idxes = self._sample_proportional(batch_size)
        p_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (p_min * self.num_in_buffer) ** (-self.beta)
        p_sample = self.it_sum[idxes] / self.it_sum.sum()
        weights = (p_sample * self.num_in_buffer) ** (-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)

        return encoded_sample, weights, idxes

    def add_priorities(self, idxes, priorities):
        """
        Add asynchronously received priorities from learner
        """
        idxes = idxes.astype(np.int32)

        self.it_sum[idxes] = priorities ** self.alpha
        self.it_min[idxes] = priorities ** self.alpha

        self.max_priority = max(self.max_priority, np.max(priorities))

    def add_experiences(self, sample_buf):
        """
        Add asynchronously received state transitions from agent
        """
        #print(len(sample_buf))
        for s in sample_buf:
            self.store_transition(*s)