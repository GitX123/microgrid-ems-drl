'''
class
- Buffer
- PrioritizedReplayBuffer
- ReplayBuffer
'''
import numpy as np
import scipy.signal
from typing import Dict

class Buffer:
    def __init__(self, buffer_size, state_seq_shape, state_fnn_shape, n_actions, gamma=0.99, lam=0.97):
        self.trajectory_start_idx = 0
        self.buffer_counter = 0
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam

        # transition
        self.state_seq_buffer = np.zeros((self.buffer_size, *state_seq_shape))
        self.state_fnn_buffer = np.zeros((self.buffer_size, *state_fnn_shape))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros((self.buffer_size, 1))

        self.state_value_buffer = np.zeros((buffer_size, 1))
        self.action_logprob_buffer = np.zeros((buffer_size, 1))
        self.return_buffer = np.zeros((buffer_size, 1))
        self.advantage_buffer = np.zeros((buffer_size, 1))
    
    def clear(self):
        self.trajectory_start_idx = 0
        self.buffer_counter = 0
        self.state_seq_buffer = np.zeros_like(self.state_seq_buffer)
        self.state_fnn_buffer = np.zeros_like(self.state_fnn_buffer)
        self.action_buffer = np.zeros_like(self.action_buffer)
        self.reward_buffer = np.zeros_like(self.reward_buffer)
        self.state_value_buffer = np.zeros_like(self.state_value_buffer)
        self.action_logprob_buffer = np.zeros_like(self.action_logprob_buffer)
        self.return_buffer = np.zeros_like(self.return_buffer)
        self.advantage_buffer = np.zeros_like(self.advantage_buffer)

    def discounted_cumulative_sums(self, x_arr, discount):
        gae = scipy.signal.lfilter([1], [1, float(-discount)], x_arr[::-1], axis=0)[::-1]
        return gae.reshape((len(gae), 1))

    def finish_trajectory(self, last_value):
        trajectory_end_idx = self.buffer_counter
        path_slice = slice(self.trajectory_start_idx, trajectory_end_idx)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        state_values = np.append(self.state_value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * state_values[1:] - state_values[:-1]
        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma*self.lam)
        self.return_buffer[path_slice] = self.advantage_buffer[path_slice] + self.state_value_buffer[path_slice]

    def sample(self, batch_size=32):
        batch_starts = np.arange(0, self.buffer_size, batch_size)
        batch_indices = np.arange(self.buffer_size)
        np.random.shuffle(batch_indices)
        batches = [batch_indices[batch_start: batch_start+batch_size] for batch_start in batch_starts]

        return self.state_seq_buffer, \
            self.state_fnn_buffer, \
            self.action_buffer, \
            self.action_logprob_buffer, \
            self.return_buffer, \
            self.advantage_buffer, \
            batches

    def store_transition(self, state_seq, state_fnn, action, reward, state_value, action_logprob):
        idx = self.buffer_counter
        self.state_seq_buffer[idx] = state_seq
        self.state_fnn_buffer[idx] = state_fnn
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.state_value_buffer[idx] = state_value
        self.action_logprob_buffer[idx] = action_logprob

        self.buffer_counter += 1

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_seq_shape, state_fnn_shape, n_actions, alpha=0.6, beta=0.4):
        # params
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.alpha = alpha
        self.beta = beta
        
        # transition
        self.state_seq_buffer = np.zeros((self.buffer_size, *state_seq_shape))
        self.state_fnn_buffer = np.zeros((self.buffer_size, *state_fnn_shape))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_seq_buffer = np.zeros((self.buffer_size, *state_seq_shape))
        self.next_state_fnn_buffer = np.zeros((self.buffer_size, *state_fnn_shape))

        # sum tree
        n_node = buffer_size * 2 - 1
        self.sum_tree = np.zeros(n_node)

    def get_leaf(self, cdf):
        idx = self._retrieve(0, cdf)
        return idx

    def get_max_priority(self):
        max_p = np.max(self.sum_tree[-self.buffer_size:])
        if max_p == 0:
            max_p = 1.
        return max_p

    def sample(self, batch_size=32):
        idxs = np.zeros(batch_size, dtype=np.int32)
        trans_idxs = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros((batch_size, 1))

        trans_idx_start = self.buffer_size - 1
        trans_idx_end = trans_idx_start + min(self.buffer_counter, self.buffer_size)
        min_prob = np.min(self.sum_tree[trans_idx_start: trans_idx_end]) / self.sum_tree[0]
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)

        total_p = self.sum_tree[0]
        segment_size = total_p / batch_size
        for i in range(batch_size):
            segment_low = i * segment_size
            segment_high = (i + 1) * segment_size
            cdf = np.random.uniform(low=segment_low, high=segment_high)

            idx = self.get_leaf(cdf)
            idxs[i] = idx
            trans_idxs[i] = idx - self.buffer_size + 1
            
            prob = self.sum_tree[idx] / self.sum_tree[0]
            weights[i] = np.power(self.buffer_size * prob, -self.beta) / max_weight
        
        return self.state_seq_buffer[trans_idxs], \
            self.state_fnn_buffer[trans_idxs], \
            self.action_buffer[trans_idxs], \
            self.reward_buffer[trans_idxs], \
            self.next_state_seq_buffer[trans_idxs], \
            self.next_state_fnn_buffer[trans_idxs], \
            idxs, \
            weights

    def schedule_beta(self, beta_inc):
        self.beta = min(self.beta + beta_inc, 1.)

    def store_transition(self, state_seq, state_fnn, action, reward, next_state_seq, next_state_fnn):
        # transition
        transition_idx = self.buffer_counter % self.buffer_size
        self.state_seq_buffer[transition_idx] = state_seq
        self.state_fnn_buffer[transition_idx] = state_fnn
        self.action_buffer[transition_idx] = action
        self.reward_buffer[transition_idx] = reward
        self.next_state_seq_buffer[transition_idx] = next_state_seq
        self.next_state_fnn_buffer[transition_idx] = next_state_fnn

        # priority
        tree_idx = transition_idx + self.buffer_size - 1
        priority = self.get_max_priority()
        self.update_tree(tree_idx, priority)

        self.buffer_counter += 1

    def update_tree(self, idx, priority):
        new_p = np.power(priority, self.alpha)
        change = new_p - self.sum_tree[idx]
        self.sum_tree[idx] = new_p
        self._propogate(idx, change)

    def _propogate(self, idx, change):
        parent_idx = (idx - 1) // 2
        self.sum_tree[parent_idx] += change
        if parent_idx != 0:
            self._propogate(parent_idx, change)

    def _retrieve(self, idx, cdf):
        l_child_idx = 2 * idx + 1
        r_child_idx = l_child_idx + 1

        if l_child_idx >= len(self.sum_tree):
            return idx
        elif cdf <= self.sum_tree[l_child_idx]:
            return self._retrieve(l_child_idx, cdf)
        else:
            return self._retrieve(r_child_idx, cdf - self.sum_tree[l_child_idx])

class ReplayBuffer:
    def __init__(self, buffer_size, state_seq_shape, state_fnn_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        
        self.state_seq_buffer = np.zeros((self.buffer_size, *state_seq_shape))
        self.state_fnn_buffer = np.zeros((self.buffer_size, *state_fnn_shape))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_seq_buffer = np.zeros((self.buffer_size, *state_seq_shape))
        self.next_state_fnn_buffer = np.zeros((self.buffer_size, *state_fnn_shape))

    def store_transition(self, state_seq, state_fnn, action, reward, next_state_seq, next_state_fnn):
        index = self.buffer_counter % self.buffer_size

        self.state_seq_buffer[index] = state_seq
        self.state_fnn_buffer[index] = state_fnn
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_seq_buffer[index] = next_state_seq
        self.next_state_fnn_buffer[index] = next_state_fnn

        self.buffer_counter += 1

    def sample(self, batch_size) -> Dict:
        sample_range = min(self.buffer_counter, self.buffer_size)
        batch_indices = np.random.choice(sample_range, size=batch_size)

        state_seq_batch = self.state_seq_buffer[batch_indices]
        state_fnn_batch = self.state_fnn_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_seq_batch = self.next_state_seq_buffer[batch_indices]
        next_state_fnn_batch = self.next_state_fnn_buffer[batch_indices]

        return state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch