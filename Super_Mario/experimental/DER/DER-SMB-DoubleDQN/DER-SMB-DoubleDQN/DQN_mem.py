import numpy as np

class Replay_Memory:
    def __init__(self, capacity, state_shape=(4, 84, 84)):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.float16)  # normalized states
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.index = 0
        self.full = False

    def store_memory(self, state, action, reward, next_state, terminal):
        idx = self.index % self.capacity
        self.states[idx] = state.astype(np.float16)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.astype(np.float16)
        self.terminals[idx] = terminal
        self.index += 1
        if self.index >= self.capacity:
            self.full = True

    def random_memory_batch(self, batch_size):
        max_index = self.capacity if self.full else self.index
        indices = np.random.choice(max_index, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices]
        )