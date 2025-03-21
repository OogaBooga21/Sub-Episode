import numpy as np
import random
from collections import deque
import torch

class Replay_Memory:
    memory_fill_index =0
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def store_memory(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        self.memory_fill_index +=1
    
    def random_memory_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = [memory[0] for memory in batch]
        actions = [memory[1] for memory in batch]
        rewards = [memory[2] for memory in batch]
        n_states = [memory[3] for memory in batch]
        terminals = [memory[4] for memory in batch]
        
        # return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(n_states), torch.stack(terminals)
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(n_states), np.stack(terminals)
        # return states, actions, rewards, n_states, terminals
        