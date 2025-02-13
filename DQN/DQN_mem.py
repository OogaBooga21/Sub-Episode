import numpy as np
import torch
import random
from collections import deque

class Replay_Memory:
    memory_fill_index =0
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def store_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.memory_fill_index +=1
    
    def random_memory_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch