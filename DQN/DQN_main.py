# import torch

# # Create a tensor on the CPU
# x = torch.randn(3, 3)

# # Move the tensor to the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
# print(x)
# print("DQN running on",torch.cuda.get_device_name(0))

# from DQN_net import DQN
# import numpy as np
import gym

gamma = 0.95
batch_size = 128
memory_size = 1024

epsilon=1
epsilon_decay = 3000
epsilon_min= 0.05

skip_frames=4
# target_update_freq = 1000  # Update target network every N steps
rescale_factor = 1
# mini_render = False

car_racer = gym.make('CarRacing-v2', domain_randomise=True, continuous=False)
