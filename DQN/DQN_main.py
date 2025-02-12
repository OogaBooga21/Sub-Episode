# import torch

# # Create a tensor on the CPU
# x = torch.randn(3, 3)

# # Move the tensor to the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
# print(x)
# print("DQN running on",torch.cuda.get_device_name(0))

from DQN_net import DQN
import numpy as np


online_net = DQN(4,32,5)

arr = np.arange(4096).reshape(4,32,32)
print(arr)

online_net.forward(arr)