import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

class DQN(nn.Module):
    def __init__(self, frame_stack_size, img_size, output_layer_size):
        super().__init__()
        
        self.output_layer_size = output_layer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=frame_stack_size, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        with torch.no_grad():
           dummy_input = torch.zeros((1,frame_stack_size, img_size, img_size))
           dummy_output = self.convolutions(dummy_input)
           linear_input_size = np.prod(dummy_output.size()[1:])
           
        # self.dnn = nn.Sequential(
        #     nn.Linear(linear_input_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.output_layer_size)
        # )
        self.value = nn.Sequential(
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256, self.output_layer_size)
        )
        
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
        
    def forward(self, state):
        
        if state.ndim == 3: #if state is 3D, add a batch dimension
            state = state.unsqueeze(0)
        
        conv_out = self.convolutions(state) #convolutional layers
        
        
        # flat = conv_out.view(state.shape[0], -1) #flatten the output
        # action_output = self.dnn(flat) #fully connected layers        
        # return action_output
        
        value  = self.value(conv_out.view(state.shape[0], -1))
        advantage = self.advantage(conv_out.view(state.shape[0], -1))
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values