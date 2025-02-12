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
           
        self.dnn = nn.Sequential(
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256, self.output_layer_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
        # self.scheduler = StepLR(self.optimizer, step_size=4000, gamma=0.5)
        
        self.to(self.device) # move to gpu
        
        print(self)
        print("DQN running on ",torch.cuda.get_device_name(0)) #output whould be GPU name
        
    def forward(self, state):
        batch_size, frame_stack_size, height, width = state.size()
        
        conv_out = self.convolutions(state)
        conv_out = conv_out.view(batch_size, -1)
    
        action_output = self.dnn(conv_out)
        
        return action_output