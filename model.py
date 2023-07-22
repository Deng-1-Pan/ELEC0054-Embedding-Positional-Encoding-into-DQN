import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class QNetwork_conv(nn.Module):

    def __init__(self, input_channels = 3, conv_channels = 1, kernel_size = 3, stride = 1, output_dim = 7):
        super(QNetwork_conv, self).__init__()
        
        # Grayscale conversion
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size, stride, padding = 1)
        
        # Additional convolutional layers 
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 2, stride)

        # Fully connected layers
        self.fc1 = nn.Linear(8, 32)  
        # self.fc2 = nn.Linear(256, 256) 
        # self.fc21 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, time_step, position_info, state, PE_switch):
        """Builds the convolutional network"""
        # position_info = torch.tensor([position_info])
        
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        # If PE is added
        if PE_switch:
            time_step = time_step.to(torch.float)  
            position_info = position_info.to(torch.float)
            if x.shape[0] == 1:
                x = torch.cat((position_info.unsqueeze(0), torch.cat((time_step.unsqueeze(0), x), dim=1)), dim=1)
            else:
                x = torch.cat((position_info, torch.cat((time_step, x), dim=1)), dim=1)
                
                
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc21(x))
        
        return self.fc3(x)