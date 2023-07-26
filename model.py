import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork_conv(nn.Module):

    def __init__(self, PE_switch, input_channels = 3, conv_channels = 16, kernel_size = 3, stride = 1, output_dim = 7):
        super(QNetwork_conv, self).__init__()
        
        # Grayscale conversion
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size, stride, padding = 1)
        
        # Additional convolutional layers 
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 2, stride)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32) 
        self.fc3 = nn.Linear(32, output_dim)
    
    def positional_encoding(self, x, timestep):
        """Create a unique timestep encoding format
        
        Params
        ======
            x (tensor) : the latent space of image
            timestep (int): the current timestep
        """
        
        with torch.no_grad():
            if x.shape[0] == 1:
                for i in range(len(x)):
                    if i % 2 == 0:
                        x[i] += torch.sin(timestep / torch.pow(10000, 2 * torch.div(x[i], 2, rounding_mode='floor') / len(x)))
                    elif i % 2 != 0:
                        x[i] += torch.cos(timestep / torch.pow(10000, 2 * torch.div(x[i], 2, rounding_mode='floor') / len(x)))
            else:
                for i in range(len(x)):
                    if i % 2 == 0:
                        x[i] += torch.sin(timestep[i] / torch.pow(10000, 2 * torch.div(x[i], 2, rounding_mode='floor') / len(x)))
                    elif i % 2 != 0:
                        x[i] += torch.cos(timestep[i] / torch.pow(10000, 2 * torch.div(x[i], 2, rounding_mode='floor') / len(x)))

        return x.detach()


    def forward(self, timestep, state, PE_switch):
        """Builds the convolutional network"""
        
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # x = x.flatten()
        
        if PE_switch:
            x = self.positional_encoding(x, timestep)
                    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
