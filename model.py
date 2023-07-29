import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

KEY_WORD = 'MiniGrid'   # check if the env is Minigrid
OUTPUT_DIM = 3 # 3 - MiniGrid-Empty 4 - LunarLander
INPUT_LAYER = 64 # 64 - MiniGRid-Empty 8 - LunarLander

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    def __init__(self, env_name, PE_pos, input_channels = 3, conv_channels = 16, kernel_size = 3, stride = 1, output_dim = OUTPUT_DIM):
        super(QNetwork, self).__init__()
        
        self.env_name = env_name
        self.PE_pos = PE_pos
        
        if KEY_WORD in env_name:
            # Grayscale conversion
            self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size, stride, padding = 1)
            
            # Additional convolutional layers 
            self.conv2 = nn.Conv2d(conv_channels, conv_channels, 2, stride)

        # Fully connected layers
        self.fc1 = nn.Linear(INPUT_LAYER, 32)
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
            if self.PE_pos == 'obs':
                # Create tensors of the same shape as x for the sin and cos components
                i = torch.arange(x.size(-1), device=device).float().unsqueeze(0).repeat(x.size(0), 1)
                sin_component = torch.sin(timestep / torch.pow(10000, 2 * torch.div(i, 2, rounding_mode='floor') / x.size(-1))).unsqueeze(-1).unsqueeze(-1)
                cos_component = torch.cos(timestep / torch.pow(10000, 2 * torch.div(i, 2, rounding_mode='floor') / x.size(-1))).unsqueeze(-1).unsqueeze(-1)

                # Apply the positional encoding
                if x.shape[0] == 1:
                    x[0, ::2] += sin_component[0, ::2]
                    x[0, 1::2] += cos_component[0, 1::2]
                else:
                    x[:, ::2] += sin_component[:, ::2]
                    x[:, 1::2] += cos_component[:, 1::2]
                    
            elif self.PE_pos == 'latent':
                # Create tensors of the same shape as x for the sin and cos components
                i = torch.arange(x.size(-1), device=device).float().unsqueeze(0).repeat(x.size(0), 1)
                sin_component = torch.sin(timestep / torch.pow(10000, 2 * torch.div(i, 2, rounding_mode='floor') / x.size(-1)))
                cos_component = torch.cos(timestep / torch.pow(10000, 2 * torch.div(i, 2, rounding_mode='floor') / x.size(-1)))

                # Apply the positional encoding
                if x.shape[0] == 1:
                    x[0, ::2] += sin_component[0, ::2]
                    x[0, 1::2] += cos_component[0, 1::2]
                else:
                    x[:, ::2] += sin_component[:, ::2]
                    x[:, 1::2] += cos_component[:, 1::2]

        return x.detach()


    def forward(self, timestep, state, PE_switch):
        """Builds the convolutional network"""
        
        if KEY_WORD in self.env_name:
            if PE_switch and self.PE_pos == 'obs':
                state = self.positional_encoding(state, timestep)
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
        else:
            x = state
        
        if PE_switch and self.PE_pos == 'latent':
            x = self.positional_encoding(x, timestep)
                    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
