import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

KEY_WORD = 'MiniGrid'   # check if the env is Minigrid
OUTPUT_DIM = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    def __init__(self, env_name, PE_pos, CONV_SWITCH, output_dim = OUTPUT_DIM):
        super(QNetwork, self).__init__()
        
        self.env_name = env_name
        self.PE_pos = PE_pos
        self.CONV_SWITCH = CONV_SWITCH
        
        if CONV_SWITCH:
            INPUT_LAYER = 36 
        else:
            INPUT_LAYER = 27 
        
        if CONV_SWITCH:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(INPUT_LAYER, 256)
        self.fc2 = nn.Linear(256, 256) 
        self.fc3 = nn.Linear(256, 256) 
        self.fc4 = nn.Linear(256, 256) 
        self.fc5 = nn.Linear(256, output_dim)
    
    def positional_encoding(self, x, timestep):
        """Create a unique timestep encoding format
            
        Params
        ======
            x (tensor) : the latent space of image
            timestep (int): the current timestep
        """
        
        with torch.no_grad():
            
            # Remember the original shape of x
            original_shape = x.shape
            
            # Flatten x
            x = x.view(x.size(0), -1)
            
            if self.PE_pos == 'obs':
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
                    
            elif self.PE_pos == 'latent':
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
                        
            # Reshape x back to its original shape
            x = x.view(original_shape)

        return x.detach()


    def forward(self, timestep, state, PE_switch):
        """Builds the convolutional network"""
        
        if not self.CONV_SWITCH:
            state = state.view(state.size(0), -1)
        
        if KEY_WORD in self.env_name:
            if PE_switch and self.PE_pos == 'obs':
                state = self.positional_encoding(state, timestep)
            
            if self.CONV_SWITCH:
                state = state.permute(0, 3, 1, 2)  # Change the input shape to (batch_size, num_channels, height, width)
                x = F.relu(self.conv1(state))
                x = self.pool1(x)  # Apply max pooling
                x = F.relu(self.conv2(x))
                x = self.pool2(x)  # Apply max pooling
                x = F.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
            else:
                x = state
        else:
            x = state
        
        if PE_switch and self.PE_pos == 'latent' and self.CONV_SWITCH:
            x = self.positional_encoding(x, timestep)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        if PE_switch and self.PE_pos == 'latent' and not self.CONV_SWITCH:
            x = self.positional_encoding(x, timestep)
        
        return self.fc5(x)
