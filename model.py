import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork_conv(nn.Module):

    def __init__(self, input_channels = 3, conv_channels = 1, kernel_size = 3, stride = 1, output_dim = 7):
        super(QNetwork_conv, self).__init__()
        
        # Grayscale conversion
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size, stride, padding = 1)
        
        # Additional convolutional layers 
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 2, stride)

        # Fully connected layers
        self.fc1 = nn.Linear(8, 64)  
        self.fc2 = nn.Linear(64, 64) 
        # self.fc22 = nn.Linear(32, 32) 
        self.fc3 = nn.Linear(64, output_dim)
        
    @torch.no_grad()
    def encode_state(self, state, timestep):
        """Apply positional encoding to a state.
        
        Params
        ======
            d (int): the input state
            timestep (int): the current timestep
        """
        # Use a sinusoidal encoding for each dimension of the state
        timestep = torch.tensor(timestep).to(device)
        
        state_encoding = torch.zeros(state.shape).to(device)
        
        dim = state.shape[1]
        
        for i in range(dim):
            if i % 2 == 0:
                state_encoding[0][i] += torch.sin(timestep / torch.pow(10000, torch.tensor(2 * (i // 2) / dim).to(device))) 
            elif i % 2 != 0:
                state_encoding[0][i] += torch.cos(timestep / torch.pow(10000, torch.tensor(2 * (i // 2) / dim).to(device)))
                
        state = torch.cat((state_encoding, state), dim=1)
            
        return state.detach()

    def forward(self, time_step, state):
        """Builds the convolutional network"""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x= self.encode_state(x, time_step)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc22(x))
        
        return self.fc3(x)


# DQN 模型的输入还需要进行仔细的思考，先把普通的DQN在minigrid empty跑出来 (已完成)
# DQN的输入到底是什么 单纯的image还是一个复合输入 (目前是image，后续需要考虑将PE融入也就是第四步)
# 能跑出来结果后记得缩小视野 (已完成 缩小至3x3)
# 以上的部分弄完之后可以尝试将PE融入到算法里面去 (已完成)ß