import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=500):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
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
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(fc4_units, fc4_units)
        self.fc5 = nn.Linear(256, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # state = self.pos_encoder(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc2(x))
        return self.fc5(x)
    
    
class QNetwork_conv(nn.Module):

    def __init__(self, input_channels = 7, conv_channels = 1, kernel_size = 3, stride = 1, output_dim = 7):
        super(QNetwork_conv, self).__init__()
        
        # Grayscale conversion
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size, stride, padding = 1)
        
        # Additional convolutional layers 
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 2, stride)

        # Fully connected layers
        self.fc1 = nn.Linear(12, 128)  
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, state):
        """Builds the convolutional network"""
        if state.shape[1] != 7:
            print("Error is coming")
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
