import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork_conv

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_minigrid():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, PE_switch):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork_conv(PE_switch).to(device)
        self.qnetwork_target = QNetwork_conv(PE_switch).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def encode_state(self, timestep):
        """Create a unique timestep encoding format
        
        Params
        ======
            timestep (int): the current timestep
        """
        
        state_encoding = []
        
        for i in range(2):
            if i % 2 == 0:
                state_encoding.append(np.sin(timestep / np.power(10000, 2 * (i // 2) / 2)))
            elif i % 2 != 0:
                state_encoding.append(np.cos(timestep / np.power(10000, 2 * (i // 2) / 2)))

        return np.array(state_encoding)
    
    def step(self, state, action, reward, next_state, done, time_step, position_info, PE_switch):
        # Access image from state dict
        if len(state) == 2:
            state = state[0]['image']
            next_state = next_state['image']
        else:
            state = state['image']
            next_state = next_state['image']
            
        time_step = self.encode_state(time_step)
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, time_step, position_info)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, PE_switch)

    def act(self, time_step, position_info, state, PE_switch, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if len(state) == 2:
            state = state[0]['image'] 
        else:
            state = state['image'] 
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        if PE_switch:
            time_step = torch.tensor(self.encode_state(time_step)).to(device)
            position_info = torch.tensor(position_info).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(time_step, position_info, state, PE_switch)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # return np.argmax(action_values.cpu().data.numpy())
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            # return torch.from_numpy(self.action_space.sample())

    def learn(self, experiences, gamma, PE_switch):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, time_step, position_info = experiences
        
        states = states.squeeze(0)
        next_states = next_states.squeeze(0)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(time_step, position_info, next_states, PE_switch).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(time_step, position_info, states, PE_switch).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "time_step", "position_info"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, time_step, position_info):
        """Add a new experience to memory."""
        position_info = position_info.copy()
        time_step = time_step.copy()
        
        if len(state) == 2:
            state = state[0]['image']
            next_state = next_state['image']
        else:
            state = state
            next_state = next_state
            
        # Create a new namedtuple instance for this experience
        e = self.experience(state, action, reward, next_state, done, time_step, position_info)

        # Append the new experience namedtuple
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Concatenate states instead of stacking
        states = torch.from_numpy(np.concatenate([e.state for e in experiences if e is not None])).float().to(device)
        
        # Reshape state to add batch dim
        states = states.reshape(BATCH_SIZE, 3, 3, 3)
        
        # Same for next states
        next_states = torch.from_numpy(np.concatenate([e.next_state for e in experiences if e is not None])).float().to(device)  
        next_states = next_states.reshape(BATCH_SIZE, 3, 3, 3)
        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        time_step = torch.from_numpy(np.vstack([e.time_step for e in experiences if e is not None])).float().to(device)
        position_info = torch.from_numpy(np.vstack([e.position_info for e in experiences if e is not None])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, time_step, position_info)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    