import os
import csv
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

FILE_PATH = '/ELEC0054-Embedding-Positional-Encoding-into-DQN/'
ACTION_SIZE = 3 # for e-greedy
UPDATE_SOFT = True # Soft update or hard
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64 # 64        # minibatch size
LEARNING_START = 5000   # run this number of time-step before learning
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
KEY_WORD = 'MiniGrid'   # check if the env is Minigrid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_minigrid():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, PE_switch, env_name, PE_pos):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            PE_switch (bool) : switch for Positional Encoding
            env_name (str) : the name of the environmnet
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.PE_switch = PE_switch
        self.seed_record = seed
        self.env_name = env_name
        self.PE_pos = PE_pos

        # Q-Network
        self.qnetwork_local = QNetwork(env_name, PE_pos).to(device)
        self.qnetwork_target = QNetwork(env_name, PE_pos).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, env_name)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, time_step):
        if KEY_WORD in self.env_name:
            # Access image from state dict
            if len(state) == 2:
                state = state[0]['image']
                next_state = next_state['image']
            else:
                state = state['image']
                next_state = next_state['image']
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, time_step)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > LEARNING_START: #BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, time_step, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        if len(state) == 2:
            if KEY_WORD in self.env_name:
                state = state[0]['image']
            else:
                state = state[0]
        else:
            if KEY_WORD in self.env_name:
                state = state['image'] 
            else:
                state = state
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        if self.PE_switch:
            time_step = torch.tensor(time_step).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(time_step, state, self.PE_switch)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            if KEY_WORD in self.env_name:
                return random.choice(np.arange(ACTION_SIZE))
            else:
                return random.choice(np.arange(self.action_size))
            
    def write_loss_to_csv(self, loss, filename):
        # Check if file exists
        file_exists = os.path.isfile(filename)

        with open(filename, 'a') as csvfile:  # 'a' mode appends to the existing file
            headers = ['loss']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)

            # Only write headers once
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerow({'loss': loss})

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, time_step = experiences
        
        states = states.squeeze(0)
        next_states = next_states.squeeze(0)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(time_step, next_states, self.PE_switch).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(time_step, states, self.PE_switch).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        if self.PE_switch and self.PE_pos == 'obs':
            self.write_loss_to_csv(loss = loss.item(), filename=FILE_PATH + 'loss_seed_' + str(self.seed_record ) + '_with_PE_obs.csv')  # write loss to CSV
        elif self.PE_switch and self.PE_pos == 'latent':
            self.write_loss_to_csv(loss = loss.item(), filename= FILE_PATH + 'loss_seed_' + str(self.seed_record ) + '_with_PE_latent.csv')  # write loss to CSV
        else:
            self.write_loss_to_csv(loss = loss.item(), filename=FILE_PATH + 'loss_seed_' + str(self.seed_record ) + '_without_PE.csv')
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
        if UPDATE_SOFT:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        else:
            target_model.load_state_dict(local_model.state_dict())
            
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, env_name):
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "time_step"])
        self.seed = random.seed(seed)
        self.env_name = env_name
    
    def add(self, state, action, reward, next_state, done, time_step):
        """Add a new experience to memory."""
        # time_step = time_step.copy()
        
        if len(state) == 2:
            if KEY_WORD in self.env_name:
                state = state[0]['image']
                next_state = next_state['image']
            else:
                state = state[0]
        else:
            state = state
            next_state = next_state
            
        # Create a new namedtuple instance for this experience
        e = self.experience(state, action, reward, next_state, done, time_step)

        # Append the new experience namedtuple
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        if KEY_WORD in self.env_name:
            # Concatenate states instead of stacking
            states = torch.from_numpy(np.concatenate([e.state for e in experiences if e is not None])).float().to(device)
            
            # Reshape state to add batch dim
            states = states.reshape(BATCH_SIZE, 3, 3, 3)
            
            # Same for next states
            next_states = torch.from_numpy(np.concatenate([e.next_state for e in experiences if e is not None])).float().to(device)  
            next_states = next_states.reshape(BATCH_SIZE, 3, 3, 3)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        time_step = torch.from_numpy(np.vstack([e.time_step for e in experiences if e is not None])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, time_step)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    