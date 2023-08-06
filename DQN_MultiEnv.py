import os
import re
import csv
import glob
import gymnasium as gym
import pandas as pd
import random
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from dqn_agent_MultiEnv import Agent_minigrid
from wrapper.Wrapper import MyViewSizeWrapper
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from minigrid.wrappers import ViewSizeWrapper, RGBImgPartialObsWrapper

FILE_PATH = './' #'/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/'
KEY_WORD = 'MiniGrid'   # check if the env is Minigrid
RENDER = False # For minigrid only
CONV_SWITCH = True
MAX_T = 200
BUDGET = 2_000
env_name = 'MiniGrid-Empty-8x8-v0' # 'MiniGrid-FourRooms-v0' # 'MiniGrid-Empty-8x8-v0' # 'MiniGrid-DoorKey-8x8-v0'

plt.ion()  # enable interactive mode

# Get a list of all CSV files that contain 'seed' in their name
files = glob.glob(FILE_PATH + '*seed*.csv')

# Loop over the list of files and remove each one
for file in files:
    try:
        os.remove(file)
        print(f"File {file} has been removed successfully")
    except:
        print(f"Error while deleting file : {file}")

# Remove the files that are trained last round
files_m = glob.glob(FILE_PATH + '*checkpoint*.pth')

# Loop over the list of files and remove each one
for file in files_m:
    try:
        os.remove(file)
        print(f"File {file} has been removed successfully")
    except:
        print(f"Error while deleting file : {file}")
    

if KEY_WORD in env_name:
    # Whether to view the env or not  
    if RENDER:
        env = gym.make(env_name, render_mode='rgb_array')
        env = MyViewSizeWrapper(env, agent_view_size=3)
    elif CONV_SWITCH:
        env = gym.make(env_name)
        env = RGBImgPartialObsWrapper(env)
    else:
        env = gym.make(env_name)
        env = ViewSizeWrapper(env, agent_view_size=3)
        
    if CONV_SWITCH:
        state_size = list(np.shape(env.observation_space.sample()['image'][32:56, 16:40, :]))
    else:
        state_size = list(np.shape(env.observation_space.sample()['image']))
    action_size = env.action_space.n
else:
    env = gym.make(env_name)
    
    state_size = np.shape(env.observation_space.sample())
    action_size = env.action_space.n

def compute_custom_means(lst):
    # Get the maximum length of the sublists
    max_len = max(len(sublist) for sublist in lst)

    # Initialize a list to store the means
    means = []

    # For each position up to max_len
    for i in range(max_len):
        # Get the values at this position in each sublist (if it exists)
        values = [sublist[i] for sublist in lst if i < len(sublist)]
        # Compute the mean of the values and append it to the list of means
        means.append(np.mean(values))

    return means

def dqn(n_episodes, render, PE_switch, PE_pos, max_t=MAX_T, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    smooth_scores, scores, episode_scores = [], [], []
    smmoth_scores_window = deque(maxlen=10)
    eps = eps_start                    # initialize epsilon
    max_score, min_score = -1000, 1000
    df = pd.DataFrame()

    budget = BUDGET
    t_global = 0
    
    while t_global < budget:
        
        n_episodes += 1
        state = env.reset(seed = int(seed))
        score = 0
        
        
        try:
            for t in range(max_t):
                t_global += 1
                print(f"\rNow is step {t}", end="")
                action = agent.act(t, state, eps)
                next_state, reward, done, info, Dict = env.step(action)

                if RENDER:
                    position_info = np.array(env.agent_pos)
                    
                    if len(state) == 2:
                        plt.imshow(state[0]['env_image'])
                    else:
                        plt.imshow(state['env_image'])

                    
                    plt.title(f"Episode {n_episodes}, Step {t}, action {action}, coor {position_info}")
                    plt.pause(0.001) # pause briefly to redraw

                agent.step(state, action, reward, next_state, done, t)
                state = next_state
                score += reward
                episode_scores.append(score)
                
                if done:
                    break 
                          
            
        except KeyboardInterrupt:
            plt.ioff()
            plt.show()       
        
        print(f'\nThe reward for episode {n_episodes} is {score}')
            
        smmoth_scores_window.append(score)
        scores.append(score)              # save most recent score
        max_score = max(max_score, score) # save max score
        min_score = min(min_score, score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        if n_episodes % 10 == 0:
            smooth_scores.append(np.mean(smmoth_scores_window))
        
    if PE_switch:
        torch.save(agent.qnetwork_local.state_dict(), FILE_PATH + env_name + '_' + str(seed) + '_' + PE_pos + '_checkpoint.pth')
    else:
        torch.save(agent.qnetwork_local.state_dict(), FILE_PATH + env_name + '_' + str(seed) + '_withoutPE_checkpoint.pth')
    
    if PE_switch and PE_pos == 'obs':
        with open(FILE_PATH + 'scores_seed_'+ str(seed) + '_with_PE_obs.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Scores'])
            
            for i, score in enumerate(episode_scores):
                writer.writerow([i+1, score])
                    
    elif PE_switch and PE_pos == 'latent':
        with open(FILE_PATH + 'scores_seed_'+ str(seed) + '_with_PE_latent.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Scores'])
            
            for i, score in enumerate(episode_scores):
                writer.writerow([i+1, score])
    else:
        with open(FILE_PATH + 'scores_seed_'+ str(seed) + '_without_PE.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestep', 'Scores'])
            
            for i, score in enumerate(episode_scores):
                writer.writerow([i+1, score])
    
    return scores, max_score, min_score, smooth_scores

n_episodes = 0
return_without_PE, return_with_PE_obs, return_with_PE_latent = [], [], []

PE_adding_position = [None, 'obs', 'latent'] # [None, 'obs', 'latent']

for idx, PE_switch in enumerate([False, True, True]): # enumerate([False, True, True]):
    PE_pos = PE_adding_position[idx]
    for seed in np.random.randint(9999, size=3):
        

        agent = Agent_minigrid(state_size, action_size, seed, PE_switch, env_name, PE_pos, CONV_SWITCH)
        scores, max_score, min_score, smooth_scores = dqn(n_episodes, RENDER, PE_switch, PE_pos)
        
        if PE_switch and PE_pos == 'obs':
            return_with_PE_obs.append(scores)
            
            max_score_with_PE_obs = max_score
            min_score_with_PE_obs = min_score
            
            with open(FILE_PATH + 'rewards_seed_'+ str(seed) + '_with_PE_obs.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
                    
        elif PE_switch and PE_pos == 'latent':
            return_with_PE_latent.append(scores)
            
            max_score_with_PE_latent = max_score
            min_score_with_PE_latent = min_score
            
            with open(FILE_PATH + 'rewards_seed_' + str(seed) + '_with_PE_latent.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
            
        else:
            return_without_PE.append(scores)
            
            max_score_without_PE = max_score
            min_score_without_PE = min_score

            with open(FILE_PATH + 'rewards_seed_' + str(seed) + '_without_PE.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])

def compute_custom_means(lst):
    # Get the maximum length of the sublists
    max_len = max(len(sublist) for sublist in lst)

    # Initialize a list to store the means
    means = []

    # For each position up to max_len
    for i in range(max_len):
        # Get the values at this position in each sublist (if it exists)
        values = [sublist[i] for sublist in lst if i < len(sublist)]
        # Compute the mean of the values and append it to the list of means
        means.append(np.mean(values))

    return means

def compute_custom_means_score(lst):
    # Get the maximum length of the sublists
    max_len = max(len(sublist) for sublist in lst)

    # Initialize a list to store the means
    means = []

    # For each position up to max_len
    for i in range(max_len):
        # Get the values at this position in each sublist (if it exists)
        values = [sublist[i] for sublist in lst if i < len(sublist) and sublist[i] != 0]
        
        # If there are no non-zero values at this position, append 0 to the means
        if len(values) == 0:
            means.append(0)
        else:
            # Compute the mean of the non-zero values and append it to the list of means
            means.append(np.mean(values))

    return means

return_with_PE_obs = glob.glob('*reward*with_PE*obs*.csv')
return_with_PE_latent = glob.glob('*reward*with_PE*latent*.csv')
return_with_PE_obs_without_PE = glob.glob('*reward*without_PE*.csv')

# Create an empty list to store the numpy arrays
return_arrays_with_PE_obs, return_arrays_with_PE_latent, return_arrays_without_PE = [], [], []

# Loop through the files
for file in return_with_PE_obs:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Reward'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_with_PE_obs.append(loss)
    
# Loop through the files
for file in return_with_PE_latent:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Reward'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_with_PE_latent.append(loss)
    
for file in return_with_PE_obs_without_PE:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Reward'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_without_PE.append(loss)

mean_return_without_PE = np.array(compute_custom_means(return_arrays_without_PE))
mean_return_with_PE_obs = np.array(compute_custom_means(return_arrays_with_PE_obs))
mean_return_with_PE_latent = np.array(compute_custom_means(return_arrays_with_PE_latent))

return_without_PE_flat = [item for sublist in return_arrays_without_PE for item in sublist]
return_with_PE_flat_obs = [item for sublist in return_arrays_with_PE_obs for item in sublist]
return_with_PE_flat_latent = [item for sublist in return_arrays_with_PE_latent for item in sublist]

std_dev_without_PE = np.std(np.array(return_without_PE_flat))
std_dev_with_PE_obs = np.std(np.array(return_with_PE_flat_obs))
std_dev_with_PE_latent = np.std(np.array(return_with_PE_flat_latent))

lower_without_PE = (mean_return_without_PE - std_dev_without_PE).astype(np.float64)
upper_without_PE = (mean_return_without_PE + std_dev_without_PE).astype(np.float64)
lower_with_PE_obs = (mean_return_with_PE_obs - std_dev_with_PE_obs).astype(np.float64)
upper_with_PE_obs = (mean_return_with_PE_obs + std_dev_with_PE_obs).astype(np.float64)
lower_with_PE_latent = (mean_return_with_PE_latent - std_dev_with_PE_latent).astype(np.float64)
upper_with_PE_latent = (mean_return_with_PE_latent + std_dev_with_PE_latent).astype(np.float64)

max_score_without_PE = max(return_without_PE_flat)
min_score_without_PE = min(return_without_PE_flat)
max_score_with_PE_obs = max(return_with_PE_flat_obs)
min_score_with_PE_obs = min(return_with_PE_flat_obs)
max_score_with_PE_latent = max(return_with_PE_flat_latent)
min_score_with_PE_latent = min(return_with_PE_flat_latent)

fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(np.arange(len(mean_return_without_PE)), mean_return_without_PE, label = 'without_PE')
ax.plot(np.arange(len(mean_return_with_PE_obs)), mean_return_with_PE_obs, label = 'with_PE_obs')
ax.plot(np.arange(len(mean_return_with_PE_latent)), mean_return_with_PE_latent, label = 'with_PE_latent')

ax.fill_between(np.arange(len(mean_return_without_PE)), lower_without_PE, upper_without_PE, color='blue', alpha=0.2)
ax.fill_between(np.arange(len(mean_return_with_PE_obs)), lower_with_PE_obs, upper_with_PE_obs, color='red', alpha=0.2)
ax.fill_between(np.arange(len(mean_return_with_PE_latent)), lower_with_PE_latent, upper_with_PE_latent, color='green', alpha=0.2)

ax.axhline(y=max_score_with_PE_obs, color='r', linestyle='--', label='Max score with PE in obs')
ax.axhline(y=min_score_with_PE_obs, color='g', linestyle='--', label='Min score with PE in obs')
ax.axhline(y=max_score_with_PE_latent, color='c', linestyle='--', label='Max score with PE in latent')
ax.axhline(y=min_score_with_PE_latent, color='m', linestyle='--', label='Min score with PE in latent')
ax.axhline(y=max_score_without_PE, color='b', linestyle='--', label='Max score without PE')
ax.axhline(y=min_score_without_PE, color='y', linestyle='--', label='Min score without PE')

ax.set_ylabel('Score/Return')
ax.set_xlabel('Episode #')
ax.set_title(f"For environment '{env_name}' (3 round seed test)")

ax.legend()

plt.savefig(FILE_PATH + env_name + "_reward.png")
plt.close()

# Find all CSV files that contain 'loss_seed' in their names
files_with_PE_obs = glob.glob('*loss_seed*with_PE*obs*.csv')
files_with_PE_latent = glob.glob('*loss_seed*with_PE*latent*.csv')
files_without_PE = glob.glob('*loss_seed*without_PE*.csv')

# Create an empty list to store the numpy arrays
loss_arrays_with_PE_obs, loss_arrays_with_PE_latent, loss_arrays_without_PE = [], [], []

# Loop through the files
for file in files_with_PE_obs:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['loss'].to_numpy()
    
    # Append the numpy array to the list
    loss_arrays_with_PE_obs.append(loss)
    
for file in files_with_PE_latent:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['loss'].to_numpy()
    
    # Append the numpy array to the list
    loss_arrays_with_PE_latent.append(loss)
    
for file in files_without_PE:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['loss'].to_numpy()
    
    # Append the numpy array to the list
    loss_arrays_without_PE.append(loss)
    
mean_loss_without_PE = np.array(compute_custom_means(loss_arrays_without_PE))
mean_loss_with_PE_obs = np.array(compute_custom_means(loss_arrays_with_PE_obs))
mean_loss_with_PE_latent = np.array(compute_custom_means(loss_arrays_with_PE_latent))

loss_without_PE_flat = [item for sublist in loss_arrays_without_PE for item in sublist]
loss_with_PE_flat_obs = [item for sublist in loss_arrays_with_PE_obs for item in sublist]
loss_with_PE_flat_latent = [item for sublist in loss_arrays_with_PE_latent for item in sublist]

std_dev_without_PE = np.std(np.array(loss_without_PE_flat))
std_dev_with_PE_obs = np.std(loss_with_PE_flat_obs)
std_dev_with_PE_latent = np.std(loss_with_PE_flat_latent)

lower_without_PE = (mean_loss_without_PE - std_dev_without_PE).astype(np.float64)
upper_without_PE = (mean_loss_without_PE + std_dev_without_PE).astype(np.float64)
lower_with_PE_obs = (mean_loss_with_PE_obs - std_dev_with_PE_obs).astype(np.float64)
upper_with_PE_obs = (mean_loss_with_PE_obs + std_dev_with_PE_obs).astype(np.float64)
lower_with_PE_latent = (mean_loss_with_PE_latent - std_dev_with_PE_latent).astype(np.float64)
upper_with_PE_latent = (mean_loss_with_PE_latent + std_dev_with_PE_latent).astype(np.float64)
                    
# fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 2 columns
fig, axs = plt.subplots(figsize=(16, 9)) 

mean_loss_without_PE_series = pd.Series(mean_loss_without_PE)
mean_loss_with_PE_obs_series = pd.Series(mean_loss_with_PE_obs)
mean_loss_with_PE_latent_series = pd.Series(mean_loss_with_PE_latent)

# Calculate the rolling average
rolling_avg_without_PE = mean_loss_without_PE_series.rolling(window=100).mean()
rolling_avg_with_PE_obs = mean_loss_with_PE_obs_series.rolling(window=100).mean()
rolling_avg_with_PE_latent = mean_loss_with_PE_latent_series.rolling(window=100).mean()

# Calculate the rolling standard deviation
rolling_std_without_PE = mean_loss_without_PE_series.rolling(window=100).std()
rolling_std_with_PE_obs = mean_loss_with_PE_obs_series.rolling(window=100).std()
rolling_std_with_PE_latent = mean_loss_with_PE_latent_series.rolling(window=100).std()

# Calculate the upper and lower bounds for the plots
upper_without_PE = rolling_avg_without_PE + rolling_std_without_PE
lower_without_PE = rolling_avg_without_PE - rolling_std_without_PE

upper_with_PE_obs = rolling_avg_with_PE_obs + rolling_std_with_PE_obs
lower_with_PE_obs = rolling_avg_with_PE_obs - rolling_std_with_PE_obs

upper_with_PE_latent = rolling_avg_with_PE_latent + rolling_std_with_PE_latent
lower_with_PE_latent = rolling_avg_with_PE_latent - rolling_std_with_PE_latent

# Plot the rolling average with the standard deviation boundaries
axs.plot(np.arange(len(rolling_avg_without_PE)), rolling_avg_without_PE, color='blue', label = 'without_PE')
axs.fill_between(np.arange(len(rolling_avg_without_PE)), lower_without_PE, upper_without_PE, color='blue', alpha=0.2)

axs.plot(np.arange(len(rolling_avg_with_PE_obs)), rolling_avg_with_PE_obs, color='red', label = 'with_PE_obs')
axs.fill_between(np.arange(len(rolling_avg_with_PE_obs)), lower_with_PE_obs, upper_with_PE_obs, color='red', alpha=0.2)

axs.plot(np.arange(len(rolling_avg_with_PE_latent)), rolling_avg_with_PE_latent,color='green', label = 'with_PE_latent')
axs.fill_between(np.arange(len(rolling_avg_with_PE_latent)), lower_with_PE_latent, upper_with_PE_latent, color='green', alpha=0.2)


# Plot the rolling average
axs.plot(np.arange(len(rolling_avg_without_PE)), rolling_avg_without_PE, color='blue', label = 'without_PE')
axs.plot(np.arange(len(rolling_avg_with_PE_obs)), rolling_avg_with_PE_obs, color='red', label = 'with_PE_obs')
axs.plot(np.arange(len(rolling_avg_with_PE_latent)), rolling_avg_with_PE_latent,color='green', label = 'with_PE_latent')

axs.set_ylabel('Loss')
axs.set_xlabel('# Training Iter')
axs.legend()

fig.suptitle(f"For environment '{env_name}' (3 round seed test)")
plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure
plt.savefig(FILE_PATH + env_name + "_loss.png")
plt.close()

def rolling_average(arr, window):
    return pd.Series(arr).rolling(window=window).mean()

def rolling_std_dev(arr, window):
    return pd.Series(arr).rolling(window=window).std()

def upper_bound(arr, window):
    return rolling_average(arr, window) + rolling_std_dev(arr, window)

def lower_bound(arr, window):
    return rolling_average(arr, window) - rolling_std_dev(arr, window)

window_size = 100  # Adjust this value based on your needs

mean_return_without_PE_smooth = rolling_average(mean_return_without_PE, window_size)
upper_without_PE_smooth = upper_bound(mean_return_without_PE, window_size)
lower_without_PE_smooth = lower_bound(mean_return_without_PE, window_size)

mean_return_with_PE_obs_smooth = rolling_average(mean_return_with_PE_obs, window_size)
upper_with_PE_obs_smooth = upper_bound(mean_return_with_PE_obs, window_size)
lower_with_PE_obs_smooth = lower_bound(mean_return_with_PE_obs, window_size)

mean_return_with_PE_latent_smooth = rolling_average(mean_return_with_PE_latent, window_size)
upper_with_PE_latent_smooth = upper_bound(mean_return_with_PE_latent, window_size)
lower_with_PE_latent_smooth = lower_bound(mean_return_with_PE_latent, window_size)

length = min(len(mean_return_without_PE_smooth), len(mean_return_with_PE_obs_smooth), len(mean_return_with_PE_latent_smooth))

fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(np.arange(length), mean_return_without_PE_smooth[:length], label = 'without_PE')
ax.plot(np.arange(length), mean_return_with_PE_obs_smooth[:length], label = 'with_PE_obs')
ax.plot(np.arange(length), mean_return_with_PE_latent_smooth[:length], label = 'with_PE_latent')

ax.fill_between(np.arange(length), lower_without_PE_smooth[:length], upper_without_PE_smooth[:length], color='blue', alpha=0.2)
ax.fill_between(np.arange(length), lower_with_PE_obs_smooth[:length], upper_with_PE_obs_smooth[:length], color='red', alpha=0.2)
ax.fill_between(np.arange(length), lower_with_PE_latent_smooth[:length], upper_with_PE_latent_smooth[:length], color='green', alpha=0.2)

ax.axhline(y=max_score_with_PE_obs, color='r', linestyle='--', label='Max score with PE in obs')
ax.axhline(y=min_score_with_PE_obs, color='g', linestyle='--', label='Min score with PE in obs')
ax.axhline(y=max_score_with_PE_latent, color='c', linestyle='--', label='Max score with PE in latent')
ax.axhline(y=min_score_with_PE_latent, color='m', linestyle='--', label='Min score with PE in latent')
ax.axhline(y=max_score_without_PE, color='b', linestyle='--', label='Max score without PE')
ax.axhline(y=min_score_without_PE, color='y', linestyle='--', label='Min score without PE')

ax.set_ylabel('Score/Return')
ax.set_xlabel('Episode #')
ax.set_title(f"For environment '{env_name}' (3 round seed test)")

ax.legend()

plt.savefig(FILE_PATH + env_name + "_reward_rolling.png")
plt.close()

# SCORE
return_with_PE_obs = glob.glob('*scores*with_PE*obs*.csv')
return_with_PE_latent = glob.glob('*scores*with_PE*latent*.csv')
return_with_PE_obs_without_PE = glob.glob('*scores*without_PE*.csv')

# Create an empty list to store the numpy arrays
return_arrays_with_PE_obs, return_arrays_with_PE_latent, return_arrays_without_PE = [], [], []

# Loop through the files
for file in return_with_PE_obs:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Scores'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_with_PE_obs.append(loss)
    
# Loop through the files
for file in return_with_PE_latent:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Scores'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_with_PE_latent.append(loss)
    
for file in return_with_PE_obs_without_PE:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the 'loss' column and convert it to a numpy array
    loss = df['Scores'].to_numpy()
    
    # Append the numpy array to the list
    return_arrays_without_PE.append(loss)
    
    
mean_return_without_PE = np.array(compute_custom_means_score(return_arrays_without_PE))
mean_return_with_PE_obs = np.array(compute_custom_means_score(return_arrays_with_PE_obs))
mean_return_with_PE_latent = np.array(compute_custom_means_score(return_arrays_with_PE_latent))


mean_return_without_PE[mean_return_without_PE == 0] = None
mean_return_with_PE_obs[mean_return_with_PE_obs == 0] = None
mean_return_with_PE_latent[mean_return_with_PE_latent == 0] = None


fig, ax = plt.subplots(figsize=(16, 9))


ax.scatter(np.arange(len(mean_return_without_PE)), mean_return_without_PE, label = 'without_PE')
ax.scatter(np.arange(len(mean_return_with_PE_obs)), mean_return_with_PE_obs, label = 'with_PE_obs')
ax.scatter(np.arange(len(mean_return_with_PE_latent)), mean_return_with_PE_latent, label = 'with_PE_latent')


ax.set_ylabel('Score/Return')
ax.set_xlabel('Episode #')
ax.set_title(f"For environment '{env_name}' (3 round seed test)")

ax.legend(loc="best")

plt.savefig(FILE_PATH + env_name + "_score.png")
plt.close()

def new_forward(self, timestep, state, PE_switch):
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

files = glob.glob('*checkpoint*.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a dictionary to store the PCA and t-SNE results for each PE_pos
results = {'withoutPE': {'PCA': [], 't_SNE': []}, 'obs': {'PCA': [], 't_SNE': []}, 'latent': {'PCA': [], 't_SNE': []}}

# Create a colormap that transitions from blue to red
cmap = LinearSegmentedColormap.from_list('blue_to_red', ['blue', 'red'])

# Assuming your files are in a list called files
for file_name in files:
    
    pattern = r"(.+)-(.+)-(.+)_(\d+)_(.+)_checkpoint"

    match = re.match(pattern, file_name)

    env_name = match.group(1) + "-" + match.group(2) + "-" + match.group(3)
    seed = int(match.group(4))
    PE_pos = match.group(5)
    
    if PE_pos == 'withoutPE':
        PE_switch = False
    elif PE_pos == 'obs' or PE_pos == 'latent':
        PE_switch = True

    if KEY_WORD in env_name:
        # Whether to view the env or not  
        if RENDER:
            env = gym.make(env_name, render_mode='rgb_array')
            env = MyViewSizeWrapper(env, agent_view_size=3)
        elif CONV_SWITCH:
            env = gym.make(env_name)
            env = RGBImgPartialObsWrapper(env)
        else:
            env = gym.make(env_name)
            env = ViewSizeWrapper(env, agent_view_size=3)
            
        if CONV_SWITCH:
            state_size = list(np.shape(env.observation_space.sample()['image'][32:56, 16:40, :]))
        else:
            state_size = list(np.shape(env.observation_space.sample()['image']))
        action_size = env.action_space.n
    else:
        env = gym.make(env_name)
        
        state_size = np.shape(env.observation_space.sample())
        action_size = env.action_space.n

    n_episodes = 0
        
    agent = Agent_minigrid(state_size, action_size, seed, PE_switch, env_name, PE_pos, CONV_SWITCH)
    
    if device.type == 'cpu':
        agent.qnetwork_local.load_state_dict(torch.load(file_name, map_location=torch.device('cpu'))) # torch.load(file_name, map_location=torch.device('cpu')
    else:
        agent.qnetwork_local.load_state_dict(torch.load(file_name))

    agent.qnetwork_local.forward = new_forward.__get__(agent.qnetwork_local, agent.qnetwork_local.__class__)

    # Collect the states and their corresponding feature representations and state values
    states = []
    features = []
    state_values = []
    for _ in range(100):
        for i in range(200):
            state = env.reset(seed=int(seed) if not CONV_SWITCH else None)
            
            if CONV_SWITCH:
                processed_state = state[0]['image'][32:56, 16:40, :]
            else:
                processed_state = state[0]['image']
            
            states.append(processed_state)
            state_tensor = torch.from_numpy(processed_state).float().unsqueeze(0).to(device)
            
            feature = agent.qnetwork_local(i, state_tensor, False).detach().cpu().numpy()
            features.append(feature)
            
            # Calculate the state value
            state_value = agent.qnetwork_local(i, state_tensor, False).max(1)[0].detach().cpu().numpy()
            state_values.append(state_value)


    # Convert the lists to numpy arrays
    states = np.array(states)
    features = np.array(features)
    state_values = np.array(state_values)
    features = features.reshape(features.shape[0], -1)
    
    if np.all(features == 0):
        print(f'Skipping PCA and t-SNE for {PE_pos} - Seed {seed} due to zero features.')
        # Create empty results with title for PCA and t-SNE
        results[PE_pos]['PCA'].append((None, None, seed))
        results[PE_pos]['t_SNE'].append((None, None, seed))
        continue

    # Apply PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)

    # Normalize the state values to a range between 0 and 1
    state_values_normalized = (state_values - state_values.min()) / (state_values.max() - state_values.min() + 1e-10)

    # Generate colors for the points
    # cmap = matplotlib.colormaps.get_cmap('bwr')
    colors = cmap(state_values_normalized)

    # Store the PCA results
    results[PE_pos]['PCA'].append((principalComponents, colors, seed))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)

    embedded_features = tsne.fit_transform(features.reshape(features.shape[0], -1))

    # Normalize the state values to a range between 0 and 1
    state_values_normalized = (state_values - state_values.min()) / (state_values.max() - state_values.min() + 1e-10)

    # cmap = matplotlib.colormaps.get_cmap('br')
    colors = cmap(state_values_normalized)

    # Store the t-SNE results
    results[PE_pos]['t_SNE'].append((embedded_features, colors, seed))

# Create the plots after all the results have been collected
fig, axs = plt.subplots(6, 3, figsize=(15, 30), dpi=100)

for i, PE_pos in enumerate(['withoutPE', 'obs', 'latent']):
    for j, method in enumerate(['PCA', 't_SNE']):
        for k, (result, colors, seed) in enumerate(results[PE_pos][method]):
            axs[i*2+j, k].set_title(f'{PE_pos} - {method} - Plot {k+1} - Seed {seed}')
            if result is None:
                axs[i*2+j, k].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axs[i*2+j, k].transAxes)
                continue
            axs[i*2+j, k].scatter(result[:,0], result[:,1], c=colors)
            axs[i*2+j, k].set_xlabel(f'{method} Component 1')
            axs[i*2+j, k].set_ylabel(f'{method} Component 2')
            for tick in axs[i*2+j, k].get_xticklabels():
                tick.set_rotation(45)  # rotate x-axis labels 45 degrees

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # adjust the space between plots
plt.savefig(FILE_PATH + env_name + '-PCA_t_SNE_plot')
plt.show()
plt.close()

print("ALL DONE")

# os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")
