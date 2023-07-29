import os
import csv
import glob
import gymnasium as gym
import pandas as pd
import random
import torch
import numpy as np
from collections import deque
from dqn_agent_MultiEnv import Agent_minigrid
from wrapper.Wrapper import MyViewSizeWrapper
import matplotlib.pyplot as plt
from minigrid.wrappers import ViewSizeWrapper

KEY_WORD = 'MiniGrid'   # check if the env is Minigrid
RENDER = False # For minigrid only

plt.ion()  # enable interactive mode

# Get a list of all CSV files that contain 'seed' in their name
files = glob.glob('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/*seed*.csv')

# Loop over the list of files and remove each one
for file in files:
    try:
        os.remove(file)
        print(f"File {file} has been removed successfully")
    except:
        print(f"Error while deleting file : {file}")

env_name = 'MiniGrid-Empty-8x8-v0'

# Remove the files that are trained last round
if os.path.exists('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/' + str(env_name) + "checkpoint.pth"):
    os.remove('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/' + str(env_name) + "checkpoint.pth")
  
    print("The file is deleted")
    
# if os.path.exists('loss.csv'):
#     os.remove("loss.csv")
#     print("The file is deleted")
    
# if os.path.exists('rewards.csv'):
#     os.remove("rewards.csv")
#     print("The file is deleted")

  
if KEY_WORD in env_name:
    # Whether to view the env or not  
    if RENDER:
        env = gym.make(env_name, render_mode='rgb_array')
        env = MyViewSizeWrapper(env, agent_view_size=3)
    else:
        env = gym.make(env_name)
        env = ViewSizeWrapper(env, agent_view_size=3)
        
    state_size = list(np.shape(env.observation_space.sample()['image']))
    action_size = env.action_space.n
else:
    env = gym.make(env_name)
    
    state_size = np.shape(env.observation_space.sample())
    action_size = env.action_space.n
    

env.reset()

n_episodes = 0

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

def dqn(n_episodes, render, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    smooth_scores, scores = [], []
    smmoth_scores_window = deque(maxlen=10)
    eps = eps_start                    # initialize epsilon
    max_score, min_score = -1000, 1000

    budget = 2_000
    t_global = 0
    
    while t_global < budget:
        
        n_episodes += 1
        state = env.reset()
        score = 0
        
        try:
            for t in range(max_t):
                t_global += 1
                print(f"\rNow is step {t}", end="")
                action = agent.act(t, state, eps)
                next_state, reward, done, info, Dict = env.step(action)

                if render and env_name == 'MiniGrid-Empty-8x8-v0':
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
        
    torch.save(agent.qnetwork_local.state_dict(), '/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/' + str(env_name) + 'checkpoint.pth')
    
    return scores, max_score, min_score, smooth_scores

return_without_PE, return_with_PE_obs, return_with_PE_latent = [], [], []

PE_adding_position = [None, 'obs', 'latent']

for idx, PE_switch in enumerate([False, True, True]):  # [False, True]
    PE_pos = PE_adding_position[idx]
    for seed in np.random.randint(9999, size=3):
        

        agent = Agent_minigrid(state_size=state_size, action_size=action_size, seed=seed, PE_switch = PE_switch, env_name = env_name, PE_pos = PE_pos)
        scores, max_score, min_score, smooth_scores = dqn(n_episodes, RENDER)
        
        if PE_switch and PE_pos == 'obs':
            return_with_PE_obs.append(scores)
            
            max_score_with_PE_obs = max_score
            min_score_with_PE_obs = min_score
            
            with open('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/rewards_seed_'+ str(seed) + '_with_PE_obs.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
        elif PE_switch and PE_pos == 'latent':
            return_with_PE_latent.append(scores)
            
            max_score_with_PE_latent = max_score
            min_score_with_PE_latent = min_score
            
            with open('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/rewards_seed_'+ str(seed) + '_with_PE_latent.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
            
        else:
            return_without_PE.append(scores)
            
            max_score_without_PE = max_score
            min_score_without_PE = min_score

            with open('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/rewards_seed_'+ str(seed) + '_without_PE.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
                    

mean_return_without_PE = np.array(compute_custom_means(return_without_PE))
mean_return_with_PE_obs = np.array(compute_custom_means(return_with_PE_obs))
mean_return_with_PE_latent = np.array(compute_custom_means(return_with_PE_latent))

return_without_PE_flat = [item for sublist in return_without_PE for item in sublist]
return_with_PE_flat_obs = [item for sublist in return_with_PE_obs for item in sublist]
return_with_PE_flat_latent = [item for sublist in return_with_PE_latent for item in sublist]

std_dev_without_PE = np.std(np.array(return_without_PE_flat))
std_dev_with_PE_obs = np.std(np.array(return_with_PE_flat_obs))
std_dev_with_PE_latent = np.std(np.array(return_with_PE_flat_latent))

lower_without_PE = (mean_return_without_PE - std_dev_without_PE).astype(np.float64)
upper_without_PE = (mean_return_without_PE + std_dev_without_PE).astype(np.float64)
lower_with_PE_obs = (mean_return_with_PE_obs - std_dev_with_PE_obs).astype(np.float64)
upper_with_PE_obs = (mean_return_with_PE_obs + std_dev_with_PE_obs).astype(np.float64)
lower_with_PE_latent = (mean_return_with_PE_latent - std_dev_with_PE_latent).astype(np.float64)
upper_with_PE_latent = (mean_return_with_PE_latent + std_dev_with_PE_latent).astype(np.float64)

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

plt.savefig("/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/" + env_name + "_reward.png")
plt.close()


# Find all CSV files that contain 'loss_seed' in their names
files_with_PE_obs = glob.glob('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/*loss_seed*with_PE*obs*.csv')
files_with_PE_latent = glob.glob('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/*loss_seed*with_PE*latent*.csv')
files_without_PE = glob.glob('/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/*loss_seed*without_PE*.csv')

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
                    
fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 2 columns

# Plot for without_PE
axs[0].plot(np.arange(len(mean_loss_without_PE)), mean_loss_without_PE, color='blue', label = 'without_PE')
axs[0].fill_between(np.arange(len(mean_loss_without_PE)), lower_without_PE, upper_without_PE, color='blue', alpha=0.2)
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('# Training Iter')
# axs[0].set_title(f"For environment '{env_name}' without PE (3 round seed test)")
axs[0].legend()

# Plot for with_PE
axs[1].plot(np.arange(len(mean_loss_with_PE_obs)), mean_loss_with_PE_obs, color='red', label = 'with_PE_obs')
axs[1].fill_between(np.arange(len(mean_loss_with_PE_obs)), lower_with_PE_obs, upper_with_PE_obs, color='red', alpha=0.2)
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('# Training Iter')
# axs[1].set_title(f"For environment '{env_name}' with PE in obs (3 round seed test)")
axs[1].legend()

# Plot for with_PE
axs[2].plot(np.arange(len(mean_loss_with_PE_latent)), mean_loss_with_PE_latent,color='green', label = 'with_PE_latent')
axs[2].fill_between(np.arange(len(mean_loss_with_PE_latent)), lower_with_PE_latent, upper_with_PE_latent, color='green', alpha=0.2)
axs[2].set_ylabel('Loss')
axs[2].set_xlabel('# Training Iter')
# axs[2].set_title(f"For environment '{env_name}' with PE in latent (3 round seed test)")
axs[2].legend()

plt.title("For environment '{env_name}' (3 round seed test)")
plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure
plt.savefig("/mnt/ELEC0054-Embedding-Positional-Encoding-into-DQN/" + env_name + "_loss.png")
plt.close()

# os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")