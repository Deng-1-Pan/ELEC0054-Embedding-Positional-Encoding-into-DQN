import os
import csv
import glob
import gymnasium as gym
import pandas as pd
import random
import torch
import numpy as np
from collections import deque
from dqn_agent_minigrid import Agent_minigrid
from wrapper.Wrapper import MyViewSizeWrapper
import matplotlib.pyplot as plt
from minigrid.wrappers import ViewSizeWrapper

KEY_WORD = 'MiniGrid'   # check if the env is Minigrid
RENDER = False # For minigrid only

plt.ion()  # enable interactive mode

# Get a list of all CSV files that contain 'seed' in their name
files = glob.glob('*seed*.csv')

# Loop over the list of files and remove each one
for file in files:
    try:
        os.remove(file)
        print(f"File {file} has been removed successfully")
    except:
        print(f"Error while deleting file : {file}")

env_name = "LunarLander-v2" # 'MiniGrid-Empty-8x8-v0'

# Remove the files that are trained last round
if os.path.exists(str(env_name) + "checkpoint.pth"):
    os.remove(str(env_name) + "checkpoint.pth")
  
    print("The file is deleted")
    
if os.path.exists('loss.csv'):
    os.remove("loss.csv")
    print("The file is deleted")
    
if os.path.exists('rewards.csv'):
    os.remove("rewards.csv")
    print("The file is deleted")

  
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

def dqn(n_episodes, render, PE_switch, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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

    budget = 20_000
    t_global = 0
    
    while t_global < budget:
        
        n_episodes += 1
        state = env.reset()
        score = 0
        
        try:
            for t in range(max_t):
                t_global += 1
                print(f"\rNow is step {t}", end="")
                action = agent.act(t, state, PE_switch, eps)
                next_state, reward, done, info, Dict = env.step(action)

                if render and env_name == 'MiniGrid-Empty-8x8-v0':
                    position_info = np.array(env.agent_pos)
                    
                    if len(state) == 2:
                        plt.imshow(state[0]['env_image'])
                    else:
                        plt.imshow(state['env_image'])

                    
                    plt.title(f"Episode {n_episodes}, Step {t}, action {action}, coor {position_info}")
                    plt.pause(0.001) # pause briefly to redraw

                agent.step(state, action, reward, next_state, done, t, PE_switch)
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
        
    torch.save(agent.qnetwork_local.state_dict(), str(env_name) + 'checkpoint.pth')
    
    return scores, max_score, min_score, smooth_scores

return_without_PE, return_with_PE = [], []

for PE_switch in [False, True]:
    for seed in np.random.randint(9999, size=3):

        agent = Agent_minigrid(state_size=state_size, action_size=action_size, seed=seed, PE_switch = PE_switch, env_name = env_name)
        scores, max_score, min_score, smooth_scores = dqn(n_episodes, RENDER, PE_switch)
        
        if PE_switch:
            return_with_PE.append(scores)
            
            max_score_with_PE = max_score
            min_score_with_PE = min_score
            
            with open('rewards_seed_'+ str(seed) + '_with_PE.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])            
            
        else:
            return_without_PE.append(scores)
            
            max_score_without_PE = max_score
            min_score_without_PE = min_score

            with open('rewards_seed_'+ str(seed) + '_without_PE.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])
                
                for i, reward in enumerate(scores):
                    writer.writerow([i+1, reward])
                    
mean_return_without_PE = np.array(compute_custom_means(return_without_PE))
mean_return_with_PE = np.array(compute_custom_means(return_with_PE))

return_without_PE_flat = [item for sublist in return_without_PE for item in sublist]
return_with_PE_flat = [item for sublist in return_with_PE for item in sublist]

std_dev_without_PE = np.std(np.array(return_without_PE_flat))
std_dev_with_PE = np.std(np.array(return_with_PE_flat))

lower_without_PE = (mean_return_without_PE - std_dev_without_PE).astype(np.float64)
upper_without_PE = (mean_return_without_PE + std_dev_without_PE).astype(np.float64)
lower_with_PE = (mean_return_with_PE - std_dev_with_PE).astype(np.float64)
upper_with_PE = (mean_return_with_PE + std_dev_with_PE).astype(np.float64)
                    
fig, ax = plt.subplots(figsize=(16, 9))


ax.plot(np.arange(len(mean_return_without_PE)), mean_return_without_PE, label = 'without_PE')
ax.plot(np.arange(len(mean_return_with_PE)), mean_return_with_PE, label = 'with_PE')

ax.fill_between(np.arange(len(mean_return_without_PE)), lower_without_PE, upper_without_PE, color='blue', alpha=0.2)
ax.fill_between(np.arange(len(mean_return_with_PE)), lower_with_PE, upper_with_PE, color='red', alpha=0.2)


ax.axhline(y=max_score_with_PE, color='r', linestyle='--', label='Max score with PE')
ax.axhline(y=min_score_with_PE, color='g', linestyle='--', label='Min score with PE')
ax.axhline(y=max_score_without_PE, color='b', linestyle='--', label='Max score without PE')
ax.axhline(y=min_score_without_PE, color='y', linestyle='--', label='Min score without PE')


ax.set_ylabel('Score/Return')
ax.set_xlabel('Episode #')
ax.set_title(f"For environment '{env_name}' (3 round seed test)")

ax.legend()

plt.savefig("./" + env_name + ".png")
plt.close()