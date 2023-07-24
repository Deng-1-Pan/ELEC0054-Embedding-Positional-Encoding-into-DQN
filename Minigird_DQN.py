import os
import csv
import gymnasium as gym
import minigrid
import pandas as pd
# import gym_minigrid
import random
import torch
import numpy as np
from collections import deque
from dqn_agent_minigrid import Agent_minigrid
from wrapper.Wrapper import MyViewSizeWrapper
import matplotlib.pyplot as plt
from minigrid.wrappers import ViewSizeWrapper

plt.ion()  # enable interactive mode

env_name = 'MiniGrid-Empty-8x8-v0'

render, PE_switch = False, True

if os.path.exists(str(env_name) + "checkpoint.pth"):
    os.remove(str(env_name) + "checkpoint.pth")
  
    print("The file is deleted")
    
if os.path.exists('loss.csv'):
    os.remove("loss.csv")
    print("The file is deleted")
    
if os.path.exists('rewards.csv'):
    os.remove("rewards.csv")
    print("The file is deleted")

  

# Whether to view the env or not  
if render:
    env = gym.make(env_name, render_mode='rgb_array')
    env = MyViewSizeWrapper(env, agent_view_size=3)
else:
    env = gym.make(env_name)
    env = ViewSizeWrapper(env, agent_view_size=3)
    


env.reset()

state_size = list(np.shape(env.observation_space.sample()['image']))
action_size = env.action_space.n
  
agent = Agent_minigrid(state_size=state_size, action_size=action_size, seed=0, PE_switch = PE_switch)

n_episodes = 0

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

    budget = 2_000_000
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
                position_info = np.array(env.agent_pos)
                next_state, reward, done, info, Dict = env.step(action)

                if render:
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

scores, max_score, min_score, smooth_scores = dqn(n_episodes, render, PE_switch)

with open('rewards.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'Reward'])
    
    for i, reward in enumerate(scores):
        writer.writerow([i+1, reward])

fig = plt.figure(figsize=(16, 9))
plt.plot(np.arange(len(scores)), scores)
# plt.plot([i for i in range(0, n_episodes, 10)], smooth_scores, color = 'orange') 
plt.axhline(y=max_score, color='r', linestyle='--')
plt.axhline(y=min_score, color='g', linestyle='--')
plt.ylabel('Score/Return')
plt.xlabel('Episode #')
plt.title(f"For environment '{env_name}' ")
plt.savefig("./" + env_name + ".png")
plt.close()

data = pd.read_csv('loss.csv')
data.plot(kind='line', figsize=(16, 9))

plt.xlabel('Episode')
plt.ylabel('Loss')
plt.savefig("./" + "loss" + ".png")
plt.close()
# agent.qnetwork_local.load_state_dict(torch.load('./model/MiniGrid-Empty-8x8-v0checkpoint-4-32-32.pth'))

# view_fig, view_ax = plt.subplots()
# view_fig.show()

# for i in range(40):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _, _ = env.step(action)
    
#         img = state['image']
        
#         view_ax.clear()
#         view_ax.set_title("Agent View - Step {}".format(j))
#         view_ax.set_title("Agent View - Step {} - Reward: {:.2f}".format(j, reward))
#         view_ax.imshow(img)

#         view_fig.canvas.draw()
#         view_fig.canvas.flush_events()
#         plt.pause(0.01)
        
#         if done:
#             break 

# plt.close(view_fig)            
# env.close()



# watch an untrained agent
# state = env.reset()
# done = False
# for j in range(5000):
# # while done:
#     action = agent.act(state)
#     print(action)
#     env.render()
#     state, reward, done, info, Dict = env.step(action)
#     if done:
#         break 
        
# env.close()

# done = False
# times = 0
# while not done:
#     print(f"Round {times}")
#     action = env.action_space.sample()
#     state, reward, done, _, _ = env.step(action)
#     print(state['mission']) # mission is useless
#     env.render()
#     times += 1
    
# env.close()
# print(reward)