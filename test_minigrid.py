import os
import gymnasium as gym
import minigrid
# import gym_minigrid
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
from dqn_agent_minigrid import Agent_minigrid
import matplotlib.pyplot as plt

env_name = 'MiniGrid-Empty-8x8-v0'

env = gym.make(env_name, render_mode='human')
env.reset()
  
agent = Agent_minigrid(state_size=list(np.shape(env.observation_space.sample()['image'])), action_size=env.action_space.n, seed=0)



def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            print(f"\rNow is step {t}", end="")
            action = agent.act(state, eps)
            next_state, reward, done, info, Dict = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        print(f'\nThe reward for episode {i_episode} is {score}')
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

print("Done")

# if os.path.exists(str(env_name) + "checkpoint.pth"):
#   os.remove(str(env_name) + "checkpoint.pth")
#   print("The file is deleted")
# else:
#   print("The file does not exist")

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