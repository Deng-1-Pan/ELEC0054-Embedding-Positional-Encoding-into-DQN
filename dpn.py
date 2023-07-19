import os
import gym
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt

env_name = 'LunarLander-v2'
env = gym.make(env_name)
# env.reset()
env.seed(0)
print('State shape: ', env.observation_space.shape[0])
print('Number of actions: ', env.action_space.n)


if os.path.exists(str(env_name) + "checkpoint.pth"):
  os.remove(str(env_name) + "checkpoint.pth")
  print("The file is deleted")
else:
  print("The file does not exist")
  


agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

# watch an untrained agent
state = env.reset()
for j in range(2):
    action = agent.act(state, j)
    # env.render(close=True)
    state, reward, done, _ = env.step(action)
    if done:
        break 
        
env.close()

n_episodes=5000

def dqn(n_episodes, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995): # lunar lander n_episodes=2000, max_t=1000
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
    smooth_scores = []
    smmoth_scores_window = deque(maxlen=10)
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score, min_score = -1000, 1000
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, t, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        smmoth_scores_window.append(score)
        scores.append(score)              # save most recent score
        max_score = max(max_score, score) # save max score
        min_score = min(min_score, score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            smooth_scores.append(np.mean(smmoth_scores_window))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
        
    print('\nAfter {:d} episodes \tThe Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    torch.save(agent.qnetwork_local.state_dict(), str(env_name) + 'checkpoint.pth')
    return scores, max_score, min_score, smooth_scores

scores, max_score, min_score, smooth_scores = dqn(n_episodes)

# plot the scores
fig = plt.figure(figsize=(16, 9))
plt.plot(np.arange(len(scores)), scores)
plt.plot([i for i in range(0, n_episodes, 10)], smooth_scores, color = 'orange') 
plt.axhline(y=max_score, color='r', linestyle='--')
plt.axhline(y=min_score, color='g', linestyle='--')
plt.ylabel('Score/Return')
plt.xlabel('Episode #')
plt.title(f"For environment '{env_name}' ")
plt.savefig("./" + env_name + ".png")