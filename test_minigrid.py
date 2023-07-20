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
from minigrid.wrappers import ViewSizeWrapper

env_name = 'MiniGrid-Empty-8x8-v0'

if os.path.exists(str(env_name) + "checkpoint.pth"):
  os.remove(str(env_name) + "checkpoint.pth")
  print("The file is deleted")
else:
  print("The file does not exist")

env = gym.make(env_name)
# env = gym.make(env_name)
env = ViewSizeWrapper(env, agent_view_size=3)
env.reset()
  
agent = Agent_minigrid(state_size=list(np.shape(env.observation_space.sample()['image'])), action_size=env.action_space.n, seed=0)

n_episodes=1000

def dqn(n_episodes, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
            print(f"\rNow is step {t}", end="")
            action = agent.act(t, state, eps)
            next_state, reward, done, info, Dict = env.step(action)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            
            if done:
                break 
            
            
        print(f'\nThe reward for episode {i_episode} is {score}')
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

fig = plt.figure(figsize=(16, 9))
plt.plot(np.arange(len(scores)), scores)
plt.plot([i for i in range(0, n_episodes, 10)], smooth_scores, color = 'orange') 
plt.axhline(y=max_score, color='r', linestyle='--')
plt.axhline(y=min_score, color='g', linestyle='--')
plt.ylabel('Score/Return')
plt.xlabel('Episode #')
plt.title(f"For environment '{env_name}' ")
plt.savefig("./" + env_name + ".png")

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