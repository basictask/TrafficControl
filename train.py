"""
 ▄▄▄▄▄▄▄ ▄▄▄▄▄▄   ▄▄▄▄▄▄ ▄▄▄ ▄▄    ▄
█       █   ▄  █ █      █   █  █  █ █
█▄     ▄█  █ █ █ █  ▄   █   █   █▄█ █
  █   █ █   █▄▄█▄█ █▄█  █   █       █
  █   █ █    ▄▄  █      █   █  ▄    █
  █   █ █   █  █ █  ▄   █   █ █ █   █
  █▄▄▄█ █▄▄▄█  █▄█▄█ █▄▄█▄▄▄█▄█  █▄▄█
This is the file used to train the reinforcement learning agent.
"""
# Imports
from suppl import ACTIONS, apply_decay
from agent import Agent
from environment import Environment
import os
import torch
import numpy as np
from collections import deque
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))

#%% Read the parameters from the configuration file

n_episodes = args['train'].getint('n_episodes')
max_t = args['train'].getint('max_t')
eps_start = args['train'].getfloat('eps_start')
eps_end = args['train'].getfloat('eps_end')
eps_decay = args['train'].getfloat('eps_decay')

#%% Run the training

scores = []
eps = eps_start
scores_window = deque(maxlen=100)  # Keeping track of the last 100 scores

env = Environment()
agent = Agent(env.state_shape, env.action_shape)

for e in range(n_episodes):
    state = env.reset()
    score = 0
    for t in range(max_t):
        start, end, action = agent.act(state, eps)  # Choose action based on state
        print(f'start: {start}, end: {end}, action: {ACTIONS[action]}')
        next_state, reward = env.step(start, end, action)  # Execute action in the environment
        agent.step(state, start, end, action, reward, next_state)  # Record info in agent
        state = next_state
        score += reward

    # Record scores
    scores.append(score)
    scores_window.append(score)
    scores_avg = np.mean(scores_window)
    eps = apply_decay(eps, eps_end, eps_decay)  # Apply decay

    print(f'episode: {e}, score: {score}, average score: {scores_avg}')

    # Save NNs state
    if e % 10 == 0:
        torch.save(agent.qnetwork_local.state_dict(), './models/local.pth')
        torch.save(agent.qnetwork_target.state_dict(), './models/target.pth')
