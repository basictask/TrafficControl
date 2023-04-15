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
from trial_functions import play_one_episode
from suppl import ACTIONS, apply_decay
from environment import Environment
from agent import Agent
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
scores_window = deque(maxlen=50)  # Keeping track of the last 100 scores

env = Environment()
agent = Agent(env.state_shape, env.action_shape, env.state_high)

for e in range(n_episodes):
    state = env.reset()
    score = 0
    for t in range(max_t):
        # Choose action based on state
        start, end, action, was_random = agent.act(state, eps)
        # Execute action in the environment
        next_state, reward, successful = env.step(start, end, action)
        # Record the info in the agent
        agent.step(state, start, end, action, reward, next_state, successful)
        # Update state
        state = next_state
        # Update score
        score += reward
        # Logging
        print('step: {}, start: {}, end: {}, action: {},\treward: {}, successful: {}, random: {}'.format(
            t, start, end, ACTIONS[action], reward, successful, was_random
        ))

    # Record scores
    scores.append(score)
    scores_window.append(score)
    scores_avg = np.mean(scores_window)
    eps = apply_decay(eps, eps_end, eps_decay)  # Apply decay

    print('episode: {}, epsilon: {:.2f}, score: {}, average score: {:.2f}'.format(e, eps, score, scores_avg))

    # Save NNs state
    if e % 10 == 0:
        torch.save(agent.qnetwork_local.state_dict(), './models/local.pth')
        torch.save(agent.qnetwork_target.state_dict(), './models/target.pth')


#%% Letting the agent play for a certain number of steps to try the new protocol

scores_test = play_one_episode(env, agent, max_t)
