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
from suppl import ACTIONS, apply_decay, save_fig
from trial_functions import play_one_episode
from environment import Environment
from agents.agent_gnn2 import Agent
# from agents.agent_gcnn import Agent
import matplotlib.pyplot as plt
from collections import deque
import configparser
import numpy as np
import datetime
import os
# Additional settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))

#%% Read the parameters from the configuration file

n_episodes = args['train'].getint('n_episodes')
eps_start = args['train'].getfloat('eps_start')
eps_decay = args['train'].getfloat('eps_decay')
eps_end = args['train'].getfloat('eps_end')
max_t = args['train'].getint('max_t')

#%% Run the training

scores_history = []
scores_avg_history = []
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
        print('step: {},\tstart: {},\tend: {},\taction: {} reward: {} successful: {}, random: {}'.format(
            t, start, end, (ACTIONS[action] + ',').ljust(20), (str(reward) + ',').ljust(10), successful, was_random
        ))

    # Record scores
    scores_history.append(score)
    scores_window.append(score)
    scores_avg = np.mean(scores_window)
    scores_avg_history.append(scores_avg)
    eps = apply_decay(eps, eps_end, eps_decay)  # Apply decay

    print('episode: {}, epsilon: {:.2f}, score: {}, average score: {:.2f}\n'.format(e, eps, score, scores_avg))

    # Save NNs state
    if e % 10 == 0:
        agent.save_models()


#%% Letting the agent play for a certain number of steps to try the new protocol

scores_test = play_one_episode(env, agent, max_t)

#%% Plotting

# Scores history
plt.Figure(figsize=(6, 6))
plt.title('Training scores')
plt.plot(scores_history)
plt.xlabel('Episode')
plt.ylabel('Score')
save_fig('scores_history ' + str(datetime.datetime.now()))
plt.show()

plt.Figure(figsize=(6, 6))
plt.title('Windowed average scores')
plt.plot(scores_avg_history)
plt.xlabel('Episode')
plt.ylabel('Average score')
save_fig('scores_avg_history ' + str(datetime.datetime.now()))
plt.show()
