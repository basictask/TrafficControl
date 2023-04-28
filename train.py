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
import pandas as pd

# Imports
from suppl import ACTIONS, apply_decay, save_fig
from trial_functions import play_one_episode
from environment import Environment
from agents.agent_gnn2 import Agent
# from agents.agent_gcnn import Agent
# from agents.agent_snn import Agent
# from agents.agent_enn import Agent
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
    agent.reset()
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

#%% Plotting and saving run results

# Info
architecture = agent.__module__.split('.')[-1]
timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

# Save history
agent.save_history(architecture, timestamp)

# Scores history
plt.Figure(figsize=(6, 6))
plt.title(f'Training scores ({architecture})')
plt.plot(scores_history)
plt.xlabel('Episode')
plt.ylabel('Score')
save_fig(f'scores-history_{architecture}_{timestamp}')
plt.show()

# Windowed average scores history
plt.Figure(figsize=(6, 6))
plt.title(f'Windowed average scores ({architecture})')
plt.plot(scores_avg_history)
plt.xlabel('Episode')
plt.ylabel(f'Average score (window size={scores_window.maxlen})')
save_fig(f'scores-avg-history_{architecture}_{timestamp}')
plt.show()

# Average scores history
plt.Figure(figsize=(6, 6))
plt.title(f'Average score ({architecture})')
plt.plot(np.cumsum(scores_history) / (np.arange(1, len(scores_history) + 1)))
plt.xlabel('Episode')
plt.ylabel('Average score')
save_fig(f'scores-avg-history_{architecture}_{timestamp}')
plt.show()

#%% Print scores and mean scores

logs_df = pd.DataFrame({'scores': scores_history,
                        'window': scores_avg_history})
logs_df.to_csv(f'./logs/scores_log_{architecture}_{timestamp}.csv', sep='\t', header=True, index=False)

#%% Letting the agent play for a certain number of steps to try the new protocol

scores_test = play_one_episode(env, agent, max_t)
