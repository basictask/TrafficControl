# Imports
from suppl import ACTION_NAMES
from trial_functions import play_one_episode
from environment import Environment
from agents.agent_gnn2 import Agent
import matplotlib.pyplot as plt
from collections import deque
import configparser
import numpy as np
import os
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))

env = Environment()
agent = Agent(env.state_shape, env.action_shape, env.state_high)

next_state, reward, successful = env.step(4, 8, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(8, 0, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(0, 2, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(2, 5, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(5, 9, ACTION_NAMES['add_roundabout'])
next_state, reward, successful = env.step(9, 7, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(7, 3, ACTION_NAMES['add_lane'])
next_state, reward, successful = env.step(3, 1, ACTION_NAMES['remove_lane'])
next_state, reward, successful = env.step(1, 4, ACTION_NAMES['remove_lane'])
