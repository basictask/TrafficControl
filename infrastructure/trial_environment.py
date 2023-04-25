"""
This simple script is used to arbitrarily input actions to an environment
"""
# Imports
from trial_functions import test_a_r_env
from environment import Environment
from agents.agent_gnn2 import Agent
import configparser
import os
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))

env = Environment()
agent = Agent(env.state_shape, env.action_shape, env.state_high)

test_a_r_env(env, 2, 9, 'add_lane')
test_a_r_env(env, 9, 4, 'add_roundabout')
test_a_r_env(env, 4, 1, 'remove_lane')
test_a_r_env(env, 1, 8, 'add_lane')
test_a_r_env(env, 8, 3, 'add_roundabout')
test_a_r_env(env, 3, 7, 'add_lane')
