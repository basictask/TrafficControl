"""
This file is used to execute many actions on the reader in a short time interval in order to test its robustness.
"""
# Imports
from trial_functions import *
from suppl import *
import numpy as np

# Params
vrate = 20
n_steps = 10000
show_win = False
offset = (-100, -100)
paths_to_gen = 100
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_5_intersection.html'
entry_points = list('ACDEFHJI')

# Constants
r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
n_nodes = r.matrix.shape[0]
nodes = np.arange(n_nodes)
n_actions = len(ACTIONS)
actions = np.arange(n_actions)

n_episodes = 500
n_builds = 30  # How many builds to execute at once

start, end, action = -1, -1, -1
roads, vehicle_mtx, signals = -1, -1, -1

for i in range(n_episodes):
    r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
    print(f'episode: {i}')

    for j in range(n_builds):
        start = np.random.choice(nodes)
        end = np.random.choice(nodes)
        action = np.random.choice(actions)
        action_name = ACTIONS[action]
        junction_start = JUNCTION_TYPES[r.matrix.loc[start, start]]
        junction_end = JUNCTION_TYPES[r.matrix.loc[end, end]]

        print(f'iteration: {j}, start: {start}, end: {end}, action: {action_name}, start: {junction_start}, end: {junction_end}')  # Show logs

        if action_name == 'add_lane':
            successful = r.add_lane(start, end)
        elif action_name == 'remove_lane':
            successful = r.remove_lane(start, end)
        elif action_name == 'add_road':
            successful = r.add_road(start, end)
        elif action_name == 'remove_road':
            successful = r.remove_road(start, end)
        elif action_name == 'add_righthand':
            successful = r.add_righthand(end)
        elif action_name == 'add_roundabout':
            successful = r.add_roundabout(end)
        elif action_name == 'add_trafficlight':
            successful = r.add_trafficlight(end)
        else:
            raise IllegalActionError(f'Undefined action: {action}')

        roads, vehicle_mtx, signals = r.get_matrices()
        start_sim(roads, vehicle_mtx, offset, steps_per_update, n_steps, show_win, signals)
