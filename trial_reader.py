"""
This is a small script to test the inner workings of the reader class used in the project.
The program reads a construction protocol, adds in predefined entry points and displays a pygame window with the constructed city.
There are predefined cities and entry points added as comments.
@author: Daniel Kuknyo
"""

# Imports
from trial_functions import *
import os
os.chdir('/home/daniel/Documents/ELTE/trafficControl')


# %% Set up the reader from a .html GeoGebra construction protocol

# filepath = 'cities/simple.html'
# filepath = 'cities/starcity.html'
filepath = 'cities/bakats.html'

# Points that are valid for entering the traffic system
# entry_points = ['A','C','G','J'] # Simple
# entry_points = ['A','D','F','H','J'] # Star city
entry_points = ['A', 'M', 'E', 'K', 'J', 'I', 'B', 'F', 'C', 'D', 'T']  # Bakats area

vrate = 60  # Rate of vehicles coming in from each entry point
max_lanes = 3  # How many lanes are allowd going from A --> B (1-directional definition)
offset = (-500, -500)  # The simulation window offset
n_steps = 0  # How many steps to simulate (in case there's no Sim window)
show_win = True  # True if the Simulation window shall be displayed
test_add = True  # Modifying this to True will result in testing the add/remove functions of the reader class
paths_to_gen = 10  # How many paths to generate
path_dist = 'normal'  # One of 'normal', 'uniform'
steps_per_update = 5  # How many steps the game takes in the interval of one frame update

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist, max_lanes)  # Construct the Reader object to generate the vehicle matrices

if not test_add:
    roads, vehicle_mtx = r.get_matrices()
    start_sim(roads, vehicle_mtx, offset, steps_per_update, n_steps, show_win)

# %% Testing add function
"""
Note: This time we are using the letter_to_number function to define a node in the graph. 
This is a necessary step for visualization as the RL environment will refer to it in numeric form
"""


if test_add:
    # Testing the add_lane method
    test_a_r_roads(r, 'add', 'lane', 'E', 'Q')
    test_a_r_roads(r, 'add', 'lane', 'E', 'L')
    test_a_r_roads(r, 'add', 'lane', 'M', 'T')
    test_a_r_roads(r, 'add', 'lane', 'T', 'M')
    print('Adding done...\n')

    # Testing the remove_lane method
    test_a_r_roads(r, 'remove', 'lane', 'G', 'F')
    test_a_r_roads(r, 'remove', 'lane', 'F', 'G')
    test_a_r_roads(r, 'remove', 'lane', 'G', 'D')
    test_a_r_roads(r, 'remove', 'lane', 'D', 'G')
    test_a_r_roads(r, 'remove', 'lane', 'D', 'C')
    test_a_r_roads(r, 'remove', 'lane', 'C', 'F')
    print('Removing done...\n')

    # Testing the add_road method
    test_a_r_roads(r, 'add', 'road', 'I', 'F')
    test_a_r_roads(r, 'add', 'road', 'B', 'G')
    print('Adding on starting graph done...\n')

    # Testing the remove_road method
    test_a_r_roads(r, 'remove', 'road', 'I', 'J')
    test_a_r_roads(r, 'remove', 'road', 'B', 'I')
    print('Removing on starting graph done...\n')

    # Testing the add_lane on segments that currently have lanes
    test_a_r_roads(r, 'add', 'lane', 'E', 'M')
    test_a_r_roads(r, 'add', 'lane', 'E', 'M')
    test_a_r_roads(r, 'add', 'lane', 'E', 'L')
    test_a_r_roads(r, 'add', 'lane', 'M', 'T')
    print('Adding multiple lanes done...\n')

    # Testing the remove_lane on segments that already have lanes
    test_a_r_roads(r, 'remove', 'lane', 'E', 'L')
    test_a_r_roads(r, 'add', 'lane', 'E', 'L')
    test_a_r_roads(r, 'remove', 'lane', 'E', 'L')
    test_a_r_roads(r, 'remove', 'lane', 'E', 'L')
    test_a_r_roads(r, 'remove', 'lane', 'E', 'L')
    print('Removing even more lanes done...\n')

    # Testing junction adding/removing
    test_a_r_junct(r, 'trafficlight', 'Q')
    test_a_r_junct(r, 'trafficlight', 'T')
    print('Adding traffic lights done...\n')

    test_a_r_junct(r, 'roundabout', 'M')

    # Generate all the matrices and go
    roads, vehicle_mtx, signals = r.get_matrices()

    # Display the final representation matrix
    print(pretty_matrix(r.matrix))
    print(f'Signals: {signals}')

    start_sim(roads, vehicle_mtx, offset, steps_per_update, n_steps, show_win, signals)
    print('Done.')
