"""
This is a small script to test the inner workings of the reader class used in the project.
The program reads a construction protocol, adds in predefined entry points and displays a pygame window with the constructed city.
There are predefined cities and entry points added as comments.

@author: Daniel Kuknyo
"""

# Imports
from city_constructor import *
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
n_steps = 0  # How many steps to simulate (in case there's no Sim window)
show_win = True  # True if the Simulation window shall be displayed
test_add = True  # Modifying this to True will result in testing the add/remove functions of the reader class
paths_to_gen = 10  # How many paths to generate
path_dist = 'normal'  # One of 'normal', 'uniform'
steps_per_update = 5  # How many steps the game takes in the interval of one frame update

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist, max_lanes)  # Construct the Reader object to generate the vehicle matrices

if not test_add:
    roads, vehicle_mtx = r.get_matrices()
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win)

# %% Tessting add function
"""
Note: This time we are using the letter_to_number function to define a node in the graph. 
This is a necessary step for visualization as the RL environment will refer to it in numeric form
"""

if test_add:
    # Testing the add_lane method
    r.add_lane(l2n('E'), l2n('Q'))
    r.add_lane(l2n('E'), l2n('L'))
    r.add_lane(l2n('M'), l2n('T'))
    r.add_lane(l2n('T'), l2n('M'))
    print('Adding done...')

    # Testing the remove_lane method
    r.remove_lane(l2n('G'), l2n('F'))
    r.remove_lane(l2n('F'), l2n('G'))
    r.remove_lane(l2n('G'), l2n('D'))
    r.remove_lane(l2n('D'), l2n('G'))
    r.remove_lane(l2n('D'), l2n('C'))
    r.remove_lane(l2n('C'), l2n('F'))
    print('Removing done...')

    # Testing the add_road method
    r.add_road(l2n('I'), l2n('F'))
    r.add_road(l2n('B'), l2n('G'))
    print('Adding on starting graph done...')

    # Testing the remove_road method
    r.remove_road(l2n('I'), l2n('J'))
    r.remove_road(l2n('B'), l2n('I'))
    print('Removing on starting graph done...')

    # Testing the add_lane on segments that currently have lanes
    r.add_lane(l2n('E'), l2n('M'))  # (E, M) already should have a lane at this point
    r.add_lane(l2n('E'), l2n('M'))
    r.add_lane(l2n('E'), l2n('L'))
    r.add_lane(l2n('M'), l2n('T'))
    print('Adding multiple lanes done...')

    # Testing the remove_lane on segments that already have lanes
    r.remove_lane(l2n('E'), l2n('L'))

    # try:
    #     r.remove_lane(l2n('E'), l2n('L'))
    # except SegmentRemovalError:
    #     print("Segment removal overflow (correct behavior)")

    roads, vehicle_mtx = r.get_matrices()
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win)
    print('Done.')
