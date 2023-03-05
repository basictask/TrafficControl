"""
This is a small script to test the inner workings of the reader class used in the project.
The program reads a construction protocol, adds in predefined entry points and displays a pygame window with the constructed city.
There are predefined cities and entry points added as comments.

@author: Daniel Kuknyo
"""

# Imports
from trafficSimulator import *
from city_constructor import *
from suppl import *

import os
os.chdir('/home/daniel/Documents/ELTE/trafficControl')
TEST_ADD = True  # Modifying this to True will result in testing the add/remove functions of the reader class

# %% Set up the reader from a .html GeoGebra construction protocol

# filepath = 'cities/simple.html'
# filepath = 'cities/starcity.html'
filepath = 'cities/bakats.html'

# Points that are valid for entering the traffic system
# entry_points = ['A','C','G','J'] # Simple
# entry_points = ['A','D','F','H','J'] # Star city
entry_points = ['A', 'M', 'E', 'K', 'J', 'I', 'B', 'F', 'C', 'D', 'T']  # Bakats area

vrate = 60  # Rate of vehicles coming in from each entry point
paths_to_gen = 10  # How many paths to generate
path_dist = 'normal'  # One of 'normal', 'uniform'
steps_per_update = 5  # How many steps the game takes in the interval of one frame update

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)  # Construct the Reader object to generate the vehicle matrices

if not TEST_ADD:
    roads, vehicle_mtx = r.get_matrices()
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)

# %% Tessting add function
"""
Note: This time we are using the letter_to_number function to define a node in the graph. 
This is a necessary step for visualization as the RL environment will refer to it in numeric form
"""

if TEST_ADD:
    # In this case we are testing the add and remove functions
    r.add_segment(letter_to_number('E'), letter_to_number('Q'))
    r.add_segment(letter_to_number('E'), letter_to_number('L'))
    r.add_segment(letter_to_number('M'), letter_to_number('T'))
    r.add_segment(letter_to_number('T'), letter_to_number('M'))

    r.remove_segment(letter_to_number('G'), letter_to_number('F'))
    r.remove_segment(letter_to_number('F'), letter_to_number('G'))
    r.remove_segment(letter_to_number('G'), letter_to_number('D'))
    r.remove_segment(letter_to_number('D'), letter_to_number('G'))
    r.remove_segment(letter_to_number('D'), letter_to_number('C'))
    r.remove_segment(letter_to_number('C'), letter_to_number('F'))

    r.add_road(letter_to_number('I'), letter_to_number('F'))
    r.add_road(letter_to_number('B'), letter_to_number('G'))

    r.remove_road(letter_to_number('I'), letter_to_number('J'))
    r.remove_road(letter_to_number('B'), letter_to_number('I'))

    roads, vehicle_mtx = r.get_matrices()
    # print(roads, '\n\n', vehicle_mtx)  # Debugging
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)
