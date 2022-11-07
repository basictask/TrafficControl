"""
This is a small script to test the inner workings of the reader class used in the project.
The program reads a construction protocol, adds in predefined entry points and displays a pygame window with the constructed city.
There are predefined cities and entry points added as comments.

@author: Daniel Kuknyo
"""

# Imports
import os
os.chdir('/home/daniel/Documents/ELTE/trafficControl')

from trafficSimulator import *
from city_constructor import *

test_add = False # Modifying this to True will result in testing the add/remove functions of the reader class

#%% Set up the reader from a .html GeoGebra construction protocol

# filepath = 'cities/Simple.html'
# filepath = 'cities/starcity.html'
filepath = 'cities/bakats.html'

# Points that are valid for entering the traffic system
# entry_points = ['A','C','G','J'] # Simple
# entry_points = ['A','D','F','H','J'] # Star city
entry_points = ['A', 'M', 'E', 'K', 'J', 'I', 'B', 'F', 'C', 'D', 'T'] # Bakats area 

vrate = 60 # Rate of vehicles coming in from each entry point
paths_to_gen = 10 # How many paths to generate
path_dist = 'normal' # One of 'normal', 'uniform' 
steps_per_update = 5 # How many steps the game takes in the interval of one frame update

r = reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

if(not test_add):
    roads, vehicle_mtx = r.get_matrices()
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)

#%% Tessting add function

if(test_add):
    r.add_segment('E', 'G')
    r.add_segment('D', 'H')
    r.add_segment('H', 'D')
    r.add_segment('G', 'H')
    r.add_segment('H', 'G')
    r.add_segment('F', 'H')
    r.add_segment('H', 'F')

    # roads, vehicle_mtx = r.get_matrices()
    # start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)

    #%% Testing remove function

    r.remove_segment('B', 'H')
    r.remove_segment('H', 'B')
    r.remove_segment('F', 'B')

    roads, vehicle_mtx = r.get_matrices()

    #%%
    start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)

