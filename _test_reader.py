# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:35:03 2022

This is a demonstration of how a city can be set up with a geogebra construction protocol

@author: daniel_kuknyo
"""

# Imports
import os
os.chdir('C:\\Users\\daniel_kuknyo\\Documents\\ELTE\\trafficControl')

from trafficSimulator import *
from city_constructor import reader

#%% Set up and start the simulation

def start_sim(roads, vehicle_mtx, offset, steps_per_update):
    sim = Simulation()
    sim.create_roads(roads)
    sim.create_gen(vehicle_mtx)
    
    win = Window(sim)
    win.offset = offset # (x, y) tuple
    win.run(steps_per_update = steps_per_update)
    
#%% Set up the reader from a .html GeoGebra construction protocol

# filepath = 'cities/simple.html'
# filepath = 'cities/starcity.html'
filepath = 'cities/bakats.html'

# Points that are valid for entering the traffic system
# entry_points = ['A','C','E','G'] # Simple
# entry_points = ['A','D','F','H','J'] # Star city
entry_points = ['P','O','Q','M','K','J','T','S','R','G','E','F'] # Bakats area 

# Rate of vehicles coming in from each entry point
vrate = 60

# How many paths to generate
paths_to_gen = 8

# One of 'normal', 'uniform' 
path_dist = 'normal'                   

# How many steps the game takes in the interval of one frame update
steps_per_update = 5

r = reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
roads, vehicle_mtx = r.get_matrices()

#%% Tessting add function

r.add_segment('E', 'G')
r.add_segment('D', 'H')

roads, vehicle_mtx = r.get_matrices()

#start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)

#%% Testing remove function

r.remove_segment('E', 'G')

roads, vehicle_mtx = r.get_matrices()

start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)
