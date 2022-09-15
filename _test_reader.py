# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:35:03 2022

This is a demonstration of how a city can be set up with a geogebra construction protocol

@author: daniel_kuknyo
"""
# Imports
from trafficSimulator import *
from city_constructor import reader

#%% Set up the reader

filepath        = 'cities/simple_color.html'    # A .html GeoGebra construction protocol
entry_points    = ['A','C','E','G']             # Points that are valid for entering the traffic system
vrate           = 60                            # Rate of vehicles coming in from each entry point
paths_to_gen    = 6                             # How many paths to generate
path_dist    = 'normal'                         # One of 'normal', 'uniform' 

r = reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
roads, vehicle_mtx = r.get_matrices()

#%% Set up the simulation

sim = Simulation()
sim.create_roads(roads)
sim.create_gen(vehicle_mtx)

#%% Start simulation

win = Window(sim)
win.offset = (-150, -110)
win.run(steps_per_update=5)
