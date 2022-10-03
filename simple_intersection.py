# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:28:02 2022

@author: daniel_kuknyo
"""

# Imports
import os
os.chdir('C:\\Users\\daniel_kuknyo\\Documents\\ELTE\\trafficControl')

from trafficSimulator import *


#%%
filepath = 'cities/simple_intersection.html'

entry_points = ['A','C','E','D'] 
vrate = 60
paths_to_gen = 8
path_dist = 'uniform'                   
steps_per_update = 5

r = reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
roads, vehicle_mtx = r.get_matrices()

start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)