# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:28:02 2022

@author: daniel_kuknyo

Simple intersection: type 1 junction
"""

# Imports
import os
os.chdir('C:\\Users\\daniel_kuknyo\\Documents\\ELTE\\trafficControl')

from trafficSimulator import *
from city_constructor import *

#%%
filepath = 'cities/simple_2lane_intersection.html'
entry_points = ['A','F','I','L','C','J','D','K'] 
vrate = 60
paths_to_gen = 6
path_dist = 'uniform'                   
steps_per_update = 5

r = reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

r.remove_segment('B','A')
r.remove_segment('F','E')
r.remove_segment('I','H')
r.remove_segment('G','L')
r.remove_segment('C','B')
r.remove_segment('H','J')
r.remove_segment('E','D')
r.remove_segment('K','G')

roads, vehicle_mtx = r.get_matrices()

#%%
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)