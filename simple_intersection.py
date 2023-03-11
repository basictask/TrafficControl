"""
Simple intersection: type 1 junction
"""
# Imports
from city_constructor import *

import os
os.chdir('/home/daniel/Documents/ELTE/trafficControl')

filepath = 'cities/simple_2lane_intersection.html'
entry_points = ['A', 'F', 'I', 'L', 'C', 'J', 'D', 'K']
vrate = 60
paths_to_gen = 6
path_dist = 'uniform'                   
steps_per_update = 5

# Construct the reader
r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

# Do some operations
r.remove_lane(l2n('B'), l2n('A'))
r.remove_lane(l2n('F'), l2n('E'))
r.remove_lane(l2n('G'), l2n('L'))
r.remove_lane(l2n('C'), l2n('B'))
r.remove_lane(l2n('H'), l2n('J'))
r.remove_lane(l2n('E'), l2n('D'))
r.remove_lane(l2n('K'), l2n('G'))

# Create the matrices and start the simulation
roads, vehicle_mtx = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update)