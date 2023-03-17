"""
This is a demonstration of a very simple intersection for testing purposes
"""

from trafficSimulator import *
from city_constructor import Reader
from suppl import *

vrate = 60
n_steps = 0
show_win = True
paths_to_gen = 6
steps_per_update = 5
path_dist = 'normal'
filepath = '../cities/simple_intersection.html'
entry_points = list('ACDFGJIL')

# Construct the reader
r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

# Do some operations
r.remove_lane(l2n('B'), l2n('C'))
r.remove_lane(l2n('K'), l2n('B'))
r.remove_lane(l2n('J'), l2n('K'))

r.remove_lane(l2n('D'), l2n('E'))
r.remove_lane(l2n('E'), l2n('H'))
r.remove_lane(l2n('H'), l2n('I'))

r.remove_lane(l2n('A'), l2n('B'))
r.remove_lane(l2n('B'), l2n('E'))
r.remove_lane(l2n('E'), l2n('F'))

r.remove_lane(l2n('G'), l2n('H'))
r.remove_lane(l2n('H'), l2n('K'))
r.remove_lane(l2n('K'), l2n('L'))

# Create the matrices and start the simulation
roads, vehicle_mtx = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win)

