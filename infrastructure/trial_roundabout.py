"""
This is a test file to design a roundabout
Simple intersection design
        A
        |
E ----- B ----- C
        |
        D
"""

from city_constructor import Reader
from suppl import *

vrate = 20
n_steps = 0
show_win = True
paths_to_gen = 4
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_1_intersection.html'
entry_points = list('ACDE')
radius = 10

center_node = l2n('B')

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

r.add_roundabout(center_node)
# r.remove_roundabout(center_node)
# r.add_trafficlight(center_node)

roads, vehicle_mtx, signals = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals)
