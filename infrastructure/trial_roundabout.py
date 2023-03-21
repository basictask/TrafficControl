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
from trial_functions import *
from suppl import *

vrate = 20
n_steps = 0
show_win = True
paths_to_gen = 100
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_5_intersection.html'
entry_points = list('ACDEFHJI')
radius = 10

center_node = l2n('B')

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

test_a_r_roads(r, 'add', 'lane', 'A', 'B')
test_a_r_junct(r, 'roundabout', 'B')
test_a_r_junct(r, 'roundabout', 'G')
test_a_r_roads(r, 'add', 'lane', 'D', 'G')
test_a_r_roads(r, 'add', 'road', 'E', 'D')
test_a_r_roads(r, 'add', 'road', 'I', 'D')
test_a_r_junct(r, 'trafficlight', 'G')

roads, vehicle_mtx, signals = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals)
