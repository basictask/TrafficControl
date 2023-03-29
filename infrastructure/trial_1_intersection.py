"""
This is a test file to design a simple 4-way intersection
        A
        |
E ----- B ----- C
        |
        D
"""
from trial_functions import *
from suppl import *

# Params
vrate = 10
n_steps = 0
show_win = True
paths_to_gen = 100
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_1_intersection.html'
entry_points = list('ACDE')

# Construct the reader
r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

# Create infrastructure
# test_a_r_roads(r, 'add', 'road', 'A', 'C')
test_a_r_roads(r, 'add', 'road', 'A', 'B')
test_a_r_roads(r, 'add', 'road', 'C', 'B')
test_a_r_roads(r, 'add', 'road', 'D', 'B')
test_a_r_roads(r, 'add', 'road', 'E', 'B')
test_a_r_junct(r, 'trafficlight', 'B')

roads, vehicle_mtx, signals = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals)
