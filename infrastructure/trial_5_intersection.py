"""
This is a test file to design intersections. The layout of the nodes is as shown below:
     A   C            H
     \\ /             |
F ---- B ------------ G ---- J
      / \\            |
     E   D            I
"""
from trial_functions import *
from suppl import *

# Params
vrate = 40
n_steps = 0
show_win = True
paths_to_gen = 100
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_5_intersection.html'
entry_points = list('ACDEFHJI')

center_node = l2n('B')

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)

test_a_r_roads(r, 'add', 'lane', 'A', 'B')
test_a_r_junct(r, 'roundabout', 'B')
test_a_r_junct(r, 'roundabout', 'G')
test_a_r_roads(r, 'add', 'lane', 'D', 'G')
test_a_r_roads(r, 'add', 'road', 'E', 'D')
test_a_r_roads(r, 'add', 'road', 'I', 'D')
test_a_r_junct(r, 'trafficlight', 'D')
test_a_r_roads(r, 'add', 'road', 'G', 'D')
test_a_r_roads(r, 'add', 'road', 'H', 'C')
test_a_r_roads(r, 'add', 'road', 'C', 'G')
test_a_r_roads(r, 'add', 'lane', 'B', 'G')
test_a_r_roads(r, 'add', 'lane', 'G', 'B')
test_a_r_junct(r, 'trafficlight', 'C')
test_a_r_roads(r, 'add', 'road', 'A', 'C')
test_a_r_roads(r, 'add', 'road', 'A', 'F')
test_a_r_roads(r, 'add', 'road', 'E', 'F')
test_a_r_roads(r, 'add', 'road', 'E', 'D')
test_a_r_junct(r, 'roundabout', 'A')
test_a_r_junct(r, 'roundabout', 'F')
test_a_r_junct(r, 'roundabout', 'E')
test_a_r_roads(r, 'add', 'lane', 'H', 'J')
test_a_r_roads(r, 'add', 'lane', 'J', 'I')
test_a_r_junct(r, 'trafficlight', 'H')
test_a_r_junct(r, 'trafficlight', 'J')
test_a_r_junct(r, 'trafficlight', 'I')
test_a_r_roads(r, 'add', 'lane', 'B', 'G')
test_a_r_roads(r, 'add', 'lane', 'G', 'B')
test_a_r_roads(r, 'add', 'lane', 'H', 'H')

for x in entry_points:
    test_a_r_junct(r, 'roundabout', x)

for x in entry_points:
    test_a_r_junct(r, 'righthand', x)

roads, vehicle_mtx, signals = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals)
