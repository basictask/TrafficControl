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

r.add_roundabout(4)
r.add_roundabout(8)
r.add_roundabout(3)
r.add_lane(7, 3)
r.add_lane(7, 8)
r.add_lane(7, 8)
r.add_roundabout(6)
r.add_lane(7, 6)

roads, vehicle_mtx, signals = r.get_matrices()
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals)
