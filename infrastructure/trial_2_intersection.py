"""
This is a test file to design traffic lights to simple right-hand intersections for a configuration of 2 intersections at the same time
        C       F
        |       |
A ----- B ----- D ----- G
        |       |
        E       H

Example list that gets created here: [[[0, 9], [10, 8]], [[12, 2], [13, 11]]]
Which is a list of junctions: [junction1, junction2, ...]
Each junction is a pair of 2-lists where each element is a road: [[road1, road3], [road2, road4]]
Where each pair is an opposite set of roads that sync in phase with each other. The other pair is the other set of roads that sync in the opposite phase
"""
from city_constructor import Reader
from suppl import *
import pandas as pd

vrate = 20
n_steps = 0
show_win = True
paths_to_gen = 6
steps_per_update = 5
path_dist = 'uniform'
filepath = '../cities/test_2_intersection.html'
entry_points = list('ACEFGH')

r = Reader(filepath, entry_points, vrate, paths_to_gen, path_dist)
r.remove_road(l2n('E'), l2n('B'))
# r.add_lane(l2n('E'), l2n('B'))
r.add_lane(l2n('H'), l2n('D'))
r.add_lane(l2n('F'), l2n('D'))
r.add_lane(l2n('F'), l2n('D'))

# Create the matrices and start the simulation
roads, vehicle_mtx, _ = r.get_matrices()

nodes = [(100.82, 94.34), (188.22, 95.34)]  # B, C

print(count_incoming_lanes(r.matrix, l2n('B')))
print(count_incoming_lanes(r.matrix, l2n('B')))
print(count_incoming_lanes(r.matrix, l2n('D')))
print(count_incoming_lanes(r.matrix, l2n('D')))

signals_to_create = []
for trafficlight_node in nodes:
    signals_ind = [i for i, (start, end) in enumerate(roads) if end == trafficlight_node]
    signal_roads = [(start, end) for (start, end) in roads if end == trafficlight_node]
    angles = [find_angle(start, end, absolute=False) for start, end in signal_roads]
    ds = pd.Series(angles, index=signals_ind).sort_values(ascending=True)
    junction_signals = [[], []]
    i = 0
    while i < len(ds):
        for k in range(2):
            signal = []
            j = 0
            while i + j < len(ds) and ds.iloc[i + j] == ds.iloc[i]:
                signal.append(ds.index[i + j])
                j += 1
            i += j
            junction_signals[k].extend(signal)
    signals_to_create.append(junction_signals)

print(signals_to_create)
# signals_to_create = [[[0, 8], [7]], [[10, 2], [11, 9]]]
start_sim(roads, vehicle_mtx, (-150, -110), steps_per_update, n_steps, show_win, signals_to_create)
