# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:18:49 2022

This is a road simulator application that uses reinforcement learning to adjust 
road capacities and traffic light cycles in order to optimize traffic flow.
The goal for the agent is to find out which configuration of roads lead to the 
most throughput of cars and the least traffic jams. 

The project uses the TrafficSimulator application of BilHim for the base traffic control. 

@author: Daniel Kuknyo
"""

from trafficSimulator import *
import sys
sys.path.append('C:/Users/Daniel Kuknyo/Downloads/TrafficControlRL/trafficSimulator-src/')

sim = Simulation()

# Driver parameters
n = 15  # Number of drivers
a = 2  # Maximum acceleration
b = 20  # Comfortable deceleration
c = 5 
r = 10
l = 300
n_steps = 0
steps_per_update = 5

# Add roads to the simulation
sim.create_roads([
    ((300, 98), (0, 98)),
    ((0, 102), (300, 102)),
    ((180, 60), (0, 60)),
    ((220, 55), (180, 60)),
    ((300, 30), (220, 55)),
    ((180, 60), (160, 98)),
    ((158, 130), (300, 130)),
    ((0, 178), (300, 178)),
    ((300, 182), (0, 182)),
    ((160, 102), (155, 180))
])

# Add vehicle generator
sim.create_gen({
    'vehicle_rate': 60,
    'vehicles': [
        [1, {"path": [4, 3, 2]}],
        [1, {"path": [0]}],
        [1, {"path": [1]}],
        [1, {"path": [6]}],
        [1, {"path": [7]}]
    ]
})

# Start simulation
win = Window(sim, steps_per_update, n_steps)
win.offset = (-150, -110)
win.run()
