from trafficSimulator import *

# Create simulation
sim = Simulation()

# Add multiple roads
sim.create_roads([
    ((300, 98), (0, 98)),       # 0
    ((0, 102), (300, 102)),     # 1
    ((180, 60), (0, 60)),       # 2
    ((220, 55), (180, 60)),     # 3
    ((300, 30), (220, 55)),     # 4
    ((180, 60), (160, 98)),     # 5
    ((158, 130), (300, 130)),   # 6
    ((0, 178), (300, 178)),     # 7
    ((300, 182), (0, 182)),     # 8
    ((160, 102), (155, 180))    # 9
    
])

sim.create_gen({
    'vehicle_rate': 60,
    'vehicles': [
        [1, {"path": [4, 3, 2]}],
        [0.1, {"path": [0]}],
        [0.1, {"path": [1]}],
        [0.1, {"path": [6]}],
        [0.1, {"path": [7]}]
    ]
})

# Start simulation
win = Window(sim)
win.offset = (-150, -110)
win.run(steps_per_update=5)