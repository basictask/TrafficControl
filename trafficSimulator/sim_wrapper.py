def start_sim(roads, vehicle_mtx, offset, steps_per_update):
    from trafficSimulator.simulation import Simulation
    from trafficSimulator.window import Window
    sim = Simulation()
    sim.create_roads(roads)
    sim.create_gen(vehicle_mtx)
    win = Window(sim)
    win.offset = offset # (x, y) tuple
    win.run(steps_per_update = steps_per_update)
