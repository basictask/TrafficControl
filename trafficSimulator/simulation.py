from .road import Road
from copy import deepcopy
from .traffic_signal import TrafficSignal
from .vehicle_generator import VehicleGenerator


class Simulation:
    def __init__(self, config=None):
        if config is None:
            config = {}

        # Inner params
        self.t = None
        self.dt = None
        self.roads = None
        self.generators = None
        self.frame_count = None
        self.traffic_signals = None

        # Set default configuration
        self.t = 0.0  # Time keeping
        self.dt = 1 / 60  # Simulation time step
        self.frame_count = 0  # Frame count keeping
        self.total_vehicles_distance = 0

        # Arrays to store roads, generators and traffic signals
        self.roads = []
        self.generators = []
        self.traffic_signals = []

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def create_road(self, start, end):
        road = Road(start, end)
        self.roads.append(road)
        return road

    def create_roads(self, road_list):
        for road in road_list:
            self.create_road(*road)

    def create_gen(self, config=None):
        if config is None:
            config = {}
        gen = VehicleGenerator(self, config)
        self.generators.append(gen)
        return gen

    def create_signal(self, roads, config=None):
        if config is None:
            config = {}
        roads = [[self.roads[i] for i in road_group] for road_group in roads]
        sig = TrafficSignal(roads, config)
        self.traffic_signals.append(sig)
        return sig

    def update(self):
        # Update every road
        for road in self.roads:
            road.update(self.dt)

        # Add vehicles
        for gen in self.generators:
            gen.update()

        for signal in self.traffic_signals:
            signal.update(self)

        # Check roads for out of bounds vehicle
        for road in self.roads:
            # If road has no vehicles, continue
            if len(road.vehicles) == 0:
                continue
            # If road has vehicles
            vehicle = road.vehicles[0]
            # If first vehicle is out of road bounds
            if vehicle.x >= road.length:
                # If vehicle has a next road
                if vehicle.current_road_index + 1 < len(vehicle.path):
                    # Update current road to next road
                    vehicle.current_road_index += 1
                    # Create a copy and reset some vehicle properties
                    new_vehicle = deepcopy(vehicle)
                    new_vehicle.x = 0
                    # Add it to the next road
                    next_road_index = vehicle.path[vehicle.current_road_index]
                    self.roads[next_road_index].vehicles.append(new_vehicle)
                # In all cases, remove it from its road
                road.vehicles.popleft()
        # Increment time
        self.t += self.dt
        self.frame_count += 1

    def run(self, steps):
        for _ in range(steps):
            self.update()
