from .vehicle import Vehicle
from numpy.random import randint


class VehicleGenerator:
    def __init__(self, sim, config=None):
        """
        Vehicle generators are attached to the nodes of the graph. They spawn vehicles onto roads
        :param sim: Simulation object
        :param config: Optional configurations
        """
        if config is None:
            config = {}

        # Set default configurations
        self.sim = sim
        self.vehicle_rate = 20
        self.vehicles = [(1, {})]
        self.last_added_time = 0

        # Update configurations
        for attr, val in config.items():
            setattr(self, attr, val)

        # Calculate properties
        self.upcoming_vehicle = self.generate_vehicle()

    def generate_vehicle(self):
        """
        Returns a random vehicle from self.vehicles with random proportions
        """
        total = sum(pair[0] for pair in self.vehicles)
        r = randint(1, total+1)
        for (weight, config) in self.vehicles:
            r -= weight
            if r <= 0:
                return Vehicle(config)

    def update(self) -> bool:
        """
        If time elasped after last added vehicle is greater than vehicle_period, generate a vehicle
        :return: True: spawn happened with new vehicle, False: spawn didn't happen
        """
        if self.sim.t - self.last_added_time >= 60 / self.vehicle_rate:
            road = self.sim.roads[self.upcoming_vehicle.path[0]]      
            if len(road.vehicles) == 0 or road.vehicles[-1].x > self.upcoming_vehicle.s0 + self.upcoming_vehicle.l:
                self.upcoming_vehicle.time_added = self.sim.t  # If there is space for the generated vehicle; add it
                road.vehicles.append(self.upcoming_vehicle)
                self.last_added_time = self.sim.t  # Reset last_added_time and upcoming_vehicle
            self.upcoming_vehicle = self.generate_vehicle()
            return True
        return False
