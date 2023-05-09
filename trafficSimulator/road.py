from scipy.spatial import distance
from collections import deque
import os
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.ini'))


class Road:
    def __init__(self, start, end):
        """
        Sets up a Road object in the simulation
        :param start: Starting coordinate (x, y) of the road
        :param end: Ending coordinate (x, y) of the road
        """
        self.slow_factor = args['road'].getfloat('slow_factor')
        self.slow_distance = args['road'].getint('slow_distance')

        # Set default configuration
        self.end = end
        self.start = start
        self.vehicles = deque()
        self.traffic_signal = None
        self.has_traffic_signal = False
        self.traffic_signal_group = None
        self.length = distance.euclidean(self.start, self.end)
        self.angle_sin = (self.end[1] - self.start[1]) / self.length
        self.angle_cos = (self.end[0] - self.start[0]) / self.length

    def set_traffic_signal(self, signal, group):
        """
        Attaches a traffic signal to a given junction
        :param signal: Signal object with the correct parameters
        :param group: Road group that the signal will control
        :return: None
        """
        self.traffic_signal = signal
        self.traffic_signal_group = group
        self.has_traffic_signal = True

    @property
    def traffic_signal_state(self):
        """
        Gets the traffic signal state: True: green, False: red
        :return: None
        """
        if self.has_traffic_signal:
            i = self.traffic_signal_group
            return self.traffic_signal.current_cycle[i]
        return True

    def update(self, dt):
        """
        Updates the vehicle positions on the roads
        :param dt: Time elapsed
        :return: None
        """
        n = len(self.vehicles)
        vehicles_distance = 0
        if n > 0:
            vehicles_distance += self.vehicles[0].update(None, dt)  # Update first vehicle

            for i in range(1, n):  # Update other vehicles
                lead = self.vehicles[i-1]
                vehicles_distance += self.vehicles[i].update(lead, dt)

            # Check for traffic signal
            if self.traffic_signal_state:  # If traffic signal is green or doesn't exist
                if self.has_traffic_signal:
                    self.vehicles[0].unstop()
                    for vehicle in self.vehicles:
                        vehicle.unslow()
                else:
                    if self.vehicles[0].x >= self.length - self.slow_distance and self.vehicles[0].v_max >= self.vehicles[0].v_max * self.slow_factor:
                        self.vehicles[0].slow(self.slow_factor * self.vehicles[0].get__v_max)
                    for vehicle in self.vehicles:
                        if vehicle.x < self.slow_distance:
                            vehicle.unstop()
                            vehicle.unslow()

            else:  # If traffic signal is red
                if self.vehicles[0].x >= self.length - self.traffic_signal.slow_distance:
                    self.vehicles[0].slow(self.traffic_signal.slow_factor * self.vehicles[0].get__v_max)  # Slow vehicles in slowing zone

                # Check if the vehicle is in the stop zone
                vehicle_stop_close = self.vehicles[0].x >= self.length - self.traffic_signal.stop_distance
                vehicle_stop_far = self.vehicles[0].x <= self.length - self.traffic_signal.stop_distance / 2
                if vehicle_stop_close and vehicle_stop_far:
                    self.vehicles[0].stop()

        return vehicles_distance
