from scipy.spatial import distance
from collections import deque


class Road:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.traffic_signal = None
        self.traffic_signal_group = None

        self.vehicles = deque()
        self.length = distance.euclidean(self.start, self.end)
        self.angle_sin = (self.end[1] - self.start[1]) / self.length
        self.angle_cos = (self.end[0] - self.start[0]) / self.length
        # self.angle = np.arctan2(self.end[1]-self.start[1], self.end[0]-self.start[0])
        self.has_traffic_signal = False

    def set_traffic_signal(self, signal, group):
        self.traffic_signal = signal
        self.traffic_signal_group = group
        self.has_traffic_signal = True

    @property
    def traffic_signal_state(self):
        if self.has_traffic_signal:
            i = self.traffic_signal_group
            return self.traffic_signal.current_cycle[i]
        return True

    def update(self, dt):
        n = len(self.vehicles)
        vehicles_distance = 0
        if n > 0:
            vehicles_distance += self.vehicles[0].update(None, dt)  # Update first vehicle

            for i in range(1, n):  # Update other vehicles
                lead = self.vehicles[i-1]
                vehicles_distance += self.vehicles[i].update(lead, dt)

            # Check for traffic signal
            if self.traffic_signal_state:  # If traffic signal is green or doesn't exist then let vehicles pass
                self.vehicles[0].unstop()
                for vehicle in self.vehicles:
                    vehicle.unslow()
            else:  # If traffic signal is red
                if self.vehicles[0].x >= self.length - self.traffic_signal.slow_distance:
                    # self.vehicles[0].slow(self.traffic_signal.slow_factor * self.vehicles[0]._v_max)  # Slow vehicles in slowing zone
                    self.vehicles[0].slow(self.traffic_signal.slow_factor * self.vehicles[0].get__v_max)  # Slow vehicles in slowing zone

                # Check if the vehicle is in the stop zone
                vehicle_stop_close = self.vehicles[0].x >= self.length - self.traffic_signal.stop_distance
                vehicle_stop_far = self.vehicles[0].x <= self.length - self.traffic_signal.stop_distance / 2
                if vehicle_stop_close and vehicle_stop_far:
                    self.vehicles[0].stop()

        return vehicles_distance
