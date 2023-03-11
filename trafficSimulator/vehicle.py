import numpy as np


class Vehicle:
    def __init__(self, config=None):
        # Set default configuration
        if config is None:
            config = {}

        self.a_max = 1.44
        self.a = 0
        self.b_max = 4.61
        self.current_road_index = 0
        self.l = 4
        self.T = 1
        self.x = 0
        self.path = []
        self.s0 = 4
        self.stopped = False
        self.sqrt_ab = 2 * np.sqrt(self.a_max * self.b_max)
        self.v_max = 16.6
        self.v = self.v_max
        self._v_max = self.v_max

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def update(self, lead, dt):
        # Update position and velocity
        x_change = -1
        if self.v + self.a * dt < 0:
            x_change = 1 / 2 * self.v * self.v / self.a
            self.x -= x_change
            self.v = 0
        else:
            self.v += self.a * dt
            x_change = self.v * dt + self.a * dt * dt / 2
            self.x += x_change
        
        # Update acceleration
        alpha = 0
        if lead:
            delta_x = lead.x - self.x - lead.l
            delta_v = self.v - lead.v
            alpha = (self.s0 + max(0, self.T * self.v + delta_v * self.v / self.sqrt_ab)) / delta_x

        self.a = self.a_max * (1 - (self.v / self.v_max)**4 - alpha**2)

        if self.stopped: 
            self.a = -self.b_max * self.v / self.v_max

        return x_change

    def stop(self):
        self.stopped = True

    def unstop(self):
        self.stopped = False

    def slow(self, v):
        self.v_max = v

    def unslow(self):
        self.v_max = self._v_max
