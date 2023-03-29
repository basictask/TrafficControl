import numpy as np
import configparser
args = configparser.ConfigParser()
args.read('../config.ini')


class Vehicle:
    def __init__(self, config):
        if config is None:
            config = {}

        self.a_max = args['vehicle'].getfloat('a_max')
        self.b_max = args['vehicle'].getfloat('b_max')
        self.l = args['vehicle'].getint('l')
        self.T = args['vehicle'].getint('T')
        self.s0 = args['vehicle'].getint('s0')
        self.v_max = args['vehicle'].getfloat('v_max')

        # Set default configuration
        self.a = 0
        self.x = 0
        self.path = []
        self.v = self.v_max
        self.stopped = False
        self._v_max = self.v_max
        self.current_road_index = 0
        self.sqrt_ab = 2 * np.sqrt(self.a_max * self.b_max)

        for attr, val in config.items():
            setattr(self, attr, val)

    def update(self, lead, dt):
        # Update position and velocity
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

    @property
    def get__v_max(self):
        return self._v_max
