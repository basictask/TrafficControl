import os
import numpy as np
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.ini'))


class Vehicle:
    def __init__(self, config):
        """
        Sets up a vehicle object. Called by the VehicleGenerator
        :param config: Optional Simulation object
        """
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
        """
        This is the Intelligent Driver Model implementation that defines the position of the vehicle
        :param lead: The vehicle in front of this object
        :param dt: Delta time variable
        :return: Change of position on the road
        """
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
        """
        Called when a traffic light stops a vehicle
        :return: None
        """
        self.stopped = True

    def unstop(self):
        """
        Called when a traffic light turns to green to the first vehicle in its stopping zone
        :return: None
        """
        self.stopped = False

    def slow(self, v):
        """
        Sets the maximum velocity to an updated lower velocity
        :param v: New velocity of the vehicle
        :return: None
        """
        self.v_max = v

    def unslow(self):
        """
        Sets the maximum velocity to the absolute maximum (the absolute maximum is unchangable)
        :return: None
        """
        self.v_max = self._v_max

    @property
    def get__v_max(self):
        """
        :return: The absolute maximum velocity (float)
        """
        return self._v_max
