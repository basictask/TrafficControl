import os
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.ini'))


class TrafficSignal:
    def __init__(self, roads, config=None):
        """
        Initializes a traffic signal for a given set of roads
        :param roads: A list of road indices by pairs. The pairs define the traffic light cycle
        :param config: an optional simulation object that is to be passed to the signal
        """
        if config is None:
            config = {}

        self.slow_factor = args['trafficlight'].getfloat('slow_factor')
        self.slow_distance = args['trafficlight'].getint('slow_distance')
        self.stop_distance = args['trafficlight'].getint('stop_distance')

        # Set default configuration
        self.roads = roads
        self.last_t = 0
        self.current_cycle_index = 0
        self.cycle = [(False, True), (True, False)]

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        # Calculate properties
        for i in range(len(self.roads)):
            for road in self.roads[i]:
                road.set_traffic_signal(self, i)

    @property
    def current_cycle(self):
        """
        :return: True: the light is green, False: the light is red
        """
        return self.cycle[self.current_cycle_index]
    
    def update(self, sim):
        """
        Changes the state of the traffic light cycle by stepping in the traffic light index
        Essentially sets the index to point to the updated boolean defined in the cycle
        :param sim: Simulation object that is to be passed to the traffic light update
        :return: None
        """
        cycle_length = 30
        k = (sim.t // cycle_length) % 2
        self.current_cycle_index = int(k)
