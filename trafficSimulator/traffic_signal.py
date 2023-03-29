import configparser
args = configparser.ConfigParser()
args.read('../config.ini')


class TrafficSignal:
    def __init__(self, roads, config=None):
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
        return self.cycle[self.current_cycle_index]
    
    def update(self, sim):
        cycle_length = 30
        k = (sim.t // cycle_length) % 2
        self.current_cycle_index = int(k)
