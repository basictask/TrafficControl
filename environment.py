"""
 ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄   ▄▄ ▄▄▄ ▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄▄▄▄▄▄
█       █  █  █ █  █ █  █   █   ▄  █ █       █  █  █ █  █▄█  █       █  █  █ █       █
█    ▄▄▄█   █▄█ █  █▄█  █   █  █ █ █ █   ▄   █   █▄█ █       █    ▄▄▄█   █▄█ █▄     ▄█
█   █▄▄▄█       █       █   █   █▄▄█▄█  █ █  █       █       █   █▄▄▄█       █ █   █
█    ▄▄▄█  ▄    █       █   █    ▄▄  █  █▄█  █  ▄    █       █    ▄▄▄█  ▄    █ █   █
█   █▄▄▄█ █ █   ██     ██   █   █  █ █       █ █ █   █ ██▄██ █   █▄▄▄█ █ █   █ █   █
█▄▄▄▄▄▄▄█▄█  █▄▄█ █▄▄▄█ █▄▄▄█▄▄▄█  █▄█▄▄▄▄▄▄▄█▄█  █▄▄█▄█   █▄█▄▄▄▄▄▄▄█▄█  █▄▄█ █▄▄▄█
This is the environment that handles actions, state and rewards.
The environment is explicitly meant to be used by the agent
"""
from city_constructor import Reader
from reward import RewardCalculator
from suppl import *
import os
import numpy as np
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))


class Environment:
    def __init__(self):
        """
        The constructor for the environment. Sets up parameters from config.ini and initializes the Reader object
        """
        # Set up parameters
        self.vrate = args['reader'].getint('vrate')  # Vehicle rate. Set to a higher value for more traffic.
        self.radius = args['reader'].getint('radius')  # Roundabout radius
        self.filepath = args['reader'].get('filepath')  # The construction protocol of the city
        self.n_steps = args['reader'].getint('n_steps')  # Number of steps to take in the environment. Set to 0 for indefinite.
        self.path_dist = args['reader'].get('path_dist')  # The distribution of the paths in the matrix assembler
        self.max_lanes = args['reader'].getint('max_lanes')  # The maximum number of lanes a road can have in one direction
        self.show_win = args['reader'].getboolean('show_win')  # True --> show the simulation window
        self.paths_to_gen = args['reader'].getint('paths_to_gen')  # Number of paths to generate randomly. Set to a really high number to generate all.
        self.entry_points = list(args['reader'].get('entry_points'))  # Points where vehicles can spawn. Give as a list.
        self.steps_per_update = args['reader'].getint('steps_per_update')  # Steps to take per update
        self.offset = tuple([int(x) for x in args['reader'].get('offset').split(',')])  # Offset of the view (if show_win = True)

        # Set up the reader
        self.reader = Reader(self.filepath, self.entry_points, self.vrate, self.paths_to_gen, self.path_dist)

        # Set up the rewarder
        self.rewarder = RewardCalculator()

        # State and action
        self.state = self.reader.matrix
        self.state_shape = self.state.shape  # Set up state space
        self.state_low = np.tile(0, self.state_shape)  # Lowest values for the observations
        np.fill_diagonal(self.state_low, JUNCTION_CODES[min(JUNCTION_CODES.keys())])  # Fill with the highest value junction
        self.state_high = np.tile(self.max_lanes, self.state_shape)  # Highest values for the observations
        np.fill_diagonal(self.state_high, JUNCTION_CODES[max(JUNCTION_CODES.keys())])  # Fill with the lowest value junction
        self.action_space = tuple(range(7))  # Action space - see ACTIONS for more
        self.action_shape = len(self.action_space)  # How many different actions are allowed

        check_all_attributes_initialized(self)  # Raise an error if a configuration has failed to read

    def step(self, start: int, end: int, action: int) -> (pd.DataFrame, int):
        """
        Takes a single step in the environment. If the operation is successful the state-definition matrix gets updated.
        Whether it's successful or not a reward gets calculated. The reward is an integer value as it has large scales.
        :param start: Index of the node where the infrastructure shall start
        :param end: Index of the node where the infrastructure shall end
        :param action: 0: add, 1: remove
        :return: Observation: pandas DataFrame, reward: single numerical value
        """
        action_name = ACTIONS[action]
        if action_name == 'add_lane':
            successful = self.reader.add_lane(start, end)
        elif action_name == 'remove_lane':
            successful = self.reader.remove_lane(start, end)
        elif action_name == 'add_road':
            successful = self.reader.add_road(start, end)
        elif action_name == 'remove_road':
            successful = self.reader.remove_road(start, end)
        elif action_name == 'add_righthand':
            successful = self.reader.add_righthand(end)
        elif action_name == 'add_roundabout':
            successful = self.reader.add_roundabout(end)
        elif action_name == 'add_trafficlight':
            successful = self.reader.add_trafficlight(end)
        else:
            raise IllegalActionError(f'Undefined action: {action}')

        if successful:
            roads, vehicle_mtx, signals = self.reader.get_matrices()  # Assemble new city
            total_n_vehicles, total_vehicles_distance = start_sim(roads, vehicle_mtx, self.offset, self.steps_per_update, self.n_steps, self.show_win, signals)
            self.state = self.reader.matrix  # Save observation
            reward = self.rewarder.calc_reward_successful_build(action, start, end, self.state, self.reader.points, total_n_vehicles, total_vehicles_distance)
        else:
            reward = self.rewarder.calc_reward_unsuccessful_build(action, start, end, self.state)

        return self.state, int(reward)

    def reset(self) -> pd.DataFrame:
        """
        Resets the environment in the starting state. This method only updates internal objects.
        :return: None
        """
        self.reader = Reader(self.filepath, self.entry_points, self.vrate, self.paths_to_gen, self.path_dist)  # Reset the map
        self.state = self.reader.matrix  # Reset the internal state
        return self.state

    def render_episode(self):
        roads, vehicle_mtx, signals = self.reader.get_matrices()  # Assemble new city
        show_win = True
        total_n_vehicles, total_vehicles_distance = start_sim(roads, vehicle_mtx, self.offset, self.steps_per_update, self.n_steps, show_win, signals)
        print(f'Total vehicles generated: {total_n_vehicles}, total distance taken: {total_vehicles_distance}')


# Just for testing
if __name__ == '__main__':
    env = Environment()
    # print(pretty_matrix(env.state))
    # ACTIONS = {1: 'add_lane', 2: 'remove_lane', 3: 'add_road', 4: 'remove_road', 5: 'add_righthand', 6: 'add_roundabout', 7: 'add_trafficlight'}
    s, r = env.step(1, 3, 4)
    print(s, '\n', r)
    env.render_episode()
