"""
 ▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄     ▄ ▄▄▄▄▄▄ ▄▄▄▄▄▄   ▄▄▄▄▄▄  ▄▄▄▄▄▄▄ 
█   ▄  █ █       █ █ ▄ █ █      █   ▄  █ █      ██       █
█  █ █ █ █    ▄▄▄█ ██ ██ █  ▄   █  █ █ █ █  ▄    █  ▄▄▄▄▄█
█   █▄▄█▄█   █▄▄▄█       █ █▄█  █   █▄▄█▄█ █ █   █ █▄▄▄▄▄ 
█    ▄▄  █    ▄▄▄█       █      █    ▄▄  █ █▄█   █▄▄▄▄▄  █
█   █  █ █   █▄▄▄█   ▄   █  ▄   █   █  █ █       █▄▄▄▄▄█ █
█▄▄▄█  █▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄█ █▄▄█▄▄▄█  █▄█▄▄▄▄▄▄██▄▄▄▄▄▄▄█
This class is used to calculate rewards for a specific simulation 
"""
from suppl import *
import os
import numpy as np
import pandas as pd
import configparser
args = configparser.ConfigParser()
args.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))


class RewardCalculator:
    def __init__(self):
        """
        Constructor for the reward calculator. Reads values from the config and saves them as data fields.
        See config.ini for a detailed description for the description of the variables
        :return: None
        """
        # Costs
        self.cost_add_lane_unit = args['reward'].getfloat('cost_add_lane_unit')
        self.cost_remove_lane_unit = args['reward'].getfloat('cost_remove_lane_unit')
        self.cost_add_road_unit = self.cost_add_lane_unit * 2
        self.cost_remove_road_unit = self.cost_remove_lane_unit * 2

        # Junction conversions
        self.righthand_to_roundabout = args['reward'].getfloat('righthand_to_roundabout')
        self.roundabout_to_righthand = args['reward'].getfloat('roundabout_to_righthand')
        self.righthand_to_trafficlight = args['reward'].getfloat('righthand_to_trafficlight')
        self.trafficlight_to_righthand = args['reward'].getfloat('trafficlight_to_righthand')
        self.roundabout_to_trafficlight = self.roundabout_to_righthand + self.righthand_to_trafficlight
        self.trafficlight_to_roundabout = self.trafficlight_to_righthand + self.righthand_to_roundabout

        # Bonuses and penalties
        self.short_road_bonus = args['reward'].getfloat('short_road_bonus')
        self.long_road_penalty = args['reward'].getfloat('long_road_penalty')
        self.righthand_penalty = args['reward'].getfloat('righthand_penalty')
        self.multilane_penalty = args['reward'].getfloat('multilane_penalty')
        self.trafficlight_bonus = args['reward'].getfloat('trafficlight_bonus')
        self.alone_node_penalty = args['reward'].getfloat('alone_node_penalty')
        self.no_nodes_alone_bonus = args['reward'].getfloat('no_nodes_alone_bonus')
        self.additional_lane_penalty = args['reward'].getfloat('additional_lane_penalty')
        self.multilane_penalty_threshold = args['reward'].getfloat('multilane_penalty_threshold')

        # Other
        self.reward_track = pd.Series([], dtype=float)
        self.max_lanes = args['reader'].getfloat('max_lanes')

        check_all_attributes_initialized(self)  # Raise an error if a configuration has failed to read

    def calc_reward(self, action: int, start: int, end: int, matrix: pd.DataFrame, points: dict, vehicles_dist: int) -> float:
        """
        The components of a city's evaluation:
            - The cost of building the infrastructure (in accordance with real values)
            - The evaluation for the city configuration (how pedestrian-friendly a city is)
        A build can be successful if:
            - The agent adds a new junction where it's not already present
            - The agent adds a lane to a road that hasn't yet reached max_lanes
            - The agent adds a lane where none of the directions has reached max_lanes
        A build can be unsuccessful if:
            - The agent tries to remove a lane/road where there is currently nothing to remove
            - The agent tries to add a lane/road where it has already reached maximum capacity (defined by n_lanes)
            - The agent tries to add a lane/road with the starting and ending nodes being the same (impossible)
            - The agent tries to add a junction where it is already built e.g. can't add a roundabout where it's already a roundabout
            - Note: for road adding/removal it's enough for one of the lanes to be at max. capacity or at 0 for the action to be unsuccessful
        :param action: Index of the chosen action
        :param start: Index of the starting node
        :param end: Index of the ending node
        :param matrix: State-definition matrix
        :param points: Dict of point coordinates {1: (x, y), 2: (x, y), ...}
        :param vehicles_dist: Distance taken by the vehicles
        :return: Negative reward as the agent has chosen a wrong action
        """
        reward = 0
        reward += self.calc_cost_infra(start, end, action, matrix, points)
        reward += self.calc_reward_city(matrix, points)
        reward += self.calc_reward_vehicles_dist(vehicles_dist)

        self.reward_track = pd.concat([self.reward_track, pd.Series(np.int64(reward))])
        reward_result = self.reward_track.diff().fillna(0).iloc[-1]
        return int(reward_result)

    @staticmethod
    def calc_reward_vehicles_dist(total_vehicles_distance: int) -> float:
        """
        Defines how the reward calculator should assess the number of vehicles and the distance taken by them
        Note: The numerical values are calculated by the simulation if we run
        :param total_vehicles_distance: Distance taken by the vehicles
        :return: Aggregate numerical value to describe vehicle performance
        """
        return total_vehicles_distance  # / total_n_vehicles  # How much distance did a vehicle take on average

    def calc_cost_infra(self, start: int, end: int, action: int, matrix: pd.DataFrame, points: dict) -> float:
        """
        Calculates the cost of adding/removing a single piece of infrastructure like roads, lanes and junctions
        :param start: Starting node
        :param end: Ending node (for junctions only ending node is considered)
        :param action: One of the keys from the ACTIONS dict
        :param matrix: State matrix for the map
        :param points: Dict of point coordinates {1: (x, y), 2: (x, y), ...}
        :return: Numerical based on how much reward the agent gets. Not necessarily an int.
        """
        reward = 0
        action_name = ACTIONS[action]

        # Lanes and roads
        if action_name == 'add_lane':
            length = euclidean_distance(points[start], points[end])
            reward -= length * self.cost_add_lane_unit

        elif action_name == 'remove_lane':
            length = euclidean_distance(points[start], points[end])
            reward -= length * self.cost_remove_lane_unit

        elif action_name == 'add_road':
            length = euclidean_distance(points[start], points[end])
            reward -= length * self.cost_add_road_unit

        elif action_name == 'remove_road':
            length = euclidean_distance(points[start], points[end])
            reward -= length * self.cost_remove_road_unit

        # Junctions
        elif action_name == 'add_righthand':
            current_junction = JUNCTION_TYPES[matrix.loc[end, end]]
            if current_junction == 'roundabout':
                reward -= self.righthand_to_roundabout
            elif current_junction == 'trafficlight':
                reward -= self.righthand_to_trafficlight

        elif action_name == 'add_roundabout':
            current_junction = JUNCTION_TYPES[matrix.loc[end, end]]
            if current_junction == 'righthand':
                reward -= self.roundabout_to_righthand
            elif current_junction == 'trafficlight':
                reward -= self.roundabout_to_trafficlight

        elif action_name == 'add_trafficlight':
            current_junction = JUNCTION_TYPES[matrix.loc[end, end]]
            if current_junction == 'righthand':
                reward -= self.trafficlight_to_righthand
            elif current_junction == 'roundabout':
                reward -= self.trafficlight_to_roundabout

        return reward

    def calc_reward_city(self, matrix: pd.DataFrame, points: dict) -> float:
        """
        Assesses a city by a matrix and calculates the reward for it.
        Good rewards get added here like trafficlight bonuses and lane bonuses.
        :param matrix: State-definition matrix
        :param points: Dict of point coordinates {1: (x, y), 2: (x, y), ...}
        :return: Numerical reward for all the good type of infrastructure
        """
        reward = 0

        # Calculate rewards for traffic lights
        reward += self.calc_reward_junction(matrix, 'trafficlight', self.trafficlight_bonus)

        # Calculate penalty for righthand intersections
        reward -= self.calc_reward_junction(matrix, 'righthand', self.righthand_penalty)

        # Penalize nodes that are left alone with no connection to any other node
        reward += self.calc_reward_no_nodes_alone(matrix)

        # Penalize long roads, reward short roads
        reward += self.calc_long_road_penalty(matrix, points)

        return reward

    @staticmethod
    def calc_reward_junction(matrix: pd.DataFrame, junction_type: str, reward_type: float) -> float:
        """
        Calculates the reward for a given type of junction. E.g. reward traffic lights as they are pedestrian friendly
        :param matrix: State-definition matrix
        :param junction_type: Type of junction. One of ['righthand', 'roundabout', 'trafficlight']
        :param reward_type: Which bonus should he count be multiplied with
        :return: Integer reward value
        """
        return np.sum(np.diag(matrix) == JUNCTION_CODES[junction_type]) * reward_type

    def calc_reward_no_nodes_alone(self, matrix: pd.DataFrame) -> int:
        """
        Calculates the reward/penalty for any node being left alone
        :param matrix: State-definition matrix
        :return: Integer reward value
        """
        n_alone = 0
        for i in range(matrix.shape[0]):
            no_incoming = not any(x != 0 for x in pd.Series(matrix.iloc[i, :]).drop(i).reset_index(drop=True))  # Rows: all --> node connections
            no_outgoing = not any(x != 0 for x in pd.Series(matrix.iloc[:, i]).drop(i).reset_index(drop=True))  # Columns: node --> all connections
            n_alone += int(no_incoming and no_outgoing)  # If there are no incoming and no outgoing lanes from a node add 1 to counter

        if n_alone == 0:
            return self.no_nodes_alone_bonus  # If no nodes are left out apply large bonus
        return -n_alone * self.alone_node_penalty  # If there are nodes left out apply penalty to each of them

    def calc_long_road_penalty(self, matrix: pd.DataFrame, points: dict) -> float:
        """
        Penalize long, reward short roads ==> We don't want a city where there are large, uninterrupted freeways
        :param matrix: State-definition matrix
        :param points: Dict of point coordinates {1: (x, y), 2: (x, y), ...}
        :return: Integer reward value
        """
        reward = 0
        # Calculate the mean of all points
        junctions = [points[x] for x in matrix.index]  # Find the locations for all the nodes used as junctions
        midpoint = tuple(np.mean(junctions, axis=0))  # Find the midpoint of the city
        dist_thres = max([euclidean_distance(midpoint, v) for v in points.values()]) * 1.35  # Find the distance between the furthest point and midpoint

        # Iterate over the matrix and find all roads longer than necessary. Reward short and penalize long roads
        for i in matrix.index:
            for j in matrix.columns:
                if i != j and matrix.loc[i, j] > 0:
                    if euclidean_distance(points[i], points[j]) > dist_thres:
                        reward -= self.long_road_penalty * matrix.loc[i, j]
                    else:
                        reward += self.short_road_bonus
        return reward

    def calc_multilane_penalty(self, matrix: pd.DataFrame, points: dict) -> float:
        """
        Applies a penalty to all the multilane roads per direction, in correlation with the length of the road
        :param matrix: State-definition matrix
        :param points: Dict of point coordinates {1: (x, y), 2: (x, y), ...}
        :return: Reward value
        """
        reward = 0
        for i in matrix.index:
            for j in matrix.columns:
                if i != j and matrix.loc[i, j] > self.multilane_penalty_threshold:
                    reward -= euclidean_distance(points[i], points[j]) * self.multilane_penalty
                else:
                    reward += euclidean_distance(points[i], points[j]) * self.multilane_penalty
        return reward
