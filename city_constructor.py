"""
This is the internal creator of the road configuration.
It handles adding and removing nodes from the graph.
The input is a construction from Geogebra that contains points and segments. Any other geometric shape will be ignored.
The output is a vehicle matrix and roads.

--> roads
Contains the data for road segments from node A --> B in the format (Ax, Ay) --> (Bx, By)

--> vehicle matrix
Contains the paths that vehicles take in the configuration: A --> B, B --> C, C --> D
"""


# %% Importing libraries

import os
import re
import inspect
import numpy as np
import pandas as pd
from suppl import *
from exceptions import *

# %% Class definition


class Reader:
    def __init__(self, filepath: str, entry_points: list, vrate: int, pathnum: int, path_dist: str, max_lanes: int = 3):
        """
        Constructor class for the Reader object
        :param filepath: str that points to {./folder/city_name}.html
        :param entry_points: points
        :param vrate: rate of vehicles to generate
        :param pathnum: number of individual paths a vehicle can take
        :param path_dist: distibution of paths between the total
        """
        # Error checking
        if not os.path.exists(filepath):
            raise FileNotFoundError('Input file not found: {}'.format(filepath))
        if path_dist not in ['normal', 'uniform']:  # Only valid values for distributions. Might add some later
            raise ValueError('Invalid parameter for path distribution: '.format(path_dist))

        # Params
        self.vrate = vrate
        self.pathnum = pathnum
        self.filepath = filepath
        self.path_dist = path_dist
        self.max_lanes = max_lanes
        self.entry_points = letter_to_number_lst(entry_points)  # Convert letters to numbers on entry points

        # Set up inner params
        self.locs = None
        self.graph = None
        self.paths = None
        self.points = None
        self.segments = None
        self.junctions = None
        self.paths_dual = None
        self.path_stack = None
        self.path_codes = None
        self.vehicle_mtx = None
        self.segment_map = None

        # Finalize variables
        self.read()
        self.redo_config()

    def redo_config(self) -> None:
        """
        Regenerate all the necessary configurations
        This is the base structure to set all the inner variables
        :return: None
        """
        self.gen_road_mtx()  # Create a matrix of all the roads (locs)
        self.gen_path_graph()  # Construct a graph of all nodes
        self.gen_paths()  # Find possible paths in the graph
        self.gen_vehicle_mtx()  # Create the matrix of the paths

    def assemble_segments(self, df_segments: pd.DataFrame, add_reversed: bool) -> None:
        """
        Create all the segments from df_segments
        :param df_segments: pandas DataFrame containing all the segments for any two nodes N1 --> N2
        :param add_reversed: create an one-way N1 ---> N2 or two-way N1 <--> N2 road
        :return: None
        """
        if add_reversed:  # If this is turned on all added streets will be bidirectional. Note: Ususally true only when called from self.read()
            forward_segm = list(df_segments['Definition'])
            reverse_segm = [x[::-1] for x in df_segments['Definition'] if x[::-1] not in forward_segm]  # Reverse connections that don't have reversed present already
            df_reverse = pd.DataFrame({'Definition': reverse_segm, 'N_lanes': np.ones(len(reverse_segm), dtype=int)})  # Create new DataFrame with the reversed segments
            df_segments = pd.concat([df_segments, df_reverse], axis=0, ignore_index=True, sort=False)  # Concat reversed segments to original DataFrame

        df_segments.sort_values('Definition', ignore_index=True)
        segment_map = {x: y for x, y in zip(df_segments['Definition'], list(df_segments.index))}

        self.segment_map = segment_map
        self.segments = df_segments

    def read(self) -> None:
        """
        Reads a Geogebra construction protocol that contains information about the coordinates of the nodes and the connections between them
        Example points: {0: (100.0, 100.0), 1: (250.0, 100.0), 2: (100.0, 300.0), 3: (250.0, 300.0)}
        Example df_segments: Columns=['Definition', 'N_lanes'], Definition: (start, end), N_lanes: number of lanes going from start --> end
        :return: None
        """
        df = pd.read_html(self.filepath)[0]
        df.set_index('Name', drop=True, inplace=True)
        df.sort_values(by=['Name'], inplace=True)
        ind = ['Point' in x for x in list(df.index)]

        # Separate Points and segments
        df_points = df.loc[ind].dropna(axis=1)  # Name, (x, y)
        df_segments = df.loc[[not x for x in ind]].reset_index().drop('Name', axis=1)

        # Process points
        df_points.index = [x.split(' ')[1] for x in list(df_points.index)]  # Get only letter
        df_points['Value'] = [re.findall(r'[\d.]+', x) for x in df_points['Value']]  # Find coordinates
        df_points['Value'] = [tuple([float(x) for x in lst]) for lst in df_points['Value']]  # Convert coords to float
        points = {letter_to_number(x): y for x, y in zip(list(df_points.index), df_points['Value'])}  # {ID: (x, y), ...}

        # Process segments
        df_segments['Definition'] = [re.findall(r'(?<=\()[^)]*(?=\))', x)[0] for x in df_segments['Definition']]  # Remove everything but the content in the parentheses
        df_segments['Definition'] = [re.findall(r'[A-Z]\d?', x) for x in df_segments['Definition']]  # Find the letters in the string
        df_segments['Definition'] = [tuple(letter_to_number_lst(lst)) for lst in df_segments['Definition']]  # Convert the letters to numbers
        df_segments.drop('Value', axis=1, inplace=True)
        df_segments['N_lanes'] = 1  # Assign 1 lane to each road in the starting configuration

        self.points = points  # This member holds the junctions and buffer points
        self.junctions = points  # Assign the points to the junctions member --> reference to points that are not possible to remove
        self.assemble_segments(df_segments, add_reversed=True)  # Create the segment map and list segments

    def get_n_lanes(self, start: int, end: int) -> int:
        """
        :param start: Where the beginning of the lane is
        :param end: Where the end of the lane is
        :return: Number of lanes currently going from (start --> end)
        """
        try:
            return int(self.segments.loc[self.segments['Definition'] == (start, end), 'N_lanes'])  # Number of lanes
        except TypeError:
            return 0  # In case the segment is not stored in the segments DataFrame

    def check_valid_segment(self, start: int, end: int) -> bool:
        """
        Checks if a segment is valid in order to remove it or add it
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: boolean: False --> invalid segment, True --> valid segment
        """
        caller_fn = inspect.stack()[1].function
        if start == end:  # Start and end node can't be the same
            return False
        if start not in self.points.keys() or end not in self.points.keys():  # Start and end don't exist
            return False
        if caller_fn == 'add_segment' and self.get_n_lanes(start, end) == self.max_lanes:  # Road has reached maximum capacity
            return False
        elif caller_fn == 'remove_segment' and self.get_n_lanes(start, end) == 0:  # Caller is remove_segment function but there's no segment to remove
            return False
        return True

    def add_segment(self, start: int, end: int) -> bool:
        """
        Adds a segment to the matrix between two existing points (start, end)
        Where start, end is of (x,y)
        Invalid cases --> negative reward
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: int
        """
        if self.check_valid_segment(start, end):
            num_lanes = self.get_n_lanes(start, end)
            if num_lanes == 0:  # Only one lane --> add as a new lane
                df_segments = pd.concat([self.segments, pd.DataFrame({'Definition': [(start, end)], 'N_lanes': [1]})], ignore_index=True, axis=0)
            else:
                midpoint = 1 / (num_lanes + 1)  # Calculate where between A and B the buffer point is: A ---x---> B
                coord_start = self.junctions[start]  # Get the (x, y) location of the starting point
                coord_end = self.junctions[end]  # Get the (x, y) location of the ending point
                buffer_point = calc_intermediate_point(coord_start, coord_end, midpoint)
                buffer_point_id = self.add_point(buffer_point)
                # Add the buffer point to the total points
                df_segments = pd.concat([self.segments, pd.DataFrame({'Definition': [(start, buffer_point_id),
                                                                                     (buffer_point_id, end)],
                                                                      'N_lanes': [1, 1]})], ignore_index=True, axis=0)
                df_segments.loc[df_segments['Definition'] == (start, end), 'N_lanes'] += 1

            self.assemble_segments(df_segments, add_reversed=False)
            self.redo_config()
            return True
        return False

    def remove_segment(self, start: int, end: int) -> bool:
        """
        Removes a segment specified by (start, end)
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: int
        """
        if self.check_valid_segment(start, end):  # The entered points are valid
            df_segments = self.segments
            # There's only one lane going A --> B
            if self.get_n_lanes(start, end) == 1:
                i = df_segments.loc[df_segments['Definition'] == (start, end), :].index  # Index object of segment definition
            # More lanes going A --> B ==> Find midpoint
            else:
                midpoint = 1 / (self.get_n_lanes(start, end))  # Calculate where between A and B the buffer point is: A ---x---> B
                buffer_point = calc_intermediate_point(self.junctions[start], self.junctions[end], midpoint)  # Find the point that's between start and end
                buffer_point_id = find_key_to_value(self.points, buffer_point)  # Find the ID of the buffer point
                del self.points[buffer_point_id]  # Remove the buffer point from the points dict
                i = df_segments.loc[[buffer_point_id in x for x in df_segments['Definition']], :].index  # Remove all segments that contain the buffer point ID

            if len(i) == 0:  # There's no match (this should not be possible)
                raise SegmentRemovalError('Cannot find segment: ({}, {})'.format(start, end))

            df_segments.drop(i, axis=0, inplace=True)  # Remove element with the marked index
            df_segments.index = np.arange(len(df_segments))  # Reset indices
            self.assemble_segments(df_segments, add_reversed=False)
            self.redo_config()  # Reset the environment
            return True
        return False

    def add_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <---> B e.g a bidirectional edge on the graph
        :param start: Index of the starting graph node
        :param end: Index of the ending graph node
        :return:
        """
        # Check if a road can be constructed in both ways
        if self.check_valid_segment(start, end) and self.check_valid_segment(end, start):
            self.add_segment(start, end)
            self.add_segment(end, start)
            return True
        return False

    def remove_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <---> B e.g a bidirectional edge on the graph
        We can only remove a road if a bidirectional
        :param start:
        :param end:
        :return:
        """
        if self.check_valid_segment(start, end) and self.check_valid_segment(end, start):
            self.remove_segment(start, end)
            self.remove_segment(end, start)
            return True
        return False

    def add_point(self, location: tuple) -> int:
        """
        Adds a point to the locations. No new segment gets added.
        This is an internal method only.
        Example for a list of point IDs: [0, 2, 3] ==> 1 or [0, 1, 2] ==> 3
        :param location: (x, y) coordinate tuple of the location
        :return: The identifier of the point (serves as key in the points dict)
        """
        # Find the name for the new point
        current_points = sorted(list(self.points.keys()))
        i = 0
        while i < len(current_points):  # Iterate over the points and find the next free slot
            if i != current_points[i]:  # There's a break in the assignment e.g. point was removed
                break
            i += 1
        self.points[i] = location  # Assign tuple to point location
        return i  # Return the index of the newly created point

    def gen_road_mtx(self) -> None:
        """
        Creates the matrix that is used to generate the roads
        Example locs: [((100.0, 100.0), (250.0, 100.0)), ((100.0, 300.0), (250.0, 300.0))]
        :return: None
        """
        lst = []
        for x in self.segments['Definition']:
            lst.append(tuple([self.points[x[0]], self.points[x[1]]]))
        self.locs = lst

    def gen_path_graph(self) -> None:
        """
        Creates a graph of all the nodes that can be reached from a specific node
        Example graph: {0: [1], 1: [4, 0], 2: [3], 3: [4, 2], 4: [5, 7, 8, 3, 1], 5: [6, 4], 6: [5], 7: [9, 4, 8], 8: [7, 4], 9: [7]}
        :return: None
        """
        graph = {x: [] for x in list(self.points.keys())}
        for x in self.segments['Definition']:
            graph[x[0]].append(x[1])
        self.graph = graph

    def dfs(self, g, v, seen=None, path=None) -> list:
        """
        Recursive depth seearch in a graph to determine paths between nodes
        Example paths: {0: [(0, 1, 4), (0, 1, 4, 5, 6), (0, 1, 4, 3, 2)], 1: [], 2: [(2, 3, 4), (2, 3, 4, 5, 6), (2, 3, 4, 1, 0)]}
        :param g: graph: a dict of nodes where {A: [B,C,D]} contains the nodes reachable from node A
        :param v: current node: A
        :param seen: A list of all nodes visited previously
        :param path: A the current path taken
        :return: list of nodes visited
        """
        if seen is None:
            seen = []
        if path is None:
            path = [v]

        paths = []
        seen.append(v)
        for t in g[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(self.dfs(g, t, seen[:], t_path))
        return paths

    def gen_paths(self) -> None:
        """
        Generates all the paths between entry points
        This is to determine all the possible paths that cars can take
        The different paths will be chosen from here
        The following data structures are generated here:
        Example path_stack: [[0, 3], [0, 3, 4, 9], [0, 3, 12, 11], [1, 2], [1, 2, 4, 9], [1, 2, 13, 10]]
        Example path_codes: {0: [[0, 3], [0, 3, 4, 9], [0, 3, 12, 11]], 2: [[1, 2], [1, 2, 4, 9], [1, 2, 13, 10]]}
        Example paths_dual: {0: [[(0, 1), (1, 4)], [(0, 1), (1, 4), (4, 5), (5, 6)], [(0, 1), (1, 4), (4, 3), (3, 2)]]}
        Example paths: {0: [(0, 1, 4), (0, 1, 4, 5, 6), (0, 1, 4, 3, 2)], 2: [(2, 3, 4), (2, 3, 4, 5, 6), (2, 3, 4, 1, 0)]}
        :return: None
        """
        paths = {x: [] for x in list(self.points.keys())}
        for v in self.entry_points:  # Iterate over all the entry points
            v_paths = self.dfs(self.graph, v)  # Find all the paths from an entry point
            for path in v_paths:  # Iterate over all the paths
                if path[len(path) - 1] in self.entry_points and path[len(path) - 1] != v:  # If the last node is a valid entry point
                    paths[v].append(path)  # Append to the path

        paths = drop_empty_keys(paths)
        path_codes = {x: [] for x in list(self.points.keys())}
        paths_dual = {x: [] for x in list(self.points.keys())}

        for v in paths.keys():
            dualnode = []
            c_dualnode = []
            for p in paths[v]:
                dualpath = []
                c_dualpath = []
                for i in range(1, len(p)):
                    mark = (p[i - 1], p[i])
                    dualpath.append(mark)
                    c_dualpath.append(self.segment_map[mark])
                dualnode.append(dualpath)
                c_dualnode.append(c_dualpath)
            paths_dual[v] = dualnode
            path_codes[v] = c_dualnode

        paths_dual = drop_empty_keys(paths_dual)
        path_codes = drop_empty_keys(path_codes)

        path_stack = []
        for v in path_codes.keys():
            lst = path_codes[v]
            for x in lst:
                path_stack.append(x)

        if self.pathnum > len(path_stack):
            raise Exception('The number of paths input is larger than the number of all paths.')

        self.path_stack = path_stack
        self.path_codes = path_codes
        self.paths_dual = paths_dual
        self.paths = paths

    def gen_vehicle_mtx(self) -> None:
        """
        Creates the dict that is used to create the vehicle generators
        The vehicle matrix is saved in the vehicle_mtx inner data field
        Example vehicle_rate: {'vehicle_rate': 60, 'vehicles': [[1, {'path': [0, 3, 12, 11]}], [1, {'path': [13, 10]}]}
        :return: None
        """
        gen = {'vehicle_rate': self.vrate}

        vehicles = []
        for i in range(self.pathnum):
            # Calculate the weight of the path
            weight = 0
            if self.path_dist == 'normal':  # Normal distribution
                weight = np.random.binomial(n=100, p=0.5, size=1)[0]  # Binomial approximation of the normal distribution
            elif self.path_dist == 'uniform':  # Uniform distribution
                weight = 1  # Equal weight for all paths

            # Pick a path from the path stack randomly
            path = self.path_stack[np.random.randint(0, len(self.path_stack))]
            vehicles.append([weight, {'path': path}])

        gen['vehicles'] = vehicles
        self.vehicle_mtx = gen

    def get_matrices(self):
        """
        Return the assembled matrices
        :return: The locations of the nodes (city junctions) as [x,y] coordinates; The matrix that contains the paths
        """
        return self.locs, self.vehicle_mtx


# %% If script is run


if __name__ == '__main__':
    """
    This is a short demonstration of how a Reader object can be constructed and used to display and emulate a city
    :param city: the path to the html file containing the construction protocol
    :param entry_points: nodes where a vehicle can spawn
    :param vrate: the speed of spawning vehicles
    :param path_dist: the distribution of vehicles
    """
    cities_folder = './cities/'
    filename = 'simple.html'
    city = cities_folder + filename
    vehicles_start_on = ['A', 'C', 'E', 'G']
    paths_distribution = 'uniform'
    vehicles_rate = 60
    paths_to_gen = 6

    # Construct reader
    r = Reader(city, vehicles_start_on, vehicles_rate, paths_to_gen, paths_distribution)
    roads, vehicle_mtx = r.get_matrices()
