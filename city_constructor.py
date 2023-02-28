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


# TODO: add_point, add_segment, redo_config

# %% Class definition


class Reader:
    def __init__(self, filepath, points, vehicle_rate, pathnum, distribution):
        # Error checking
        if distribution not in ['normal', 'uniform']:
            raise Exception('Invalid parameter for path distribution: ' + distribution)
        if not os.path.exists(filepath):
            raise Exception('Input file not found: ' + filepath)

        # Params
        self.pathnum = pathnum
        self.filepath = filepath
        self.vrate = vehicle_rate
        self.entry_points = points
        self.path_dist = distribution

        # Set up inner params
        self.locs = None
        self.graph = None
        self.paths = None
        self.points = None
        self.segments = None
        self.paths_dual = None
        self.path_stack = None
        self.path_codes = None
        self.vehicle_mtx = None
        self.segment_map = None
        self.read()  # Read the file
        self.redo_config()  # Re-generate all the necessary configurations

    def redo_config(self):
        # Regenerate all the necessary configurations
        self.gen_road_mtx()  # Create a matrix of all the roads (locs)
        self.gen_path_graph()  # Construct a graph of all nodes
        self.gen_paths()  # Find possible paths in the graph
        self.gen_vehicle_mtx()  # Create the matrix of the paths

    def assemble_segments(self, df_segments, add_reversed):
        # Assemble into segment map
        if add_reversed:  # If this is turned on all added streets will be bidirectional
            forward_segm = list(df_segments['Definition'])
            reverse_segm = [(x[1], x[0]) for x in df_segments['Definition']]  # Reverse all connections
            forward_segm.extend(reverse_segm)
            df_segments = pd.DataFrame({'Definition': forward_segm}, index=range(len(forward_segm)))

        df_segments.sort_values('Definition', ignore_index=True)
        segment_map = {x: y for x, y in zip(df_segments['Definition'], list(df_segments.index))}

        self.segment_map = segment_map
        self.segments = df_segments

    def read(self):
        # Read the file
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
        df_points['Value'] = [[float(x) for x in lst] for lst in df_points['Value']]  # Convert coords to float
        df_points['Value'] = [tuple(x) for x in df_points['Value']]
        locations = {x: y for x, y in zip(list(df_points.index), df_points['Value'])}

        # Process segments
        df_segments['Definition'] = [re.findall(r'(?<=\()[^)]*(?=\))', x)[0] for x in df_segments['Definition']]
        df_segments['Definition'] = [x.split(' ') for x in df_segments['Definition']]
        for i in range(len(df_segments)):
            df_segments['Definition'][i][0] = df_segments['Definition'][i][0][:-1]
        df_segments['Definition'] = [tuple(x) for x in df_segments['Definition']]
        df_segments.drop('Value', axis=1, inplace=True)

        self.points = locations
        self.assemble_segments(df_segments, add_reversed=True)  # Create the segment map and list segments

    def check_valid_segment(self, start: str, end: str) -> bool:
        """
        Checks if a segment is valid in order to remove it or add it
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: boolean: False --> invalid segment, True --> valid segment
        """
        caller_fn = inspect.stack()[1].function
        if start == end:
            return False
        elif start not in self.points.keys() or end not in self.points.keys():
            return False
        elif caller_fn == 'add_segment' and (start, end) in set(self.segments['Definition']):
            return False
        elif caller_fn == 'remove_segment' and (start, end) not in set(self.segments['Definition']):
            return False
        else:
            return True

    def add_segment(self, start: str, end: str) -> int:
        """
        Adds a segment to the matrix between existing points
        Invalid cases --> negative reward
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: int
        """
        if self.check_valid_segment(start, end):
            df_segments = pd.concat([self.segments, pd.DataFrame({'Definition': [(start, end)]})], ignore_index=True, axis=0)
            self.assemble_segments(df_segments, add_reversed=False)
            self.redo_config()
            return 1
        else:
            return 0

    def remove_segment(self, start: str, end: str) -> int:
        """
        Removes a segment specified by start and end
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: int
        """
        if self.check_valid_segment(start, end):  # The entered points are valid
            df_segments = self.segments

            i = 0
            while df_segments['Definition'][i] != (start, end):
                i += 1
            if i == len(df_segments):
                raise Exception('Cannot find segment: (' + start + ', ' + end + ')')

            df_segments.drop(i, axis=0, inplace=True)  # Remove element with the marked index
            df_segments.index = np.arange(len(df_segments))  # Reset indices
            self.assemble_segments(df_segments, add_reversed=False)
            self.redo_config()  # Reset the environment
            return 1
        else:
            return 0

    def gen_road_mtx(self) -> None:
        """
        Creates the matrix that is used to generate the roads
        :return: None
        """
        lst = []
        for x in self.segments['Definition']:
            lst.append(tuple([self.points[x[0]], self.points[x[1]]]))
        self.locs = lst

    def gen_path_graph(self) -> None:
        """
        Creates a graph of all the nodes that can be reached from a specific node
        :return:
        """
        graph = {x: [] for x in list(self.points.keys())}
        for x in self.segments['Definition']:
            graph[x[0]].append(x[1])
            # graph[x[1]].append(x[0]) # Only needed for explicit adding
        self.graph = graph

    def dfs(self, g, v, seen=None, path=None) -> list:
        """
        Recursive depth seearch in a graph to determine paths between nodes
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

        seen.append(v)

        paths = []
        for t in g[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(self.dfs(g, t, seen[:], t_path))
        return paths

    @staticmethod
    def drop_empty_keys(dct: dict) -> dict:
        """
        Drops the empty keys from a given dict and returns the dict itself
        :param dct: The dict to drop from
        :return: dict: The dict with the empty keys removed
        """
        for v in list(dct.keys()):
            if len(dct[v]) == 0:
                dct.pop(v, None)
        return dct

    def gen_paths(self) -> None:
        """
        Generates all the paths between entry points
        This is to determine all the possible paths that cars can take.
        The different paths will be chosen from here
        :return: None
        """
        paths = {x: [] for x in list(self.points.keys())}
        for v in self.entry_points:  # Iterate over all the entry points
            v_paths = self.dfs(self.graph, v)  # Find all the paths from an entry point
            for path in v_paths:  # Iterate over all the paths
                if path[len(path) - 1] in self.entry_points and path[len(path) - 1] != v:  # If the last node is a valid entry point
                    paths[v].append(path)  # Append to the path

        paths = self.drop_empty_keys(paths)

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

        paths_dual = self.drop_empty_keys(paths_dual)
        path_codes = self.drop_empty_keys(path_codes)

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
        :return: The locations of the nodes (city junctions) as [x,y] coordinates and the matrix that contains the paths
        """
        return self.locs, self.vehicle_mtx


# %% If script is run


if __name__ == '__main__':
    """
    This is a short demonstration of how a Reader object can be constructed and used to display and emulate a city
    
    @param city: the path to the html file containing the construction protocol
    @param entry_points: nodes where a vehicle can spawn
    @param vrate: the speed of spawning vehicles
    @param path_dist: the distribution of vehicles
    """
    CITIES_FOLDER = './cities/'
    filename = 'simple.html'
    city = CITIES_FOLDER + filename
    entry_points = ['A', 'C', 'E', 'G']
    vrate = 60
    paths_to_gen = 6
    path_dist = 'uniform'

    # Construct reader
    r = Reader(city, entry_points, vrate, paths_to_gen, path_dist)
    roads, vehicle_mtx = r.get_matrices()
