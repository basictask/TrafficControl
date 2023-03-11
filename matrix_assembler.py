"""
This is a supportive class that is used to construct the matrices from the segments DataFrame and the location of the points
The class is used to generate two matrices:

locs: stores the locations of the segments. For segment A --> B where each node is defined as (x, y) an entry in locs is ((x, y), (x, y))
Example for locs
    [
        ((300, 98), (0, 98)),
        ((0, 102), (300, 102)),
        ((180, 60), (0, 60)),
        ((220, 55), (180, 60)),
        ((300, 30), (220, 55)),
        ((180, 60), (160, 98)),
        ((158, 130), (300, 130)),
        ((0, 178), (300, 178)),
        ((300, 182), (0, 182)),
        ((160, 102), (155, 180))
    ]

vehicle_mtx: stores which path should we take. For each path the integer entries refer to which road should the vehicle take next.
E.g. path [4,3,2] refers to locs[4] --> locs[3] --> locs[2] where each location defines an ((x, y), (x, y)) coordinate tuple pair.
Example for vehicle_mtx:
    {
        'vehicle_rate': 60,
        'vehicles': [
            [1, {"path": [4, 3, 2]}],
            [1, {"path": [0]}],
            [1, {"path": [1]}],
            [1, {"path": [6]}],
            [1, {"path": [7]}]
        ]
    }
"""

import numpy as np
import pandas as pd
from suppl import *


class Assembler:
    def __init__(self, entry_points: list, vrate: int, pathnum: int, path_dist: str, max_lanes: int):
        self.locs = None
        self.paths = None
        self.graph = None
        self.points = None
        self.segments = None
        self.paths_dual = None
        self.path_codes = None
        self.path_stack = None
        self.vehicle_mtx = None
        self.segment_map = None

        self.vrate = vrate
        self.pathnum = pathnum
        self.max_lanes = max_lanes
        self.path_dist = path_dist
        self.entry_points = entry_points

    def redo_config(self, df_segments: pd.DataFrame, points: dict, add_reversed: bool) -> pd.DataFrame:
        """
        Regenerate all the necessary configurations
        This is the base structure to set all the inner variables
        :return: None
        """
        self.points = points
        self.assemble_segments(df_segments, add_reversed)
        self.gen_road_mtx()  # Create a matrix of all the roads (locs)
        self.gen_path_graph()  # Construct a graph of all nodes
        self.gen_paths()  # Find possible paths in the graph
        self.gen_vehicle_mtx()  # Create the matrix of the paths
        return self.segments

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
            raise TooManyPathsError('The number of paths input is larger than the number of all paths.')

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
