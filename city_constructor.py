"""
 ▄▄▄▄▄▄▄ ▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄    ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄   ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄
█       █   █       █  █ █  █  █       █       █  █  █ █       █       █   ▄  █ █  █ █  █       █       █       █   ▄  █
█       █   █▄     ▄█  █▄█  █  █       █   ▄   █   █▄█ █  ▄▄▄▄▄█▄     ▄█  █ █ █ █  █ █  █       █▄     ▄█   ▄   █  █ █ █
█     ▄▄█   █ █   █ █       █  █     ▄▄█  █ █  █       █ █▄▄▄▄▄  █   █ █   █▄▄█▄█  █▄█  █     ▄▄█ █   █ █  █ █  █   █▄▄█▄
█    █  █   █ █   █ █▄     ▄█  █    █  █  █▄█  █  ▄    █▄▄▄▄▄  █ █   █ █    ▄▄  █       █    █    █   █ █  █▄█  █    ▄▄  █
█    █▄▄█   █ █   █   █   █    █    █▄▄█       █ █ █   █▄▄▄▄▄█ █ █   █ █   █  █ █       █    █▄▄  █   █ █       █   █  █ █
█▄▄▄▄▄▄▄█▄▄▄█ █▄▄▄█   █▄▄▄█    █▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄█  █▄▄█▄▄▄▄▄▄▄█ █▄▄▄█ █▄▄▄█  █▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█ █▄▄▄█ █▄▄▄▄▄▄▄█▄▄▄█  █▄█
This is the internal creator of the road configuration.
It handles adding and removing nodes from the graph.
The input is a construction from Geogebra that contains points and segments. Any other geometric shape will be ignored.
The output is a vehicle matrix and roads.

--> roads
Contains the data for road segments from node A --> B in the format (Ax, Ay) --> (Bx, By)

--> vehicle matrix
Contains the paths that vehicles take in the configuration: A --> B, B --> C, C --> D
"""
import os
import re
import inspect
from matrix_assembler import *


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
        self.filepath = filepath
        self.max_lanes = max_lanes
        entry_points = letter_to_number_lst(entry_points)   # Convert letters to numbers on entry points
        self.assembler = Assembler(entry_points, vrate, pathnum, path_dist, max_lanes)

        # Set up inner params
        self.locs = None
        self.graph = None
        self.paths = None
        self.matrix = None
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
        self.segments = self.assembler.redo_config(df_segments=df_segments, points=points, add_reversed=True)  # Create the segment map and list segments
        self.init_matrix()

    def init_matrix(self) -> None:
        """
        This is the setup function for the inner representation of the graph object
        The matrix variable is one of the most important. It stores the number of lanes going from A --> B and the type of junction.
        matrix.loc[A, B] refers to the number of lanes going from A to B. Legitimate values are 0...max_lanes
        matrix.loc[A, A] refers to the type of junction in the node A. Legitimate values are 1: right-hand, 2: roundabout, 3: traffic light
        :return: None
        """
        ind = sorted(list(self.points.keys()))
        self.matrix = pd.DataFrame(index=ind, columns=ind)  # Create the object
        self.matrix.fillna(0, inplace=True)  # Fill up with 0
        # Iterate over the junctions and roads and fill up the missing values
        for P in self.junctions.keys():
            self.matrix.loc[P, P] = 1  # By default set to right-hand crossing
        for start, end in self.segments['Definition']:
            self.matrix.loc[start, end] = 1  # By default set to 1 lane going from A to B

    def check_valid_segment(self, start: int, end: int) -> bool:
        """
        Checks if a segment is valid in order to remove it or add it
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: False for invalid segment, True for valid segment
        """
        caller_fn = inspect.stack()[1].function
        if start == end:  # Start and end node can't be the same
            return False
        if start not in self.points.keys() or end not in self.points.keys():  # Start and end don't exist
            return False
        if caller_fn == 'add_lane' and self.matrix.loc[start, end] == self.max_lanes:  # Road has reached maximum capacity
            return False
        elif caller_fn == 'remove_lane' and self.matrix.loc[start, end] == 0:  # Caller is remove_lane function but there's no segment to remove
            return False
        return True

    def add_lane(self, start: int, end: int) -> bool:
        """
        Adds a segment to the matrix between two existing points (start, end)
        Where start, end is of (x,y)
        Invalid cases --> negative reward
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: True for successful, False for unsuccessful
        """
        if self.check_valid_segment(start, end):
            num_lanes = self.matrix.loc[start, end]
            # The number of lanes has reached the maximum
            if num_lanes == self.max_lanes:
                return False
            # Only one lane ==> add as a new lane
            elif num_lanes == 0:
                df_segments = pd.concat([self.segments, pd.DataFrame({'Definition': [(start, end)], 'N_lanes': [1]})], ignore_index=True, axis=0)
            # There's 1 or more lanes going from start to end ==> we add a midpoint and two segments
            else:
                midpoint = 1 / (num_lanes + 1)  # Calculate where between A and B the buffer point is: A ---x---> B
                coord_start = self.junctions[start]  # Get the (x, y) location of the starting point
                coord_end = self.junctions[end]  # Get the (x, y) location of the ending point
                buffer_point = calc_intermediate_point(coord_start, coord_end, midpoint)
                buffer_point_id = self.add_point(buffer_point)
                # Add the buffer point to the total points
                df_segments = pd.concat([self.segments, pd.DataFrame({'Definition': [(start, buffer_point_id), (buffer_point_id, end)],
                                                                      'N_lanes': [1, 1]})], ignore_index=True, axis=0)
            self.matrix.loc[start, end] += 1
            self.segments = self.assembler.redo_config(df_segments=df_segments, points=self.points, add_reversed=False)  # Create the segment map and list segments
            return True
        return False

    def remove_lane(self, start: int, end: int) -> bool:
        """
        Removes a segment specified by (start, end)
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: True for successful, False for unsuccessful
        """
        if self.check_valid_segment(start, end):  # The entered points are valid
            df_segments = self.segments
            # There are no lanes going from A to B
            if self.matrix.loc[start, end] == 0:
                return False
            # There's only one lane going A --> B
            elif self.matrix.loc[start, end] == 1:
                i = df_segments.loc[df_segments['Definition'] == (start, end), :].index  # Index object of segment definition
            # More lanes going A --> B ==> Find midpoint
            else:
                midpoint = 1 / (self.matrix.loc[start, end])  # Calculate where between A and B the buffer point is: A ---x---> B
                buffer_point = calc_intermediate_point(self.junctions[start], self.junctions[end], midpoint)  # Find the point that's between start and end
                buffer_point_id = find_key_to_value(self.points, buffer_point)  # Find the ID of the buffer point
                del self.points[buffer_point_id]  # Remove the buffer point from the points dict
                i = df_segments.loc[[buffer_point_id in x for x in df_segments['Definition']], :].index  # Remove all segments that contain the buffer point ID

            if len(i) == 0:  # There's no match (this should not be possible)
                raise SegmentRemovalError('Cannot find segment: ({}, {})'.format(start, end))

            self.matrix.loc[start, end] -= 1  # Decrease lane counter in matrix
            df_segments.drop(i, axis=0, inplace=True)  # Remove element with the marked index
            df_segments.index = np.arange(len(df_segments))  # Reset indices
            self.segments = self.assembler.redo_config(df_segments=df_segments, points=self.points, add_reversed=False)  # Create the segment map and list segments
            return True
        return False

    def add_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <---> B e.g a bidirectional edge on the graph
        :param start: Index of the starting graph node
        :param end: Index of the ending graph node
        :return: True for successful, False for unsuccessful
        """
        # Check if a road can be constructed in both ways
        if self.check_valid_segment(start, end) and self.check_valid_segment(end, start):
            self.add_lane(start, end)
            self.add_lane(end, start)
            return True
        return False

    def remove_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <---> B e.g a bidirectional edge on the graph
        We can only remove a road if a bidirectional
        :param start: Index of the starting graph node
        :param end: Index of the ending graph node
        :return: True for successful, False for unsuccessful
        """
        if self.check_valid_segment(start, end) and self.check_valid_segment(end, start):
            self.remove_lane(start, end)
            self.remove_lane(end, start)
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

    def get_matrices(self):
        """
        Return the assembled matrices. This method calls the method of the same name from the assembler class
        :return: The locations of the nodes (city junctions) as [x,y] coordinates; The matrix that contains the paths
        """
        return self.assembler.get_matrices()


#%% This is for testing only. Run the script and see how an assembled vehicle matrix looks.


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
