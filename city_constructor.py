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
from matrix_assembler import *
import re
import os
import configparser
args = configparser.ConfigParser()
args.read('/home/daniel/Documents/ELTE/trafficControl/config.ini')


class Reader:
    def __init__(self, filepath: str, entry_points: list, vrate: int, pathnum: int, path_dist: str):
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
        self.max_lanes = args['reader'].getint('max_lanes')
        self.roundabout_radius = args['reader'].getint('radius')
        self.trafficlight_inbound = [int(x) for x in args['trafficlight'].get('allow_inbound').split(',')]  # How many incoming lanes trafficlights allow
        self.entry_points = letter_to_number_lst(entry_points)   # Convert letters to numbers on entry points
        self.assembler = Assembler(self.entry_points, vrate, pathnum, path_dist, self.max_lanes)

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

        # Check if all entry points are legal
        check_all_coords_valid(points, self.max_lanes, self.roundabout_radius)
        check_entry_points_valid(points, self.entry_points)

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
        for node in self.junctions.keys():
            self.matrix.loc[node, node] = 1  # By default set to right-hand crossing
        for start, end in self.segments['Definition']:
            self.matrix.loc[start, end] = 1  # By default set to 1 lane going from A to B

    def check_valid_segment(self, start: int, end: int) -> bool:
        """
        Checks if a segment is valid in order to remove it or add it
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: False: invalid segment, True: valid segment
        """
        caller_fn = inspect.stack()[1].function
        if start < 0 or end < 0:  # All point indices must be positive
            return False
        if start == end:  # Start and end node can't be the same
            return False
        if start not in self.points.keys() or end not in self.points.keys():  # Start and end don't exist
            return False
        if start in self.matrix.index and end in self.matrix.index:  # If start or end are not in the matrix they are buffer points
            n_incoming = count_incoming_lanes(self.segments['Definition'], self.points, end, unique=True)
            if self.matrix.loc[end, end] == JUNCTION_CODES['trafficlight'] and n_incoming not in self.trafficlight_inbound:  # Trafficlight intersection at max. capacity
                return False
            if caller_fn == 'add_lane' and self.matrix.loc[start, end] == self.max_lanes:  # Road has reached maximum capacity
                return False
            elif caller_fn == 'remove_lane' and self.matrix.loc[start, end] == 0:  # Caller is remove_lane function but there's no segment to remove
                return False
        return True

    def get_n_lanes(self, start, end):
        """
        Counts the number of lanes going start --> end
        :param start: (x, y) coordinates of the starting node
        :param end: (x, y) coordinates of the ending node
        :return: Number of lanes in one direction
        """
        if start in self.matrix.index and end in self.matrix.index:
            return self.matrix.loc[start, end]
        else:
            return 0

    def add_lane(self, start: int, end: int) -> bool:
        """
        Adds a segment to the matrix between two existing points (start, end)
        Where start, end is of (x,y)
        Invalid cases --> negative reward
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: True: successful, False: unsuccessful
        """
        if self.check_valid_segment(start, end):
            # If there's a roundabout on the junction: 1. Remove roundabout, 2. Reconfigure, 3. Reinstall roundabout
            if self.matrix.loc[end, end] == JUNCTION_CODES['roundabout']:
                self.remove_roundabout(end)
            if self.matrix.loc[start, start] == JUNCTION_CODES['roundabout']:
                self.remove_roundabout(start)

            # The number of lanes has reached the maximum
            num_lanes = self.get_n_lanes(start, end)
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

            # Add the roundabout again
            if self.matrix.loc[end, end] == JUNCTION_CODES['roundabout']:
                self.add_roundabout(end)
            if self.matrix.loc[start, start] == JUNCTION_CODES['roundabout']:
                self.add_roundabout(start)

            self.assembler.gen_signal_list(self.matrix)  # Re-generate the list of signals in the Assembler
            return True
        return False

    def remove_lane(self, start: int, end: int) -> bool:
        """
        Removes a segment specified by (start, end)
        :param start: Node where the beginning of the road is
        :param end: Node where the end of the road is
        :return: True: successful, False: unsuccessful
        """
        if self.check_valid_segment(start, end):  # The entered points are valid
            # If there's a roundabout on the junction: 1. Remove roundabout, 2. Reconfigure, 3. Reinstall roundabout
            if self.matrix.loc[end, end] == JUNCTION_CODES['roundabout']:
                self.remove_roundabout(end)
            if self.matrix.loc[start, start] == JUNCTION_CODES['roundabout']:
                self.remove_roundabout(start)

            df_segments = self.segments
            num_lanes = self.get_n_lanes(start, end)
            # There are no lanes going A --> B
            if num_lanes == 0:
                return False
            # There's only one lane going A --> B
            elif num_lanes == 1:
                i = df_segments.loc[df_segments['Definition'] == (start, end), :].index  # Index object of segment definition
            # More lanes going A --> B ==> Find midpoint
            else:
                midpoint = 1 / (self.matrix.loc[start, end])  # Calculate where between A and B the buffer point is: A ---x---> B
                buffer_point = calc_intermediate_point(self.junctions[start], self.junctions[end], midpoint)  # Find the point that's between start and end
                buffer_point_id = find_key_to_value(self.points, buffer_point)  # Find the ID of the buffer point
                self.points.pop(buffer_point_id, None)  # Remove the buffer point from the points dict
                i = df_segments.loc[[buffer_point_id in x for x in df_segments['Definition']], :].index  # Remove all segments that contain the buffer point ID

            if len(i) == 0:  # There's no match (this should not be possible)
                raise SegmentRemovalError('Cannot find segment: ({}, {})'.format(start, end))

            self.matrix.loc[start, end] -= 1  # Decrease lane counter in matrix
            df_segments.drop(i, axis=0, inplace=True)  # Remove element with the marked index
            df_segments.index = np.arange(len(df_segments))  # Reset indices
            self.segments = self.assembler.redo_config(df_segments=df_segments, points=self.points, add_reversed=False)  # Create the segment map and list segments

            # Add the roundabout again
            if self.matrix.loc[end, end] == JUNCTION_CODES['roundabout']:
                self.add_roundabout(end)
            if self.matrix.loc[start, start] == JUNCTION_CODES['roundabout']:
                self.add_roundabout(start)

            self.assembler.gen_signal_list(self.matrix)  # Re-generate the list of signals in the Assembler
            return True
        return False

    def add_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <--> B e.g a bidirectional edge on the graph
        :param start: Index of the starting graph node
        :param end: Index of the ending graph node
        :return: True: successful, False: unsuccessful
        """
        if self.check_valid_segment(start, end) and self.check_valid_segment(end, start):  # Check if a road can be constructed in both ways
            self.add_lane(start, end)
            self.add_lane(end, start)
            return True
        return False

    def remove_road(self, start: int, end: int) -> bool:
        """
        A road is a segment both ways defined as: A <--> B e.g a bidirectional edge on the graph
        We can only remove a road if a bidirectional
        :param start: Index of the starting graph node
        :param end: Index of the ending graph node
        :return: True: successful, False: unsuccessful
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
        current_points = sorted(list(self.points.keys()))  # Find the name for the new point
        i = 0
        while i < len(current_points):  # Iterate over the points and find the next free slot
            if i != current_points[i]:  # There's a break in the assignment e.g. point was removed
                break
            i += 1
        if i < 0:
            raise IllegalPointIDError(f'Point added with illegla ID: {i}')
        self.points[i] = location  # Assign tuple to point location
        return i  # Return the index of the newly created point

    def add_righthand(self, node: int) -> bool:
        """
        Converts any type of junction into a right-hand priority intersection
        Sets the parameter for the right-hand intersection in the representation matrix
        Note: in order to be converted to right-hand, the junction must have at least as many incoming
        lanes as defined by the min/max values of the other types of infrastructure
        :param node: The index of the junction to be converted into right-hand
        :return: True: successful, False: unsuccessful
        """
        # Remove roundabout and its' buffer nodes if they exist
        if self.matrix.loc[node, node] == JUNCTION_CODES['roundabout']:
            self.remove_roundabout(node)
            self.assembler.gen_signal_list(self.matrix)  # Re-generate the list of signals in the Assembler
            return True
        elif self.matrix.loc[node, node] != JUNCTION_CODES['righthand']:  # Check if it's currently a right-hand intersection
            self.matrix.loc[node, node] = JUNCTION_CODES['righthand']  # Set the node type to right-hand in the matrix
            self.assembler.gen_signal_list(self.matrix)  # Re-generate the list of signals in the Assembler
            return True
        return False

    def add_roundabout(self, node: int) -> bool:
        """
        Adds a roundabout to an intersection defined by the location of the node
        :param node: The index of the junction to be converted into a roundabout
        :return: True: successful, False: unsuccessful
        """
        caller_fn = inspect.stack()[1].function
        if self.matrix.loc[node, node] != JUNCTION_CODES['roundabout'] or caller_fn == 'add_lane':
            node_coords = self.points[node]
            df_segments = self.segments

            # Find the nodes connected to the node we are adding the roundabout to
            connected_nodes = {}
            for start, end in df_segments['Definition']:
                if start == node:
                    connected_nodes[end] = self.points[end]
                elif end == node:
                    connected_nodes[start] = self.points[start]

            # Create new coordinates on the perimeter of the roundabout
            connected_nodes = pd.DataFrame(pd.Series(connected_nodes), columns=['end'])
            connected_nodes['buffer_coord'] = [find_closest_point_circle(x, node_coords, self.roundabout_radius) for x in connected_nodes['end']]
            connected_nodes['is_first'] = ~connected_nodes['buffer_coord'].duplicated(keep='first')
            connected_nodes['buffer_ind'] = [self.add_point(x) if y else -1 for x, y in zip(connected_nodes['buffer_coord'], connected_nodes['is_first'])]
            connected_nodes['angle'] = [find_angle(node_coords, x, absolute=False) for x in connected_nodes['end']]

            # Eliminate duplicate connections
            buffer_node_angle = dict(connected_nodes.groupby('angle').first().loc[:, 'buffer_ind'])
            connected_nodes['buffer_ind'] = [buffer_node_angle[x] for x in connected_nodes['angle']]

            # Crate lanes between the buffer points
            for ind in df_segments.index:
                connection = df_segments.loc[ind, 'Definition']
                if connection[0] == node:
                    buffer_ind = connected_nodes.loc[connection[1], 'buffer_ind']
                    df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [(buffer_ind, connection[1])], 'N_lanes': [1]})], ignore_index=True, axis=0)
                elif connection[1] == node:
                    buffer_ind = connected_nodes.loc[connection[0], 'buffer_ind']
                    df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [(connection[0], buffer_ind)], 'N_lanes': [1]})], ignore_index=True, axis=0)

            # Drop all connections to the central node
            df_segments.drop(df_segments[[node in x for x in df_segments['Definition']]].index, axis=0, inplace=True)
            df_segments.reset_index(drop=True, inplace=True)

            # Connect all the buffer nodes
            connected_nodes.sort_values(by='angle', inplace=True)
            connected_nodes.drop_duplicates(subset='buffer_ind', keep='first', inplace=True, ignore_index=True)
            for i in range(1, len(connected_nodes)):
                connection = (connected_nodes.iloc[i].loc['buffer_ind'], connected_nodes.iloc[i - 1].loc['buffer_ind'])
                df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [connection], 'N_lanes': [1]})], ignore_index=True, axis=0)
                if i == len(connected_nodes) - 1:
                    connection = (connected_nodes.iloc[0].loc['buffer_ind'], connected_nodes.iloc[i].loc['buffer_ind'])
                    df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [connection], 'N_lanes': [1]})], ignore_index=True, axis=0)

            # check_valid_df_segments(df_segments, self.points)  # For debugging

            # Reassemble
            self.matrix.loc[node, node] = JUNCTION_CODES['roundabout']
            check_valid_df_segments(df_segments, points=self.points)
            self.segments = self.assembler.redo_config(df_segments=df_segments, points=self.points, add_reversed=False)  # Create the segment map and list segments
            self.assembler.gen_signal_list(self.matrix)
            return True
        return False

    def remove_roundabout(self, node: int) -> None:
        """
        Removes all the configuration of a roundabout at a given node
        Note: The agent is not able to call this method directly therefore its void. This step is just a preparation step to convert to other types of junctions
        :param node: The index of the junction to be converted into roundabout
        :return: True: successful, False: unsuccessful
        """
        df_segments = self.segments

        # Find all buffer points within the radius of the roundabout
        roundabout_nodes = []
        for point_ind in self.points.keys():
            if euclidean_distance(self.points[point_ind], self.points[node]) < self.roundabout_radius + 0.1 and point_ind != node:  # Find all nodes within a circle
                roundabout_nodes.append(point_ind)

        # Find all nodes that the buffer points are connected to
        inds_to_drop = []
        for ind in df_segments.index:
            start, end = df_segments.loc[ind, 'Definition']
            if start in roundabout_nodes and end in roundabout_nodes:  # A buffer connection for the roundabout
                inds_to_drop.append(ind)
            elif start in roundabout_nodes:  # Outgoing connection from the roundabout
                inds_to_drop.append(ind)
                df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [(node, end)], 'N_lanes': [1]})], ignore_index=True, axis=0)  # Add new connection
            elif end in roundabout_nodes:  # Incoming connection into the roundabout
                inds_to_drop.append(ind)
                df_segments = pd.concat([df_segments, pd.DataFrame({'Definition': [(start, node)], 'N_lanes': [1]})], ignore_index=True, axis=0)  # Add new connection

        # Iterate over the buffer points and remove them
        for point_ind in roundabout_nodes:
            self.points.pop(point_ind, None)

        # Assemble segments DataFrame
        df_segments.drop(inds_to_drop, axis=0, inplace=True)
        df_segments.reset_index(drop=True, inplace=True)
        self.segments = self.assembler.redo_config(df_segments=df_segments, points=self.points, add_reversed=False)  # Create the segment map and list segments

    def add_trafficlight(self, node: int) -> bool:
        """
        Converts any type of junction into a right-hand priority intersection
        Sets the parameter for the traffic light intersection in the representation matrix for the given node
        :param node: The index of the junction to be converted into right-hand
        :return: True: successful, False: unsuccessful
        """
        roads = self.segments['Definition']  # Get the roads' configuration from the assembler
        n_incoming_lanes = count_incoming_lanes(roads, self.points, node, unique=True)  # Count how many lanes are comingin the junction
        if n_incoming_lanes in self.trafficlight_inbound and self.matrix.loc[node, node] != JUNCTION_CODES['trafficlight']:  # Check for false conditions
            # Remove roundabout and its' buffer nodes if they exist
            if self.matrix.loc[node, node] == JUNCTION_CODES['roundabout']:
                self.remove_roundabout(node)

            self.matrix.loc[node, node] = JUNCTION_CODES['trafficlight']  # Set the node type to trafficlight in the matrix
            self.assembler.gen_signal_list(self.matrix)  # Re-generate the traffic signal list
            return True
        return False

    def get_matrices(self):
        """
        Return the assembled matrices. This method calls the method of the same name from the assembler class
        :return: The locations of the nodes (city junctions) as [x,y] coordinates; The matrix that contains the paths; The list of traffic signals to add to the simulation
        """
        return self.assembler.get_locs, self.assembler.get_vehicle_mtx, self.assembler.get_singal_list


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
    roads_mtx, vehicle_mtx, signals = r.get_matrices()
