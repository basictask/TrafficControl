#%% Importing libraries

import pandas as pd
import random as rnd
import numpy as np
import re

#%% Class definition

class reader:
    def __init__(self, filepath, entry_points, vrate, pathnum, path_dist):
        # Error checking
        if(path_dist not in ['normal','uniform']):
            raise Exception('Invalid parameter for path distribution')
        
        # Params
        self.filepath = filepath
        self.vrate = vrate
        self.entry_points = entry_points
        self.pathnum = pathnum
        self.path_dist = path_dist
        
        # Set up inner params
        self.read()                 # Read the file
        self.gen_road_mtx()         # Create a matrix of all the roads (locs)
        self.gen_path_graph()       # Construct a graph of all nodes
        self.gen_paths()            # Find possible paths in the graph
        self.gen_vehicle_mtx()      # Create the matrix of the paths
                
    def read(self):
        # Read the file
        df = pd.read_html(self.filepath)[0]
        df.set_index('Name', drop=True, inplace=True)
        df.sort_values(by=['Name'], inplace=True)
        ind = ['Point' in x for x in list(df.index)]
        
        # Separate Points and segments
        df_points = df.loc[ind].dropna(axis=1) # Name, (x, y)
        df_segments = df.loc[[not x for x in ind]].reset_index().drop('Name', axis=1)
        
        # Process points
        df_points.index = [x.split(' ')[1] for x in list(df_points.index)] # Get only letter
        df_points['Value'] = [re.findall(r'[\d\.]+', x) for x in df_points['Value']] # Find coordinates
        df_points['Value'] = [[float(x) for x in l] for l in df_points['Value']] # Convert coords to float
        df_points['Value'] = [tuple(x) for x in df_points['Value']]
        locations = {x:y for x,y in zip(list(df_points.index), df_points['Value'])}
        
        # Process segments
        df_segments['Definition'] = [re.findall(r'(?<=\()[^)]*(?=\))', x)[0] for x in df_segments['Definition']]
        df_segments['Definition'] = [x.split(' ') for x in df_segments['Definition']]
        for i in range(len(df_segments)):
            df_segments['Definition'][i][0] = df_segments['Definition'][i][0][:-1] 
        df_segments['Definition'] = [tuple(x) for x in df_segments['Definition']]
        df_segments.drop('Value', axis=1, inplace=True)
        
        # Assemble into segment map
        forward_segm = list(df_segments['Definition'])
        reverse_segm = [(x[1], x[0]) for x in df_segments['Definition']]
        forward_segm.extend(reverse_segm)
        
        df_segments = pd.DataFrame({'Definition':forward_segm}, index=range(len(forward_segm)))
        df_segments.sort_values('Definition', ignore_index=True)
        segment_map = {x:y for x,y in zip(df_segments['Definition'], list(df_segments.index))}
        
        self.segment_map = segment_map
        self.points = locations
        self.segments = df_segments
        
    def gen_road_mtx(self):
        # Creates the matrix that is used to generate the roads
        lst = []
        for x in self.segments['Definition']:
            lst.append(tuple([self.points[x[0]], self.points[x[1]]]))
        self.locs = lst
    
    def gen_path_graph(self):
        # Create a graph of all the nodes that can be reached from a specific node
        graph = {x:[] for x in list(self.points.keys())}
        for x in self.segments['Definition']:
            graph[x[0]].append(x[1])
            #graph[x[1]].append(x[0]) # Only needed for explicit adding
            
        self.graph = graph
    
    def DFS(self, G, v, seen=None, path=None):
        # Recursive depth seearch in a graph to determine paths between nodes
        if seen is None: seen = []
        if path is None: path = [v]

        seen.append(v)

        paths = []
        for t in G[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(self.DFS(G, t, seen[:], t_path))
        
        return paths

    def drop_empty_keys(self, dct):
        for v in list(dct.keys()):
            if(len(dct[v]) == 0):
                dct.pop(v, None)
        return dct
    
    def gen_paths(self):
        # Generates all the paths between entry points
        paths = {x:[] for x in list(self.points.keys())}
        
        for v in self.entry_points: # Iterate over all the entry points
            v_paths = self.DFS(self.graph, v) # Find all the paths from an entry point
            for path in v_paths: # Iterate over all the paths
                if(path[len(path)-1] in self.entry_points and path[len(path)-1] != v): # If the last node is a valid entry point
                    paths[v].append(path) # Append to the path
        
        paths = self.drop_empty_keys(paths)

        path_codes = {x:[] for x in list(self.points.keys())}        
        paths_dual = {x:[] for x in list(self.points.keys())}
        for v in paths.keys():
            dualnode = []
            c_dualnode = []
            for p in paths[v]:
                dualpath = []
                c_dualpath = []
                for i in range(1, len(p)):
                    mark = (p[i-1], p[i])
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
            for l in lst:
                path_stack.append(l)
        
        if(self.pathnum > len(path_stack)):
            raise Exception('The number of paths input is larger than the number of all paths.')
        
        self.path_stack = path_stack
        self.path_codes = path_codes            
        self.paths_dual = paths_dual
        self.paths = paths
        
    def gen_vehicle_mtx(self):
        # Creates the dict that is used to create the vehicle generators
        gen = {'vehicle_rate': self.vrate}
        
        vehicles = []
        for i in range(self.pathnum):
            # Calculate the weight of the path
            weight = 0
            if(self.path_dist == 'normal'):
                weight = round(np.random.randn(), 2)
            elif(self.path_dist == 'uniform'):
                weight = 1 / self.pathnum
            
            # Pick a path from the path stack randomly
            path = self.path_stack[np.random.randint(0, len(self.path_stack))]
            vehicles.append([weight, {'path': path}])
           
        gen['vehicles'] = vehicles
        self.vehicle_mtx = gen
        
    def get_matrices(self):
        # Return the assembled matrices
        return [self.locs, self.vehicle_mtx]
        
#%% If script is run

if(__name__ == '__main__'):
    filename        = 'simple_color.html'   # A .html GeoGebra construction protocol
    entry_points    = ['A','C','E','G']     # Points that are valid for entering the traffic system
    vrate           = 60                    # Rate of vehicles coming in from each entry point
    paths_to_gen    = 6                     # How many paths to generate
    path_dist    = 'normal'              # One of 'normal', 'uniform' 
    
    r = reader(filename, entry_points, vrate, paths_to_gen, path_dist)
    roads, vehicle_mtx = r.get_matrices()
    