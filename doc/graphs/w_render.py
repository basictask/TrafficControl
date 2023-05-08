"""
This is a small script to render all dot files in this folder into png files
Input: None
Output: .png files from each dot file
"""

import os
import pydot

target_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(target_dir)

for file in os.listdir(target_dir):
    if(file.endswith('.dot')):
        print("Rendering: " + file + "... ", end = "") 
        (graph, ) = pydot.graph_from_dot_file(file)
        name = file.split('.')[0]
        graph.write_png(name + '.png')
        print("Success")

print("Done")
