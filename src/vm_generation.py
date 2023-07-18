
from math import isnan
from numpy import arange, zeros
from pandas import read_csv
from nibabel import Nifti1Image, load, save
from tqdm import tqdm
from utils import *

import networkx as nx



class VM:
    """
        Vessel Map generation given vessel tree centerlines computed with the slicer module


        :csv_table_path: The path of the centerlines csv table, generated with slicer.
        :without_contrast_volume: The path of the non contrasted CT.
        :output_directory: Directory where the vessel map should be sotred.
        :grouwth: The growth factor of the vessel map, see the paper for more details.
    """ 
    def __init__(self, csv_table_path, without_contrast_volume, output_directory,  growth=0.2) -> None:
        self.csv_table = read_csv(csv_table_path, index_col=False)
        self.without_contrast_volume = without_contrast_volume
        self.output_directory = output_directory
        self.growth = growth


    def __call__(self):
        
        # Get The nodes, edges and edge attributes
        nodes = get_nodes(self.csv_table)
        edges, attrs = get_edges(self.csv_table)

        # Creating an arbitrary oriented graph
        DiG = nx.DiGraph()
        DiG.add_nodes_from(arange(len(nodes)))
        DiG.add_edges_from(edges)

        # Computing the root node
        root_node = get_root_node(DiG, attrs)

        # Computing the rezl oriented graph based on the root node 
        G = nx.Graph()
        G.add_nodes_from(arange(len(nodes)))
        G.add_edges_from(edges)
        G = nx.bfs_tree(G, root_node)

        # Updating edges and its attributes
        new_attrs = dict()
        for key in attrs:
            if G.has_edge(key[0], key[1]):
                new_attrs[key] = attrs[key]
            elif G.has_edge(key[1], key[0]):
                new_attrs[(key[1], key[0])] = attrs[key]
        
        # Getting nodes attributes, position xyz of the node
        nodes_attrs = get_frame_attrs(G, nodes, root_node)
        nx.set_node_attributes(G, nodes_attrs)

        # Computing angles at each node 
        tetas, phis = computeAngles(G, nodes)

        # Updating edges attributes by adding the spherical coordinates of the edge in the local frame
        for key in new_attrs:
            new_attrs[key]["teta"] = tetas[key]
            new_attrs[key]["phi"] = phis[key]
            if isnan(new_attrs[key]["Radius"]):
                parent = get_parents(G, key[0])[0]
                new_attrs[key]["Radius"] = new_attrs[(parent, key[0])]["Radius"]
                print("New radius: ", new_attrs[key]["Radius"])

        
        nx.set_edge_attributes(G, new_attrs)

        # Here we should put the non contrasted image 
        img = load(self.without_contrast_volume)
        img_data = img.get_fdata()
        vm = zeros(shape=img.shape)

        xyz = nx.get_node_attributes(G, "xyz")
        longueurs = nx.get_edge_attributes(G, 'Length')
        radius = nx.get_edge_attributes(G, "Radius")
        for edge in tqdm(G.edges()):
            node0_position = xyz[edge[0]]
            node1_position = xyz[edge[1]]
            points = get_points_in_edge(node0_position, node1_position, longueurs[edge])
            for point in points:
                voxels = point_to_voxels(img, point, radius[edge], self.growth)
                for voxel in voxels: 
                    if voxel[0] < 256 and voxel[1] < 256 and voxel[2] < 256:
                        
                        vm[int(voxel[0]), int(voxel[1]), int(voxel[2])] = img_data[int(voxel[0]),            
                                                                                        int(voxel[1]), 
                                                                                        int(voxel[2])]
    
        niff_stc = Nifti1Image(vm, img.affine)
        save(niff_stc, self.output_directory + "/generated_vm.nii")



if __name__ == "__main__": 
    VM(csv_table_path="./path/to/centerlines/csv/file.csv", 
                        without_contrast_volume='path/to/the/volume.nii', 
                        output_directory='./path/to/store/the_vessel_map.nii', 
                        growth=0.2)