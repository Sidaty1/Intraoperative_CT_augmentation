from numpy import random, linalg, \
            array, concatenate, expand_dims, append, dot, cross, \
            arccos, cos, sin, arctan2
from nibabel import load
from vedo import Text3D, Arrows, Cylinder, show
import tensorflow as tf 
import networkx as nx



def volgen(ct_no_contrast_intraop, vm_preop, vm_intraop, batch_size=1):
    """
    Volume Generator that takes:

    ct_no_contrast_intraop:  input of the network
    vm_preop: input of the network
    vm_intraop: ground truth 
    """

    while True:

        # generate random image indices
        indices_intraop = random.randint(len(ct_no_contrast_intraop), size=batch_size)
        indices_preop = random.randint(len(vm_preop), size=batch_size)

        # load volumes and concatenate: Without contrast
        imgs_no_contrast_intraop = [load_volfile(ct_no_contrast_intraop[i]) for i in indices_intraop]
        vols = [concatenate(expand_dims(imgs_no_contrast_intraop, axis=0), axis=0)]
        
        # load volumes and concatenate: Without contrast
        imgs_vm_preop = [load_volfile(vm_preop[i]) for i in indices_preop]
        vols.append(concatenate(expand_dims(imgs_vm_preop, axis=0), axis=0))

        imgs_vm_intraop = [load_volfile(vm_intraop[i]) for i in indices_intraop]
        vols.append(concatenate(expand_dims(imgs_vm_intraop, axis=0), axis=0))
        yield tuple(vols)


def unet_generator(ct_no_contrast_intraop, vm_preop, vm_intraop, downsize=1):
    """
        VoxeelMorph Generator
    """
    generator = volgen(ct_no_contrast_intraop, vm_preop=vm_preop, vm_intraop=vm_intraop)
    while True:
        imgs_no_contrast_intraop, imgs_vm_preop, imgs_vm_intraop  = next(generator)
        invols = [tf.concat((tf.expand_dims(imgs_vm_preop, axis=-1), tf.expand_dims(imgs_no_contrast_intraop, axis=-1)), axis=-1)]
        outvols = [imgs_vm_intraop]
        yield (invols, outvols)


def load_volfile(filename):
    img = load(filename)
    vol = img.get_data().squeeze()
    return vol

def read_data_files(number, patient, repo, tag):
    # patient is either baseline or deformed
    # tag is: with_contrast, without_contrast or vm
    list_data = []
    for i in range(number):
        zeros = '0'* (4 - len(str(i))) + str(i)
        path = repo + patient + zeros + '/deformed_' + tag + '.nii'
        list_data.append(path)
    return list_data


def get_nodes(csv_table):
    """
        get the centerlines nodes space coordinates position 
    """
    nodes = []
    for index, raw in csv_table.iterrows():
        start_point = get_start_point(raw)
        end_point = get_end_point(raw)
        if start_point not in nodes:
            nodes.append(start_point)
        if end_point not in nodes:
            nodes.append(end_point)
    return nodes

def get_edges(csv_table):
    """
        get the centerlines edge space coordinates position. Edge is defined as a couple of nodes
    """
    edges = []
    attrs = dict()
    nodes = get_nodes(csv_table)
    for index, raw in csv_table.iterrows():
        start_point_index = get_index_of_node(nodes, get_start_point(raw))
        end_point_index = get_index_of_node(nodes, get_end_point(raw))
        radius = get_radius(raw)
        length = get_length(raw)
        attrs[(start_point_index, end_point_index)] = {"Radius": radius, "Length": length, "teta": 0, "phi": 0}
        edges.append((start_point_index, end_point_index))
    return edges, attrs
        
def get_radius(raw):
    """
        get the edge radius
    """
    return raw["Radius"]

def get_length(raw):
    """
        get the edge length
    """
    return raw['Length']

def get_index_of_node(nodes, node):
    """
        get the index of the node
    """
    for i, nd in enumerate(nodes):
        if nd == node:
            return i

def get_start_point(raw):
    """
        get the edge first node 
    """
    return [raw['StartPointPosition_R'], raw["StartPointPosition_A"], raw['StartPointPosition_S']]



def get_end_point(raw):
    """
        get the edge second node 
    """
    return [raw['EndPointPosition_R'], raw["EndPointPosition_A"], raw['EndPointPosition_S']]


def get_frame_attrs(G, xyz, root_node):

    """
        get the frame attributes: coordinates, tangente, normal and binormal 
    """
    frame_attrs = dict()
    for node in G.nodes():
        parents = get_parents(G, node)
        childs = get_childs(G, node)

        if node == root_node or len(childs) == 0:
            frame_attrs[node] = {"xyz": xyz[node], "x": xyz[node], "y": xyz[node], "z": xyz[node]}           
        else: 

            parent = parents[0]
            random_child = random.randint(0, len(childs))
            random_child_coordinates = xyz[childs[random_child]]

            parent_coordinates = xyz[parent]
            node_coordinates = xyz[node]

            x = array(node_coordinates) - array(parent_coordinates)
            y = cross(array(x), array(random_child_coordinates) - array(node_coordinates))
            z = cross(x, y)

            x = 5*x / linalg.norm(x)
            y = 5*y / linalg.norm(y)
            z = 5*z / linalg.norm(z)
                        
            frame_attrs[node] = {"xyz": xyz[node], 
                                        "x": [xyz[node][0] + x[0], xyz[node][1] + x[1], xyz[node][2] + x[2]], 
                                        "y": [xyz[node][0] + y[0], xyz[node][1] + y[1], xyz[node][2] + y[2]], 
                                        "z": [xyz[node][0] + z[0], xyz[node][1] + z[1], xyz[node][2] + z[2]]}         

    return frame_attrs

def computeAngles(G, nodes):
    """
        Compute the angles at each biformation of the graph
    """
    tetas = dict()
    phis = dict()
    xyz = nodes
    x = nx.get_node_attributes(G, "x")
    y = nx.get_node_attributes(G, "y")
    z = nx.get_node_attributes(G, "z")
    for edge in G.edges():
        if 0 in edge: 
            tetas[(edge[0], edge[1])] = 0 
            phis[(edge[0], edge[1])] = 0 
        else: 
            xyz_node_global = xyz[edge[1]]
            x_frame = x[edge[0]] 
            y_frame = y[edge[0]]
            z_frame = z[edge[0]]
            teta, phi = getTeta_phi(xyz[edge[0]], x_frame, y_frame, z_frame, xyz[edge[1]])

            tetas[(edge[0], edge[1])] = teta 
            phis[(edge[0], edge[1])] = phi             

    return tetas, phis

def getTeta_phi(new_origin, x_frame, y_frame, z_frame, point):
    """
        Get rotation between global and local frames
    """
    teta = arccos((x_frame[0] - new_origin[0])/linalg.norm(array(x_frame) - array(new_origin)))
    phi = arccos((y_frame[1] - new_origin[1])/linalg.norm(array(y_frame) - array(new_origin)))
    psi = arccos((z_frame[2] - new_origin[2])/linalg.norm(array(z_frame) - array(new_origin)))

    R1 = array([[1, 0, 0],
                     [0, cos(teta), sin(teta)],
                     [0, -sin(teta), cos(teta)]])

    R2 = array([[cos(phi), 0, -sin(phi)],
                        [0, 1, 0],
                        [sin(phi), 0, cos(phi)]])

    R3 = array([[cos(psi), sin(psi), 0],
                        [-sin(psi), cos(psi), 0],
                        [0, 0, 1]])

    # Global rotation 
    R = R3*R2*R1

    # Translation 
    T = [new_origin[0], new_origin[1], new_origin[2] ]

    local_coordinates = R.dot(point)
    local_coordinates = local_coordinates/linalg.norm(local_coordinates)
    teta_spheric = arctan2(local_coordinates[1], local_coordinates[0])
    phi_spheric = arccos(local_coordinates[2])

    return teta_spheric, phi_spheric



def get_childs(G, node):
    """
        Get all childs of a node in the graph
    """
    childs = []
    for nd in G.successors(node):
        childs.append(nd)
    return childs

def get_parents(G, node):
    """
        Get all parents of a node in the graph
    """
    parents = []
    for nd in G.predecessors(node):
        parents.append(nd)
    return parents

def get_edge_childs(G, node):
    """
        Get all childs of an edge in the graph
    """
    childs = []
    child_iter= nx.dfs_edges(G, node)
    for child in child_iter:
        childs.append(child)
    return childs


def showGraph(G, nodes, label_colors):
    """
        show the graph using vedo
    """
    cyls = []
    texts = []
    x = nx.get_node_attributes(G, "x")
    y = nx.get_node_attributes(G, "y")
    z = nx.get_node_attributes(G, "z")
    xyz = nx.get_node_attributes(G, "xyz")
    for idx, edge in enumerate(G.edges()):
        #print(edge, int(label_colors[idx]))
        cyl = Cylinder(pos=[nodes[edge[0]], nodes[edge[1]]], r= 1).color(label_colors[edge]) # , label_colors[idx], label_colors[idx]))
        cyls.append(cyl)

        position_text = (array(xyz[edge[1]]) + array(xyz[edge[0]]))/2 
        position_text -= array([0, -1, 0])
        text = Text3D(str(int(label_colors[edge])), pos=position_text, s=2).c("red")

        texts.append(text)
    Arrows = []
    for node in G.nodes():
        text = Text3D(str(node), pos=(nodes[node][0]-1, nodes[node][1]-1, nodes[node][2]-1), s=3)
        texts.append(text)
        for frame in [x[node], y[node], z[node]]:
            Arrows.append(Arrows( [xyz[node]], [array(frame)]))

    show(cyls, texts, Arrows)


def get_root_node(G, attrs):
    """
        Compute the root branch of the vessel tree as the one with the highest radius
    """
    max_rad = 0
    my_edge = (0,0)
    for edge in G.edges():
        parents = get_parents(G, edge[0])
        childs = get_childs(G, edge[1])
        if len(parents) == 0 or len(childs) == 0:
            if attrs[edge]["Radius"] > max_rad:
                max_rad = attrs[edge]["Radius"]
                my_edge = edge
    if len(get_parents(G, my_edge[0])) != 0 and len(get_childs(G, my_edge[0])) != 0:
        return my_edge[1]
    else:
        return my_edge[0]

def get_horton(G, root_node):

    """
        Get the horton index of a branch/edge
    """
    paths_from_root = dict()
    horton = dict()

    for edge in G.edges():
        node = edge[1]
        if node != root_node:
            paths_from_root[edge] = len(nx.shortest_path(G, source=root_node, target=node)) - 1
        else: 
            paths_from_root[edge] = 0
    paths_from_root = {k: v for k, v in sorted(paths_from_root.items(), key=lambda item: item[1], reverse=True)}

    for edge in paths_from_root:
        if len(get_childs(G, edge[1])) == 0:
            horton[edge] = 1
        else:
            diffents = False
            childs = get_childs(G, edge[1])
            first_child = childs[0]
            for child in childs:
                if horton[(edge[1], child)] != horton[(edge[1], first_child)]:
                    diffents = True
                    break
            if not diffents:
                horton[edge] = horton[(edge[1], first_child)] + 1
            else:
                horton_child_values = []
                for child in childs:
                    horton_child_values.append(horton[(edge[1], child)])
                horton[edge] = max(horton_child_values)

    return horton

def get_points_in_edge(node0_position, node1_position, longueur):
    """
        Sample an edge
    """
    pente = array(node1_position) - array(node0_position)
    points = []
    for i in range(0, int(2*longueur + 1)):
        t = i/(int(2*longueur))
        points.append(pente * t + node0_position)

    return points

def point_to_voxels(img, point, rad, growth):
    """
        From the Euclidean to the Image space
    """
    voxels = []
    affine = img.affine
    affine = linalg.inv(affine)
    point = append(point, 1)
    voxel = dot(affine, point)
    voxel = voxel[:-1]

    rad = rad + rad*growth
    rad = array([rad for i in range(3)])

    voxel_size = img.header.get_zooms()
    voxels_rad = rad / voxel_size
    voxels_rad = voxels_rad.astype('int')
    for i in range(-voxels_rad[0], + voxels_rad[0] + 1):
        for j in range(-voxels_rad[1], voxels_rad[1] + 1):
            for k in range(-voxels_rad[2], voxels_rad[2] + 1):
                new_voxel = voxel + array([i, j, k])
                voxels.append(new_voxel)

    return voxels