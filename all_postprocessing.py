import copy
import json
import os
import pickle
import statistics
import time

import cv2
import geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sknw
from PIL import Image
from osgeo import gdal, ogr, osr
from pygeoif import LineString
from scipy import ndimage
from scipy.ndimage import binary_dilation
from shapely import length
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon
from simplification.cutil import simplify_coords
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, skeletonize
from tqdm import tqdm
from utils import flatten

"""
    function adapted from selim_sef's solutiom to spacenet challenge 3
    used for postprocessing stitched images and creating the spacenet challenge 3 submission file
    https://github.com/SpaceNetChallenge/RoadDetector/blob/0a7391f546ab20c873dc6744920deef22c21ef3e/selim_sef-solution/tools/vectorize.py
"""


def to_line_strings(mask, sigma=0.5, threashold=0.3, small_obj_size1=300, dilation=25, return_ske=False):
    """
        function adapted from selim_sef's solutiom to spacenet challenge 3
        used for postprocessing stitched images and creating the spacenet challenge 3 submission file
        https://github.com/SpaceNetChallenge/RoadDetector/blob/0a7391f546ab20c873dc6744920deef22c21ef3e/selim_sef-solution/tools/vectorize.py
    """

    mask = gaussian(mask, sigma=sigma)
    mask = copy.deepcopy(mask)

    mask[mask < threashold] = 0
    mask[mask >= threashold] = 1
    mask = np.array(mask, dtype="uint8")
    mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    mask, _ = ndimage.label(mask)
    mask = remove_small_objects(mask, small_obj_size1)
    mask[mask > 0] = 1
    # ret_image = mask.copy()
    # ret_image[ret_image>0] = 255
    # cv2.imwrite('img_postproc_wo_dilation.png', ret_image)

    base = np.zeros((1300, 1300))
    ske = np.array(skeletonize(mask), dtype="uint8")

    base += mask[8:1308, 8:1308]

    if return_ske:
        ske[ske > 0] = 255
        return np.uint8(ske)

    graph = sknw.build_sknw(ske, multi=True)

    line_strings = []
    lines = []
    all_coords = []
    nodes = graph.nodes()
    # draw edges by pts
    for (s, e) in graph.edges():
        for k in range(len(graph[s][e])):
            ps = graph[s][e][k]['pts']
            coords = []
            start = (int(nodes[s]['o'][1]), int(nodes[s]['o'][0]))
            all_points = set()

            for i in range(1, len(ps)):
                pt1 = (int(ps[i - 1][1]), int(ps[i - 1][0]))
                pt2 = (int(ps[i][1]), int(ps[i][0]))
                if pt1 not in all_points and pt2 not in all_points:
                    coords.append(pt1)
                    all_points.add(pt1)
                    coords.append(pt2)
                    all_points.add(pt2)
            end = (int(nodes[e]['o'][1]), int(nodes[e]['o'][0]))

            same_order = True
            if len(coords) > 1:
                same_order = np.math.hypot(start[0] - coords[0][0], start[1] - coords[0][1]) <= np.math.hypot(
                    end[0] - coords[0][0], end[1] - coords[0][1])
            if same_order:
                coords.insert(0, start)
                coords.append(end)
            else:
                coords.insert(0, end)
                coords.append(start)
            coords = simplify_coords(coords, 2.0)
            # print(coords)
            all_coords.append(coords)

    # print(all_coords)
    for coords in all_coords:
        if len(coords) > 0:
            line_obj = LineString(coords)
            lines.append(line_obj)
            line_string_wkt = line_obj.wkt
            line_strings.append(line_string_wkt)
    new_lines = remove_duplicates(lines)
    new_lines = filter_lines(new_lines, calculate_node_count(new_lines))
    line_strings = [l.wkt for l in new_lines]
    lengths = [length(l) for l in new_lines]

    # return skeleton too
    ske[ske > 0] = 255

    return line_strings, lengths, np.uint8(ske), graph  #, buffered_arr #ret_mask[8:1308, 8:1308]


def remove_duplicates(lines):
    all_paths = set()
    new_lines = []
    for l, line in enumerate(lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            if (pt1, pt2) not in all_paths and (pt2, pt1) not in all_paths and not pt1 == pt2:
                new_lines.append(LineString((pt1, pt2)))
                all_paths.add((pt1, pt2))
                all_paths.add((pt2, pt1))
    return new_lines


def filter_lines(new_lines, node_count):
    filtered_lines = []
    for line in new_lines:
        points = line.coords
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))

        length = np.math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

        if not ((node_count[pt1] == 1 and node_count[pt2] > 2 or node_count[pt2] == 1 and node_count[
            pt1] > 2) and length < 10):
            filtered_lines.append(line)
    return filtered_lines


def calculate_node_count(new_lines):
    node_count = {}
    for l, line in enumerate(new_lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            pt1c = node_count.get(pt1, 0)
            pt1c += 1
            node_count[pt1] = pt1c
            pt2c = node_count.get(pt2, 0)
            pt2c += 1
            node_count[pt2] = pt2c
    return node_count


def split_line(line):
    all_lines = []
    points = line.coords
    pt1 = (int(points[0][0]), int(points[0][1]))
    pt2 = (int(points[1][0]), int(points[1][1]))
    dist = np.math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
    if dist > 10:
        new_lines = cut(line, 5)
        for l in new_lines:
            for sl in split_line(l):
                all_lines.append(sl)
    else:
        all_lines.append(line)
    return all_lines


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # This is taken from shapely manual
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def skeletonize_segmentations(image_path, save_submissions, save_graph, save_skeleton, save_mask, plot=False, single=False):
    """
    receives: data paths, save paths, booleans for plotting
    returns: True, saves submission file, graph file, skeletonised result file
    Applies post-processing steps to a segmentation result
    """

    iterating = os.listdir(image_path)
    #all_files_done = os.listdir(save_graph)
    skel1 = time.time()
    for image in tqdm(iterating):
        # if image != 'AOI_4_Shanghai_PS-MS_img1185_00_00.png':
        #     continue
        # isolate image name
        image_name = image[:-10]
        img = np.asarray(Image.open(f'{image_path}/{image}'))

        # convert to skeleton and graph, apply morphologicals
        linestrings, lens, final_img, final_graph = to_line_strings(mask=img, sigma=0.5, threashold=0.3, small_obj_size1=600, dilation=4, return_ske=False)  # sigma=0.5 small_obj_size=350

        # apply graph postprocessing
        final_graph = graph_postprocessing(final_graph, final_img, plot=plot)

        # save SpaceNet submission csv
        with open(f'{save_submissions}/{image_name}.csv', 'w') as file:
            file.write('ImageId,WKT_Pix,length_m,travel_time_s\n')
            for line, leng in zip(linestrings, lens):
                file.write(f'{image.split("_")[4]},"{line}",{leng},{leng/13.66}\n')

        # pickle graph
        pickle.dump(final_graph, open(f'{save_graph}/{image_name}.pickle', 'wb'))

        # save skeleton
        cv2.imwrite(f'{save_skeleton}/{image_name}.png', final_img)

        # save postprocessed image
        #cv2.imwrite(f'{save_mask}/{image[:-10]}.png', final_mask)

        if single:
            break
    skel2 = time.time()
    print(f'finished postprocessing in {round(skel2 - skel1, 2)}s')
    return


def skeltonize_masks(image_path, save_path, plot=False):
    """
        receives: data paths, save paths
        returns: True
        saves mask files as pickles
    """

    for image in tqdm(os.listdir(image_path)):
        # cerate base array to compensate graph construction in inflated array fro image operations
        base = np.zeros((1332, 1332))
        img = np.asarray(Image.open(f'{image_path}/{image}'))
        base[16:1316, 16:1316] += img
        # skeltonize and create graph
        ske = np.array(skeletonize(img), dtype="uint8") # base
        #ske[ske > 0] = 1
        graph = sknw.build_sknw(ske, multi=True)
        # pickle graph and rename mask name
        pickle.dump(graph, open(f'{save_path}/{os.path.splitext(image)[0]}.pickle', 'wb'))
    return True


def compare_GED_graphs(gp_graphs_path, gt_graphs_path, take_first_result, max_time, out_path):
    """
        receives: path to ground truth and proposal graphs, booleans for evaluation, saving path for result
        returns: True
        Calculates the GED for each graph in a directory.
    """

    t_OGED1 = time.time()
    all_results = {}
    # iterate over all graphs
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name # + '.pickle' # gp_graph_name.replace('_00_00', '')
        # print(f'{gt_graphs_path}/{gt_graph_name}')
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            # access graphs
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))

            # calculate first GED iteration
            iterations = 0
            for v in nx.optimize_graph_edit_distance(gp_graph, gt_graph):
                min_result = v
                iterations += 1
                if take_first_result:
                    break
        else:
            pass

        # save results
        all_results[gt_graph_name] = min_result #GED #min_result

    # determine mean and save to file
    print(sorted(all_results.values()))
    mean_GED = statistics.mean(all_results.values())
    with open(f'{out_path}/NEW_GED{str(round(mean_GED, 2)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_results, f)
    t_OGED2 = time.time()
    print(f'mean Graph Edit Distance (GD): {round(mean_GED, 2)} in {round(t_OGED2 - t_OGED1, 2)}s')
    return True


def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
        receives: graph, booleans to determine conversion
        returns: list of geodataframe of nodes
        Converts graph nodes to list of geodataframes.
        Adopted from OSMNX_funcs.
    """

    # code adapted and changed from https://github.com/gboeing/osmnx/blob/master/osmnx/save_load.py
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """
    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []
    if nodes:
        if len(G.nodes(data=True)) > 0:
            # access nodes and convert
            nodes, data = zip(*G.nodes(data=True))
            gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

            # extract geometry
            if node_geometry:
                gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['o'][0], row['o'][1]), axis=1)

            # appl crs change here
            #gdf_nodes.crs = G.graph['crs']

            to_return.append(gdf_nodes)
        else:
            print('no nodes detected')

    # not used
    if edges:
        start_time = time.time()

        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):
            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u': u, 'v': v, 'key': key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]

            # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
                    point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
                    edge_details['geometry'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        #gdf_edges.crs = G.graph['crs']
        #gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)
        # print('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]


def clean_intersections(G, tolerance=15, dead_ends=False):
    """
        receives: graph, tolerance value for matching, boolean if dead ends should be considered
        returns: intersected centroids
        Reduces clustered nodes by a buffer operation.
        Adopted from OSMNX_funcs.
    """

    """
    Clean-up intersections comprising clusters of nodes by merging them and
    returning their centroids.
    Divided roads are represented by separate centerline edges. The intersection
    of two divided roads thus creates 4 nodes, representing where each edge
    intersects a perpendicular edge. These 4 nodes represent a single
    intersection in the real world. This function cleans them up by buffering
    their points to an arbitrary distance, merging overlapping buffers, and
    taking their centroid. For best results, the tolerance argument should be
    adjusted to approximately match street design standards in the specific
    street network.
    Parameters
    ----------
    G : networkx multidigraph
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single intersection
    dead_ends : bool
        if False, discard dead-end nodes to return only street-intersection
        points
    Returns
    ----------
    intersection_centroids : geopandas.GeoSeries
        a GeoSeries of shapely Points representing the centroids of street
        intersections
    """

    # if dead_ends is False, discard dead-end nodes to only work with edge
    # intersections
    if not dead_ends:
        if 'streets_per_node' in G.graph:
            streets_per_node = G.graph['streets_per_node']
        else:
            streets_per_node = 1    # count_streets_per_node(G)

        dead_end_nodes = [node for node, count in streets_per_node.items() if count <= 1]
        G = G.copy()
        G.remove_nodes_from(dead_end_nodes)

    # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
    # overlaps
    gdf_nodes = graph_to_gdfs(G, edges=False)
    #print(gdf_nodes)
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make it iterable so we can turn it
        # int a GeoSeries
        buffered_nodes_list = [buffered_nodes]
    else:
        # get the centroids of the merged intersection polygons
        buffered_nodes_list = [polygon for polygon in buffered_nodes.geoms]

    # index mapping implemented by myself
    last_index = len(G.nodes)
    mappings = {k+last_index: [] for k in range(len(buffered_nodes_list))}

    # create dictionary mapping polygons to contained points
    for index, polygon in enumerate(buffered_nodes_list):
        for i, row in gdf_nodes.iterrows():
            if polygon.contains(row['geometry']):
                mappings[last_index+index].append(i)

    #print(mappings)

    unified_intersections = gpd.GeoSeries(buffered_nodes_list)
    intersection_centroids = unified_intersections.centroid
    return intersection_centroids, mappings


def extend_edge_to_node(graph_, edge, tolerance=45):
    """
        receives: graph, edge to extend
        returns: extended graph
        Extends to coordinates start and end of an edge to reach the assiged node, for dipslaying purposes.
    """

    # Get the coordinates of the start and end nodes of the edge
    graph = graph_.copy()
    u, v = edge
    pts = graph.edges[u, v, 0]['pts']
    first_coords = np.array(pts[0, :])
    second_coords = np.array(pts[-1, :])#np.array((graph.nodes[v]['o'][0], graph.nodes[v]['o'][1]))

    # Get the coordinates of the node to extend the edge to
    u_node_coords = np.array([graph.nodes[u]['o'][0], graph.nodes[u]['o'][1]]).reshape((1, 2))
    v_node_coords = np.array([graph.nodes[v]['o'][0], graph.nodes[v]['o'][1]]).reshape((1, 2))

    # match the nodes to the starting points of the edge
    if np.linalg.norm(abs(first_coords - u_node_coords)) < np.linalg.norm(abs(first_coords - v_node_coords)):
        # instert u node cords at beginnging an v node cords at end
        if u_node_coords not in pts:
            pts = np.vstack((u_node_coords, pts))

        if v_node_coords not in pts:
            pts = np.vstack((pts, v_node_coords))

    elif np.linalg.norm(abs(first_coords - v_node_coords)) < np.linalg.norm(abs(first_coords - u_node_coords)):
        # instert u node cords at end an v node cords at beginning
        if v_node_coords not in pts:
            pts = np.vstack((v_node_coords, pts))

        if u_node_coords not in pts:
            pts = np.vstack((pts, u_node_coords))

    """# Calculate the distance between the start and end nodes of the edge
    #edge_length = nx.shortest_path_length(graph, u, v, 'weight')
    # Calculate the distance between the start node of the edge and the node to extend to
    #node_distance = nx.shortest_path_length(graph, u, node, 'weight')
    # Calculate the ratio of the distance between the start node and the node to extend to
    # to the distance between the start and end nodes of the edge
    # handle length of 0 for edge cae graphs
    #if edge_length == 0:
    #    return graph
    #ratio = node_distance / edge_length

    # Insert the coordinates of the node into the pts array of the edge"""
    """if u == node:
        pts = np.vstack((node_coords, pts))
    elif v == node:
        pts = np.vstack((pts, node_coords))"""
    """
    else:
        new_pt_coords = np.array([u_coords[0] + ratio * (v_coords[0] - u_coords[0]),
                         u_coords[1] + ratio * (v_coords[1] - u_coords[1])]).reshape((1, 2))
        pts = np.vstack((pts, new_pt_coords))
        pts = np.vstack((pts, node_coords))"""

    # Update the edge attributes in the graph
    graph.edges[u, v, 0]['pts'] = pts
    return graph


def graph_postprocessing(G, img, plot):
    """
        receives: Graph, corresponding image, boolean if result should be plotted
        returns: postprocessed graph
        applies postprocessing procedures: intersection reduction, node contraction, node-edge connection
    """

    # remove small roundabaouts
    last_node = len(G.nodes)
    if last_node < 1:
        print('no nodes detected')
        return G
    gdf, mappings = clean_intersections(G, dead_ends=True)
    all_new_nodes = []

    for i, point in gdf.items():
        new_node = [point.x, point.y]
        all_new_nodes.append((last_node+i, {'pts': np.array([new_node], dtype='int16'), 'o': np.array(new_node)}))

    # add new nodes
    G.add_nodes_from(all_new_nodes)

    G_new = G.copy()
    for new_node, old_nodes in mappings.items():
        for old_node in old_nodes:
            G_new = nx.contracted_nodes(G_new, new_node, old_node, self_loops=False, copy=True)

    # problem: edges are in theory connected, but pixels to nodes are not drawn
    # remove edges with length 0
    #G_new.remove_edges_from([(u, v) for u, v, attr in G_new.edges(data=True) if attr['weight'] <= 30])

    G_new_2 = G_new.copy()
    for u, v in G_new_2.edges():
        if u is not v:
            G_new_2 = extend_edge_to_node(G_new_2, (u, v))

    # Graph simplification overcomplicated everything
    """
    G_new_2simpl = simplify_graph(G_new_2)
    # remove duplicate edges
    unique_edges = set()
    non_unique = list()
    for u, v, attr in G_new_2simpl.edges(data=True):
        if (u, v) not in unique_edges:
            unique_edges.add((u, v))
        else:
            non_unique.append((u, v))

    G_new_2simpl_rem = G_new_2simpl.copy()
    for u, v, attr in G_new_2simpl.edges(data=True):
        if (u, v) in non_unique and isinstance(attr['pts'], list):
            G_new_2simpl_rem.remove_edge(u, v)
            non_unique.remove((u, v))
    for u, v, attr in G_new_2simpl_rem.edges(data=True):
        if isinstance(attr['pts'], list):
            conc = np.vstack((attr['pts'][0], attr['pts'][1]))
            attr['pts'] = conc"""

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for (s, e) in G.edges():
            vals = flatten([[v] for v in G[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[0].plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in G_new.edges():
            vals = flatten([[v] for v in G_new[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[1].plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in G_new_2.edges():
            vals = flatten([[v] for v in G_new_2[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax[2].plot(ps[:, 0], ps[:, 1], 'blue')
        ps_ = np.array([i[1]['o'] for i in G.nodes(data=True)])
        ax[0].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in G_new.nodes(data=True)])
        ax[1].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in G_new_2.nodes(data=True)])
        ax[2].plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)

        ax[0].set_title(f'Graph before postprocessing')
        ax[1].set_title(f'Graph after node contraction')
        ax[2].set_title(f'Graph after edge extension')

        plt.savefig(f'{fp}/graphics/postprocessing/before_after_closeup.png')
        plt.show()

    return G_new_2 #G_new_2simpl_rem


def determine_overlap(img_size, wish_size):
    """
        receives: image size to split, size image is split into
        returns: list of tuples describing the indices to split an image along
        calculates indices on whichan image has to be split
    """
    num_pics = int(np.ceil(img_size/wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i*(wish_size-applied_step), (i+1)*wish_size - i*applied_step) for i in range(num_pics)]

    return overlap_indices


def stitch_overlap_images(sorted_images, result_path, overlap_params, old_segmentation_result, for_visual_output):
    """
        receives: dict of sorted image names, path where to save resuls, dict with parameters describing overlpping,
            boolean for color allocation, boolean if result needs to be inspected visually
        returns: Tue, results are saved
        stitches images back together, after being plit for segmentation
    """

    # images need to be binary
    if old_segmentation_result:
        street_color = 207
        background_color = 20
    else:
        street_color = 207
        background_color = 0
    if for_visual_output:
        output_color = 255
    else:
        output_color = 1

    for k, img_paths in tqdm(sorted_images.items()):
        # change base to 1301 size and crop later to allow overlap division problem
        base = np.zeros((1301, 1301))
        arrays = dict()
        base_name = img_paths[0].split('/')[-1]

        for img in img_paths:
            names = img.replace('.png', '').split('/')[-1].split('_')[-2:]
            ids = (int(names[0]), int(names[1]))
            image = Image.open(img)
            # rescale image to fit overlap parameter
            if image.size[0] != overlap_params[0][1]:
                image = image.resize((overlap_params[0][1], overlap_params[0][1]))

            # convert to array
            image_open = np.asarray(image)
            # check if image is single channel, if not, convert to single channel
            if len(image_open.shape) > 2 and image_open.shape[2] > 1:
                image_open = image_open[:, :, 0]

            # rescale color values to binary
            ret_image = image_open.copy()
            ret_image[ret_image < street_color] = 0
            ret_image[ret_image >= street_color] = 1
            arrays[ids] = ret_image

        # place arrays in big array
        it = 0
        for i, i_val in enumerate(overlap_params):
            for j, j_val in enumerate(overlap_params):
                if (i, j) in arrays.keys():
                    img = arrays[i, j]
                    base[i_val[0]:i_val[1], j_val[0]:j_val[1]] += img
                it += 1
        base[base > 0] = output_color # 1: for binary, 255: for visual

        # save files
        cv2.imwrite(f'{result_path}/{base_name}', base[:1300, :1300])

    return True


def sort_images(base_path):
    """
        receives: a path to results images
        returns: dict with images sorted back together
        sorts all files, to allocate split files back together
    """

    ret_dict = dict()
    # retrieve the image number from a string like this: 'AOI_2_Vegas_PS-RGB_img1_00_00.png' -> img1
    for name in tqdm(os.listdir(base_path)):
        ins_name = f'{name.split("_")[2]}_{name.split("_")[4]}'
        if ins_name in ret_dict.keys():
            ret_dict[ins_name].append(base_path+name)
        elif ins_name not in ret_dict.keys():
            ret_dict[ins_name] = [base_path+name]

    return ret_dict


def getGeom(inputRaster, sourceSR='', geomTransform='', targetSR=''):
    """
        receives: input image, source spatial reference, transformation parameter, target spatial reference
        returns: the inputs geometry
        copied from the OSM library
    """
    # from osmnx
    if targetSR == '':
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == '':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    # geom = ogr.Geometry(ogr.wkbPoint)
    return geomTransform


def pixelToGeoCoord(xPix, yPix, geomTransform):
    """
        receives: xpixel, ypixel, geometry of the image
        retruns: transformed tuple of coordinates
        copied from the APLS metrics script
    """
    # If you want to gauruntee lon lat output, specify TargetSR  otherwise, geocoords will be in image geo reference
    # targetSR = osr.SpatialReference()
    # targetSR.ImportFromEPSG(4326)
    # Transform can be performed at the polygon level instead of pixel level

    """if targetSR == '':
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == '':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())
    """

    # extract geometry
    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    # apply coordinate transformation
    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    """if performReprojection:
        if sourceSR == '':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)"""

    return (geom.GetX(), geom.GetY())


def plot_graph(G_p):
    """
        receives: graph
        retruns: None
        plots a graph
    """

    # draw edges by pts
    for (s, e) in G_p.edges():
        vals = flatten([[v] for v in G_p[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    nodes = G_p.nodes(data=True)
    ps = np.array([i[1]['o'] for i in nodes])

    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    plt.title('Build Graph')
    plt.show()


def convert_graph_to_geojson(G_g):
    """
        receives: graph
        retruns: point features of graph, line features of graph
        converts a graphs nodes and edges into a geojson
    """

    point_features, linestring_features = [], []

    for node in G_g.nodes(data=True):
        point = geojson.Point((node[1]['coords'][0], node[1]['coords'][1]))
        feature = geojson.Feature(geometry=point, properties={'id': node[0]})
        point_features.append(feature)

    for start, stop, attr_dict in G_g.edges(data=True):
        coords = attr_dict['coords']
        line = geojson.LineString(coords)

        feature = geojson.Feature(geometry=line, properties={})
        linestring_features.append(feature)

    return point_features, linestring_features


def convert_all_graphs_to_geojson(graph_path, RGB_image_path, MS_image_path, out_path, ms_bool):
    """
        receives: path to graphs, path to RGB images, path to MS images, saving path, boolean if input is MS or RGB
        retruns: True, saves geojsons
        converrt all graphs into georeferenced geojsons and save 3 files per graph
    """

    all_images = list()
    geosjon_time = time.time()

    # if RGB, access RG images and copy them to a single list
    if not ms_bool:
        img_type = 'PS-RGB_8bit'
    else:
        img_type = 'PS-MS'

    # extract rgb images from nested folder structure
    image_path = RGB_image_path
    for folder in os.listdir(RGB_image_path):
        for img in os.listdir(f'{RGB_image_path}/{folder}/{img_type}/'):
            all_images.append(f'{folder}/{img_type}/{img}') #SN3_roads_train_

    for img in tqdm(all_images):
        # extract images
        if not ms_bool:
            name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_', '')
        else:
            if os.path.splitext(img)[1] == '.tif':
                name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_', '') #os.path.splitext(img)[0]
            else:
                continue

        # extract graph
        if os.path.exists(f'{graph_path}{name}.pickle'):
            with open(f'{graph_path}{name}.pickle', "rb") as openfile:
                # load graph and extract geometry
                G = pickle.load(openfile)
                geom = getGeom(f'{image_path}{img}')

                # apply transformation to nodes
                for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
                    x_pix, y_pix = attr_dict['pts'][0][1], attr_dict['pts'][0][0]
                    x_WGS, y_WGS = pixelToGeoCoord(x_pix, y_pix, geomTransform=geom)
                    attr_dict["coords"] = (x_WGS, y_WGS)

                # apply transformation to edges
                for start, stop, attr_dict in G.edges(data=True):
                    coords = list()
                    for point in attr_dict['pts']:
                        coords.append(pixelToGeoCoord(point[1], point[0], geomTransform=geom))
                    attr_dict['coords'] = coords

                point_features, linestring_features = convert_graph_to_geojson(G) #f'{image_path}{name}.png')

                feature_collection_points = geojson.FeatureCollection(point_features)
                feature_collection_linestrings = geojson.FeatureCollection(linestring_features)

                # Write GeoJSON to file
                with open(f'{out_path}/qgis_geojsons/{name}_points.geojson', 'w') as f:
                    geojson.dump(feature_collection_points, f)
                with open(f'{out_path}/qgis_geojsons/{name}_linestrings.geojson', 'w') as f:
                    geojson.dump(feature_collection_linestrings, f)
                with open(f'{out_path}/sub_geojsons/{name}.geojson', 'w') as f:
                    geojson.dump(geojson.FeatureCollection(point_features + linestring_features), f)

    geosjon_time2 = time.time()
    print(f'created geojsons in {round(geosjon_time2 - geosjon_time, 2)}s')
    return True


models = ['3105_MS_combined']
fp = 'D:/SHollendonner/'

srt = time.time()

for name in models:
    base_path = f'{fp}/segmentation_results/{name}/'
    print(name)
    from_path_gt_masks = f'{fp}/not_tiled/mask_graphs_RGB/'
    if 'MS' in name:
        from_path_gt_masks = f'{fp}/not_tiled/mask_graphs_MS/'

    from_results_path = f'{fp}/segmentation_results/{name}/results/'
    from_path_stitched = f'{fp}/segmentation_results/{name}/stitched/'
    to_path_stitched_postprocessed = f'{fp}/segmentation_results/{name}/stitched_postprocessed/'
    to_path_skeletons = f'{fp}/segmentation_results/{name}/skeletons/'
    to_path_gp_graphs = f'{fp}/segmentation_results/{name}/graphs/'
    to_path_submissions = f'{fp}/segmentation_results/{name}/submissions/'
    to_path_geojsons = f'{fp}/segmentation_results/{name}/geojsons/'
    from_RGB_img_root = f'{fp}/data_3/'
    from_MS_img_root = f'{fp}/data_3/' #'{fp}/multispectral/channels_257/images/'

    print('creating filesystem')
    if not os.path.exists(from_path_stitched):
        os.mkdir(from_path_stitched)
    if not os.path.exists(to_path_skeletons):
        os.mkdir(to_path_skeletons)
    if not os.path.exists(to_path_stitched_postprocessed):
        os.mkdir(to_path_stitched_postprocessed)
    if not os.path.exists(to_path_gp_graphs):
        os.mkdir(to_path_gp_graphs)
    if not os.path.exists(to_path_submissions):
        os.mkdir(to_path_submissions)
    if not os.path.exists(to_path_geojsons):
        os.mkdir(to_path_geojsons)
    if not os.path.exists(f'{to_path_geojsons}/qgis_geojsons/'):
        os.mkdir(f'{to_path_geojsons}/qgis_geojsons/')
    if not os.path.exists(f'{to_path_geojsons}/sub_geojsons/'):
        os.mkdir(f'{to_path_geojsons}/sub_geojsons/')

    stitch = False
    skeletonize = False
    to_geojson = False
    calc_F1 = False
    calc_GED = False
    calc_topo = True

    if stitch:
        print('sorting images')
        sorted_images = sort_images(from_results_path)

        print('determine overlap')
        overlap = determine_overlap(1300, 512)

        print('stitch images')
        stitch_overlap_images(sorted_images=sorted_images,
                          result_path=from_path_stitched,
                          overlap_params=overlap,
                          old_segmentation_result=False,
                          for_visual_output=True)

    # test apls metric with ground truth comparison graph
    if skeletonize:
        print('skeletonise results, create graphs, apply postprocessing')
        skeletonize_segmentations(image_path=from_path_stitched,
                              save_submissions=to_path_submissions,
                              save_graph=to_path_gp_graphs,
                              save_skeleton=to_path_skeletons,
                              save_mask=to_path_stitched_postprocessed,
                              plot=False,
                              single=False)

    if to_geojson:
        print('converting graphs to geojsons')
        convert_all_graphs_to_geojson(graph_path=to_path_gp_graphs,
                                      RGB_image_path=from_RGB_img_root,
                                      MS_image_path=from_MS_img_root,
                                      out_path=to_path_geojsons,
                                      ms_bool=True)


stp = time.time()
print(f"complete computation took: {round(stp-srt, 2)}s")
