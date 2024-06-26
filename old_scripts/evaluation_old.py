import json
import os
import pickle
import statistics
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from utils import flatten, read_image


def calc_F1_for_img(gt, prop):
    """
        receives: ground truth image, proposal image
        returns: F1 value for input images
        calculates F1 of two images
    """

    # calculate true pos., false pos. and false neg.
    tp = np.sum(np.logical_and(prop == 1, gt == 1))
    fp = np.sum(np.logical_and(prop == 1, gt == 0))
    fn = np.sum(np.logical_and(prop == 0, gt == 1))

    # avoid division by zero
    if tp == 0:
        tp = 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # return F1
    return 2 * ((precision * recall)/(precision + recall))


def calc_F1_for_all(base_path, post_proc_state, gt_path):
    """
       receives: proposal path, folder direction for stitched or postprocessed images, ground truth path
       returns: True, saves F1 value to file
       calculates F1 of all images in two directories
   """

    # define paths
    data_path = f'{base_path}/{post_proc_state}/'

    f1_results = list()
    f1_vegas = list()
    f1_paris = list()
    f1_shanghai = list()
    f1_khartoum = list()

    for img in tqdm(os.listdir(data_path)):
        # read data in, apply naming switch
        prop = read_image(f'{data_path}{img}')
        if '_0' in img:
            img = os.path.splitext(img)[0][:-6] + '.png'
        if 'MS' in img:
            img = img.replace('MS', 'RGB')
        gt = read_image(f'{gt_path}{img}')

        # if prop is not binary yet, convert it
        if list(np.unique(prop)) != [0, 1]:
            prop[prop >= 1] = 1

        # calc F1 for images
        res = calc_F1_for_img(gt=gt, prop=prop)

        # allocate F1 scores
        if 'Vegas' in img:
            f1_vegas.append(res)
        if 'Paris' in img:
            f1_paris.append(res)
        if 'Shanghai' in img:
            f1_shanghai.append(res)
        if 'Khartoum' in img:
            f1_khartoum.append(res)
        f1_results.append(res)

    print("total f1 mean:", statistics.mean(f1_results))
    print("vegas f1 mean:", statistics.mean(f1_vegas))
    print("paris f1 mean:", statistics.mean(f1_paris))
    print("shanghai f1 mean:", statistics.mean(f1_shanghai))
    print("khartoum f1 mean:", statistics.mean(f1_khartoum))

    # write results to file
    with open(f'{base_path}/F1_scores_{post_proc_state}.txt', 'w') as file:
        file.write(f"total f1 mean: {statistics.mean(f1_results)}\n")
        file.write(f"vegas f1 mean: {statistics.mean(f1_vegas)}\n")
        file.write(f"paris f1 mean: {statistics.mean(f1_paris)}\n")
        file.write(f"shanghai f1 mean: {statistics.mean(f1_shanghai)}\n")
        file.write(f"khartoum f1 mean: {statistics.mean(f1_khartoum)}\n")

    return


def compare_two_graph_topology(Gp, Gt, node_snapping_distance, plot=False):
    #compare the topology of two graphs
    stat_values = {'matched_nodes': None, 'mean_offset': None, 'mean_path_len_similarity': None, 'mean_path_similarity': None, 'combined':None}

    Gp_node_coords = [node[1]['o'] for node in Gp.nodes(data=True)]
    Gt_node_coords = [node[1]['o'] for node in Gt.nodes(data=True)]
    if len(Gp_node_coords) < 1:
        return stat_values

    distances = euclidean_distances(Gt_node_coords, Gp_node_coords)
    #print(distances)#
    matched_nodes = dict()
    vector_lens = list()
    offset_headings = list()
    ref_vector = np.array([0, 1])

    for own_index, n_gt in enumerate(distances):
        if n_gt.argmin() <= node_snapping_distance:
            nearest_node_index = n_gt.argmin()
            gt_node = list(Gt.nodes(data=True))[own_index]
            gp_node = list(Gp.nodes(data=True))[nearest_node_index]

            vector = gp_node[1]['o'] - gt_node[1]['o']
            len_vector = np.linalg.norm(abs(vector))
            offset_heading = np.degrees(np.arccos(np.dot(vector, ref_vector) / len_vector))

            matched_nodes[gt_node[0]] = gp_node[0]
            vector_lens.append(len_vector)
            offset_headings.append(offset_heading)

    # define values
    num_mathced = len(matched_nodes)
    num_total = len(Gt_node_coords)
    matched_ = num_mathced/num_total
    mean_offset = statistics.mean(vector_lens)

    # calculate the path len similarity
    master_path_len_similarity = list()
    master_path_similarity = list()
    master_combined = list()
    for gt_n, gp_n in matched_nodes.items():
        # construct all lengths from the source node by dijkstra
        length, path = nx.single_source_dijkstra(Gt, gt_n)
        # check for each node in the gp graph, if the path exists, and compare a single dijkstra to compare the lenghts
        path_len_similarity = list()
        path_similarity=list()
        #print("here", path)
        for node, d_gt_len in length.items():
            # dont check its own node.. if it is matched, it will be 0 as well
            if node != gt_n and node in matched_nodes.keys():
                # get corresponging node number for gp graph
                corr_node = matched_nodes[node]
                # check that the path is available and not only contains itself (self loop)
                if nx.has_path(Gp, gp_n, corr_node) and gp_n != corr_node:
                    # calculate the dijsktra length
                    d_gp_len, d_gp_path = nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node) #nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node)
                    #print(path[node], d_gp_path, len(path[node]), len(d_gp_path))
                    # applying the normalized absolute difference for length and node count
                    len_normalized_diff = 1 - np.abs(d_gt_len - d_gp_len) / np.maximum(d_gt_len, d_gp_len)
                    len_normalized_path = 1 - np.abs(len(path[node]) - len(d_gp_path)) / np.maximum(len(path[node]), len(d_gp_path))
                    path_len_similarity.append(len_normalized_diff)
                    path_similarity.append(len_normalized_path)
                    #print('hurrah', d_gt_len, d_gp_len, len_normalized_diff)
            else:
                # comparing against unmatched node
                #print('trying to compare against umatched node')
                pass
        if len(path_len_similarity) > 0:
            mean_path_len_similarity = statistics.mean(path_len_similarity)
            master_path_len_similarity.append(mean_path_len_similarity)

            mean_path_similarity = statistics.mean(path_similarity)
            master_path_similarity.append(mean_path_similarity)

            combined = mean_path_len_similarity * 0.5 + mean_path_similarity * 0.25 + matched_ * 0.25
            master_combined.append(combined)

    #print('matched nodes:', matched_, mean_offset, statistics.harmonic_mean(master_path_len_similarity))
    pls = None
    ps = None
    comb = None
    if len(master_path_len_similarity) > 0:
        pls = statistics.mean(master_path_len_similarity)
        ps = statistics.mean(master_path_similarity)
        comb = statistics.mean(master_combined)
        #print(statistics.mean(master_path_len_similarity), statistics.mean(master_path_similarity))
    #, 'offset_headings': offset_headings
    stat_values['matched_nodes'] = matched_
    stat_values['mean_offset'] = mean_offset
    stat_values['mean_path_len_similarity'] = pls
    stat_values['mean_path_similarity'] = ps
    stat_values['combined'] = comb
    #stat_values = {'matched_nodes': matched_, 'mean_offset': mean_offset, 'harmonic_mean_path_len_similarity': pls}
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for (s, e) in Gp.edges():
            vals = flatten([[v] for v in Gp[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax.plot(ps[:, 0], ps[:, 1], 'blue')
        for (s, e) in Gt.edges():
            vals = flatten([[v] for v in Gt[s][e].values()])
            for val in vals:
                ps = val.get('pts', [])
                ax.plot(ps[:, 0], ps[:, 1], 'green')
        ps_ = np.array([i[1]['o'] for i in Gp.nodes(data=True)])
        ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_ = np.array([i[1]['o'] for i in Gt.nodes(data=True)])
        ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
        ps_matched = np.array([i[1]['o'] for i in Gt.nodes(data=True) if i[0] in list(matched_nodes.keys())])
        ax.scatter(ps_matched[:, 0], ps_matched[:, 1], s=20, c='black') #, 'black', markersize=6

        plt.show()

    # node snapping -> try and snap as many nodes as possible -> snapped_nodes/total_possible_snapped nodes
    #print('stats', stat_values)
    # from the snapped ndoes available in both graphs -> select 4 pairs with distance of more x weights between them
        # calculate distance, nodes and edges between them
    # from the snapped ndoes available in both graphs -> calculate betwenness centrality for each snapped node
    return stat_values


# def compare_two_graph_topology(Gp, Gt, node_snapping_distance, plot=False):
#     """
#         receives: proposal graph, ground truth graph, max distance for nodes to match, boolean for plotting results
#         returns: dict with the metrics values
#         calculate the topology between two graphs
#     """
#
#     #compare the topology of two graphs
#     stat_values = {'matched_nodes': None, 'mean_offset': None, 'mean_path_len_similarity': None, 'mean_path_similarity': None, 'combined':None}
#
#     # extract nodes
#     Gp_node_coords = [node[1]['o'] for node in Gp.nodes(data=True)]
#     Gt_node_coords = [node[1]['o'] for node in Gt.nodes(data=True)]
#     if len(Gp_node_coords) < 1:
#         return stat_values
#
#     # calculate all distances between all nodes
#     distances = euclidean_distances(Gt_node_coords, Gp_node_coords)
#
#     matched_nodes = dict()
#     vector_lens = list()
#     offset_headings = list()
#     ref_vector = np.array([0, 1])
#
#     # determine matching between gt and prop graph
#     for own_index, n_gt in enumerate(distances):
#         if n_gt.argmin() <= node_snapping_distance:
#             nearest_node_index = n_gt.argmin()
#             gt_node = list(Gt.nodes(data=True))[own_index]
#             gp_node = list(Gp.nodes(data=True))[nearest_node_index]
#
#             # calculate offset heading and length of offset
#             vector = gp_node[1]['o'] - gt_node[1]['o']
#             len_vector = np.linalg.norm(abs(vector))
#             offset_heading = np.degrees(np.arccos(np.dot(vector, ref_vector) / len_vector))
#
#             # save matchings
#             matched_nodes[gt_node[0]] = gp_node[0]
#             vector_lens.append(len_vector)
#             offset_headings.append(offset_heading)
#
#     # define values
#     num_mathced = len(matched_nodes)
#     num_total = len(Gt_node_coords)
#     matched_ = num_mathced/num_total
#     mean_offset = statistics.mean(vector_lens)
#
#     # calculate the path len similarity
#     master_path_len_similarity = list()
#     master_path_similarity = list()
#
#     # iterate over all matched nodes
#     for gt_n, gp_n in matched_nodes.items():
#         # construct all lengths from the source node by dijkstra
#         length, path = nx.single_source_dijkstra(Gt, gt_n)
#
#         # check for each node in the gp graph, if the path exists, and compare a single dijkstra to compare the lenghts
#         path_len_similarity = list()
#         path_similarity=list()
#
#         for node, d_gt_len in length.items():
#             # dont check its own node.. if it is matched, it will be 0 as well
#             if node != gt_n and node in matched_nodes.keys():
#                 # get corresponging node number for gp graph
#                 corr_node = matched_nodes[node]
#
#                 # check that the path is available and not only contains itself (self loop)
#                 if nx.has_path(Gp, gp_n, corr_node) and gp_n != corr_node:
#                     # calculate the dijsktra length
#                     d_gp_len, d_gp_path = nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node) #nx.single_source_dijkstra(Gp, gp_n, corr_node) #nx.dijkstra_path_length(Gp, gp_n, corr_node)
#
#                     # applying the normalized absolute difference for length and node count
#                     len_normalized_diff = 1 - np.abs(d_gt_len - d_gp_len) / np.maximum(d_gt_len, d_gp_len)
#                     len_normalized_path = 1 - np.abs(len(path[node]) - len(d_gp_path)) / np.maximum(len(path[node]), len(d_gp_path))
#
#                     path_len_similarity.append(len_normalized_diff)
#                     path_similarity.append(len_normalized_path)
#                     #print('hurrah', d_gt_len, d_gp_len, len_normalized_diff)
#             else:
#                 # comparing against unmatched node
#                 pass
#
#         if len(path_len_similarity) > 0:
#             mean_path_len_similarity = statistics.mean(path_len_similarity)
#             master_path_len_similarity.append(mean_path_len_similarity)
#
#             mean_path_similarity = statistics.mean(path_similarity)
#             master_path_similarity.append(mean_path_similarity)
#
#     pls = None
#     ps = None
#     comb = None
#
#     # calculate dict results values
#     if len(master_path_len_similarity) > 0:
#         pls = statistics.mean(master_path_len_similarity)
#         ps = statistics.mean(master_path_similarity)
#
#         # apply weighted formula
#         combined = pls * 0.5 + ps * 0.25 + matched_ * 0.25
#     else:
#         combined = None
#
#     stat_values['matched_nodes'] = matched_
#     stat_values['mean_offset'] = mean_offset
#     stat_values['mean_path_len_similarity'] = pls
#     stat_values['mean_path_similarity'] = ps
#     stat_values['combined'] = combined
#
#     if plot:
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#         for (s, e) in Gp.edges():
#             vals = flatten([[v] for v in Gp[s][e].values()])
#             for val in vals:
#                 ps = val.get('pts', [])
#                 ax.plot(ps[:, 0], ps[:, 1], 'blue')
#         for (s, e) in Gt.edges():
#             vals = flatten([[v] for v in Gt[s][e].values()])
#             for val in vals:
#                 ps = val.get('pts', [])
#                 ax.plot(ps[:, 0], ps[:, 1], 'green')
#         ps_ = np.array([i[1]['o'] for i in Gp.nodes(data=True)])
#         ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
#         ps_ = np.array([i[1]['o'] for i in Gt.nodes(data=True)])
#         ax.plot(ps_[:, 0], ps_[:, 1], 'r.', markersize=4)
#         ps_matched = np.array([i[1]['o'] for i in Gt.nodes(data=True) if i[0] in list(matched_nodes.keys())])
#         ax.scatter(ps_matched[:, 0], ps_matched[:, 1], s=20, c='black') #, 'black', markersize=6
#
#         plt.show()
#
#     return stat_values


def metric_per_city(save_path, metric_name, value_dict):
    c1, c2, c3, c4 = [], [], [], []
    for name, value in value_dict.items():
        if 'Vegas' in name:
            c1.append(value)
        if 'Paris' in name:
            c2.append(value)
        if 'Shanghai' in name:
            c3.append(value)
        if 'Khartoum' in name:
            c4.append(value)

    with open(f'{save_path}/{metric_name}_scores_per_city.txt', 'w') as file:
        file.write(f"vegas {metric_name} mean: {statistics.mean(c1)}\n")
        file.write(f"paris {metric_name} mean: {statistics.mean(c2)}\n")
        file.write(f"shanghai {metric_name} mean: {statistics.mean(c3)}\n")
        file.write(f"khartoum {metric_name} mean: {statistics.mean(c4)}\n")

    return


def compare_GED_graphs(gp_graphs_path, gt_graphs_path, take_first_result, max_time, out_path):
    t_OGED1 = time.time()
    all_results = {}
    relative_results = {}
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))

            iterations = 0
            # calculate GED for GP and GT
            for v in nx.optimize_graph_edit_distance(gp_graph, gt_graph):
                min_result = v
                iterations += 1
                tc = time.time()
                if take_first_result:
                    break

            # calculate GED for GP and G0 reference graph
            for v in nx.optimize_graph_edit_distance(gp_graph, nx.MultiGraph()):
                GP_G0 = v
                break
            # calculate GED for GT and G0 reference graph
            for v in nx.optimize_graph_edit_distance(gt_graph, nx.MultiGraph()):
                GT_G0 = v
                break
        else:
            pass

        all_results[gt_graph_name] = min_result #GED #min_result
        relative_results[gt_graph_name] = min_result / (GP_G0 + GT_G0)

    mean_GED = statistics.mean(all_results.values())
    #pickle.dump(all_results, open(f'D:/SHollendonner/graded/{name}_iterated_WOHUGH.pickle', 'wb'))
    with open(f'{out_path}/GED{str(round(mean_GED, 2)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_results, f)

    mean_relative_GED = statistics.mean(relative_results.values())
    with open(f'{out_path}/relative_GED{str(round(mean_relative_GED, 4)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_results, f)

    metric_per_city(save_path=out_path, metric_name='GED', value_dict=all_results)
    metric_per_city(save_path=out_path, metric_name='relGED', value_dict=relative_results)

    t_OGED2 = time.time()

    print(f'mean Graph Edit Distance (GD): {round(mean_GED, 2)} in {round(t_OGED2 - t_OGED1, 2)}s')
    return


def compare_topology(gp_graphs_path, gt_graphs_path, node_snapping_distance, out_path):
    top1 = time.time()
    all_graphs_stats = {}
    pls_vals = []
    combined_vals = []

    TOP_vegas = list()
    TOP_paris = list()
    TOP_shanghai = list()
    TOP_khartoum = list()

    print('here', gp_graphs_path, os.listdir(gp_graphs_path))
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name # + '.pickle' # gp_graph_name.replace('_00_00', '')
        # print(f'{gt_graphs_path}/{gt_graph_name}')
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))
            stat_dict = compare_two_graph_topology(Gt=gt_graph, Gp=gp_graph, node_snapping_distance=node_snapping_distance, plot=False)
            all_graphs_stats[gt_graph_name] = stat_dict
            if stat_dict['mean_path_len_similarity'] is not None:
                pls_vals.append(stat_dict['mean_path_len_similarity'])
            if stat_dict['combined'] is not None:
                combined_vals.append(stat_dict['combined'])

                if 'Vegas' in gt_graph_name:
                    TOP_vegas.append(stat_dict['combined'])
                elif 'Paris' in gt_graph_name:
                    TOP_paris.append(stat_dict['combined'])
                elif 'Shanghai' in gt_graph_name:
                    TOP_shanghai.append(stat_dict['combined'])
                elif 'Khartoum' in gt_graph_name:
                    TOP_khartoum.append(stat_dict['combined'])
                else:
                    print('not found', gt_graph_name)


    top2 = time.time()
    #    print(statistics.mean([stat['mean_offset'] for stat in all_graphs_stats.values()]))
    print(f'mean of similar path length: {round(statistics.mean(pls_vals) * 100, 2)}% in {round( top2 -top1 ,2)}s')
    print(f'mean of combined: {round(statistics.mean(combined_vals) * 100, 2)}% in {round(top2 - top1, 2)}s')
    with open(f'{out_path}/statistics_pls_{str(round(statistics.mean(pls_vals) * 100, 2)).replace(".", "_")}_comb{str(round(statistics.mean(combined_vals) * 100, 2)).replace(".", "_")}.json', 'w') as f:
        json.dump(all_graphs_stats, f)

    with open(f'{out_path}/Topology_cities.txt', 'w') as file:
        file.write(f"total TOP mean: {round(statistics.mean(combined_vals) * 100, 2)}\n")
        file.write(f"vegas TOP mean: {round(statistics.mean(TOP_vegas) * 100, 2)}\n")
        file.write(f"paris TOP mean: {round(statistics.mean(TOP_paris) * 100, 2)}\n")
        file.write(f"shanghai TOP mean: {round(statistics.mean(TOP_shanghai) * 100, 2)}\n")
        file.write(f"khartoum TOP mean: {round(statistics.mean(TOP_khartoum) * 100, 2)}\n")

    return

# def compare_topology(gp_graphs_path, gt_graphs_path, node_snapping_distance, out_path):
#     """
#         receives: path to all ground truth graphs, path to all proposal graphs, value for the node snapping distance,
#             path to save the metrics result
#         returns: True, saves metrics result to a json file
#         calculates the topology metric for all graphs in a directory
#     """
#
#     top1 = time.time()
#     all_graphs_stats = {}
#     pls_vals = []
#     combined_vals = []
#
#     # iterate over all gt graphs and search for corresponding prop graph
#     for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
#         gt_graph_name = gp_graph_name
#
#         if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
#             gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))
#
#             # apply topology calculation to two graphs
#             stat_dict = compare_two_graph_topology(Gt=gt_graph, Gp=gp_graph, node_snapping_distance=node_snapping_distance, plot=False)
#             all_graphs_stats[gt_graph_name] = stat_dict
#
#             # save topology results
#             if stat_dict['mean_path_len_similarity'] is not None:
#                 pls_vals.append(stat_dict['mean_path_len_similarity'])
#             if stat_dict['combined'] is not None:
#                 combined_vals.append(stat_dict['combined'])
#
#     top2 = time.time()
#
#     print(f'{gp_graphs_path} mean of combined: {round(statistics.harmonic_mean(combined_vals) * 100, 2)}% in {round(top2 - top1, 2)}s')
#
#     # save result to json, apply harmonic mean to penalize 0 values
#     with open(f'{out_path}/harmonic_mean_statistics_pls_{str(round(statistics.harmonic_mean(pls_vals) * 100, 2)).replace(".", "_")}_comb{str(round(statistics.harmonic_mean(combined_vals) * 100, 2)).replace(".", "_")}.json', 'w') as f:
#         json.dump(all_graphs_stats, f)
#
#     return True



models = ['3105_MS_combined']
fp = 'D:/SHollendonner'
srt = time.time()

for name in models:
    print(f'Evaluating model {name}')
    base_path = f'{fp}/segmentation_results/{name}/'

    from_path_gt_masks = f'{fp}/not_tiled/mask_graphs_RGB/'
    if 'MS' in name:
        from_path_gt_masks = f'{fp}/not_tiled/mask_graphs_MS/'  # '{fp}/not_tiled/mask_graphs_MS'

    from_gt_rehashed = f'{fp}/not_tiled/rehashed/'
    to_path_gp_graphs = f'{fp}/segmentation_results/{name}/graphs/'

    calc_F1 = True
    calc_GED = True
    calc_topo = True

    postproc_states = ['stitched', 'stitched_postprocessed']
    if calc_F1:
        print('calculate F1 score')
        calc_F1_for_all(base_path=base_path,
                        post_proc_state='stitched',
                        gt_path=from_gt_rehashed)

    if calc_GED:
        print('calculate GED score')
        compare_GED_graphs(gp_graphs_path=to_path_gp_graphs,
                           gt_graphs_path=from_path_gt_masks,
                           take_first_result=True,
                           max_time=10,
                           out_path=base_path)

    if calc_topo:
        # print('calculate similar path length score')
        compare_topology(gp_graphs_path=to_path_gp_graphs,
                         gt_graphs_path=from_path_gt_masks,
                         node_snapping_distance=30,
                         out_path=base_path)
