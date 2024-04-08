import json
import os
import pickle
import statistics
import time
# https://github.com/SpaceNetChallenge/RoadDetector/blob/0a7391f546ab20c873dc6744920deef22c21ef3e/selim_sef-solution/tools/vectorize.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


def read_image(source):
    """
        receives: image file path
        returns: image converted to a flattened array
    """
    image = np.asarray(Image.open(source)).flatten()
    return image


def flatten(l):
    return [item for sublist in l for item in sublist]


def calc_F1_for_img(tp, fp, fn):
    """
        receives: pixelwise true-positive, false-positive, and false-negative values
        returns: F1 value for input values
        calculates F1 metric of an image
    """

    if tp == 0:
        tp = 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # calculate harmonic mean
    return 2 * ((precision * recall) / (precision + recall))


def calc_IoU_for_img(tp, fp, fn):
    """
        receives: pixelwise true-positive, false-positive, and false-negative values
        returns: IoU value for input values
        calculates IoU metric of an image
    """

    if tp == 0:
        tp = 1
    IoU = tp / (fp + tp + fn)
    return IoU


def metric_per_city(save_path, metric_name, value_dict):
    """
        receives: filepath, metric name, dictionary with the metric result for each image
        returns: creates a file with the metrics split upon each city to ease evaluation
    """

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
    """
        receives: filepaths to predicted and ground truth graphs, booleans, saving path
        returns: GED and relative GED values for each graph pair
    """

    t_OGED1 = time.time()
    all_results = {}
    relative_results = {}
    for gp_graph_name in tqdm(os.listdir(gp_graphs_path)):
        gt_graph_name = gp_graph_name
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(
                open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))

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

        all_results[gt_graph_name] = min_result  # GED #min_result
        relative_results[gt_graph_name] = min_result / (GP_G0 + GT_G0)

    mean_GED = statistics.mean(all_results.values())

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
    """
        receives: filepaths to predicted and ground truth graphs, value to determine the maximum distance for snapping
            similar nodes, saving path
        returns: topology metric for all pairs of graphs
    """

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
        gt_graph_name = gp_graph_name
        if os.path.exists(f'{gt_graphs_path}/{gt_graph_name}'):
            gp_graph, gt_graph = pickle.load(open(f'{gp_graphs_path}{gp_graph_name}', 'rb')), pickle.load(
                open(f'{gt_graphs_path}{gt_graph_name}', 'rb'))
            stat_dict = compare_two_graph_topology(Gt=gt_graph, Gp=gp_graph,
                                                   node_snapping_distance=node_snapping_distance, plot=False)
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

    print(f'mean of similar path length: {round(statistics.mean(pls_vals) * 100, 2)}% in {round(top2 - top1, 2)}s')
    print(f'mean of combined: {round(statistics.mean(combined_vals) * 100, 2)}% in {round(top2 - top1, 2)}s')
    with open(
            f'{out_path}/statistics_pls_{str(round(statistics.mean(pls_vals) * 100, 2)).replace(".", "_")}_comb{str(round(statistics.mean(combined_vals) * 100, 2)).replace(".", "_")}.json',
            'w') as f:
        json.dump(all_graphs_stats, f)

    with open(f'{out_path}/Topology_cities.txt', 'w') as file:
        file.write(f"total TOP mean: {round(statistics.mean(combined_vals) * 100, 2)}\n")
        file.write(f"vegas TOP mean: {round(statistics.mean(TOP_vegas) * 100, 2)}\n")
        file.write(f"paris TOP mean: {round(statistics.mean(TOP_paris) * 100, 2)}\n")
        file.write(f"shanghai TOP mean: {round(statistics.mean(TOP_shanghai) * 100, 2)}\n")
        file.write(f"khartoum TOP mean: {round(statistics.mean(TOP_khartoum) * 100, 2)}\n")

    return


def compare_two_graph_topology(Gp, Gt, node_snapping_distance, plot=False):
    """
        receives: filepaths to predicted and ground truth graphs, value to determine the maximum distance for snapping
            similar nodes, saving path
        returns: topology metric for the pair of graphs
    """
    # compare the topology of two graphs
    stat_values = {'matched_nodes': None, 'mean_offset': None, 'mean_path_len_similarity': None,
                   'mean_path_similarity': None, 'combined': None}

    Gp_node_coords = [node[1]['o'] for node in Gp.nodes(data=True)]
    Gt_node_coords = [node[1]['o'] for node in Gt.nodes(data=True)]
    if len(Gp_node_coords) < 1:
        return stat_values

    distances = euclidean_distances(Gt_node_coords, Gp_node_coords)

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
    matched_ = num_mathced / num_total
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
        path_similarity = list()
        for node, d_gt_len in length.items():
            # dont check its own node.. if it is matched, it will be 0 as well
            if node != gt_n and node in matched_nodes.keys():
                # get corresponging node number for gp graph
                corr_node = matched_nodes[node]
                # check that the path is available and not only contains itself (self loop)
                if nx.has_path(Gp, gp_n, corr_node) and gp_n != corr_node:
                    # calculate the dijsktra length
                    d_gp_len, d_gp_path = nx.single_source_dijkstra(Gp, gp_n,
                                                                    corr_node)
                    # applying the normalized absolute difference for length and node count
                    len_normalized_diff = 1 - np.abs(d_gt_len - d_gp_len) / np.maximum(d_gt_len, d_gp_len)
                    len_normalized_path = 1 - np.abs(len(path[node]) - len(d_gp_path)) / np.maximum(len(path[node]),
                                                                                                    len(d_gp_path))
                    path_len_similarity.append(len_normalized_diff)
                    path_similarity.append(len_normalized_path)
            else:
                # comparing against unmatched node
                pass

        if len(path_len_similarity) > 0:
            mean_path_len_similarity = statistics.mean(path_len_similarity)
            master_path_len_similarity.append(mean_path_len_similarity)

            mean_path_similarity = statistics.mean(path_similarity)
            master_path_similarity.append(mean_path_similarity)

            combined = mean_path_len_similarity * 0.5 + mean_path_similarity * 0.25 + matched_ * 0.25
            master_combined.append(combined)

    pls = None
    ps = None
    comb = None

    if len(master_path_len_similarity) > 0:
        pls = statistics.mean(master_path_len_similarity)
        ps = statistics.mean(master_path_similarity)
        comb = statistics.mean(master_combined)

    stat_values['matched_nodes'] = matched_
    stat_values['mean_offset'] = mean_offset
    stat_values['mean_path_len_similarity'] = pls
    stat_values['mean_path_similarity'] = ps
    stat_values['combined'] = comb

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
        ax.scatter(ps_matched[:, 0], ps_matched[:, 1], s=20, c='black')  # , 'black', markersize=6

        plt.show()

    return stat_values


def calc_F1_IoU_for_all(base_path, post_proc_state, gt_path):
    """
        calculates the F1 and IoU metric for all images, saves the metric results into files
    """

    data_path = f'{base_path}/{post_proc_state}/'

    f1_results = list()
    f1_vegas = list()
    f1_paris = list()
    f1_shanghai = list()
    f1_khartoum = list()

    IoU_results = list()
    IoU_vegas = list()
    IoU_paris = list()
    IoU_shanghai = list()
    IoU_khartoum = list()

    tps, fps, fns, tns = [], [], [], []

    for img in tqdm(os.listdir(data_path)):
        # read data in
        prop = read_image(f'{data_path}{img}')
        if '_0' in img:
            img = os.path.splitext(img)[0][:-6] + '.png'
        if 'MS' in img:
            img = img.replace('MS', 'RGB')
        gt = read_image(f'{gt_path}{img.replace("MS", "RGB")}')

        # if prop is not binary yet, convert it
        if list(np.unique(prop)) != [0, 1]:
            prop[prop >= 1] = 1

        tp = np.sum(np.logical_and(prop == 1, gt == 1))
        fp = np.sum(np.logical_and(prop == 1, gt == 0))
        fn = np.sum(np.logical_and(prop == 0, gt == 1))
        tn = np.sum(np.logical_and(prop == 0, gt == 0))

        F1_res = calc_F1_for_img(tp=tp, fp=fp, fn=fn)
        IoU_res = calc_IoU_for_img(tp=tp, fp=fp, fn=fn)
        total = sum([tp, tn, fp, fn])

        tps.append(tp / total)
        fps.append(fp / total)
        fns.append(fn / total)
        tns.append(tn / total)

        if 'Vegas' in img:
            f1_vegas.append(F1_res)
            IoU_vegas.append(IoU_res)
        elif 'Paris' in img:
            f1_paris.append(F1_res)
            IoU_paris.append(IoU_res)
        elif 'Shanghai' in img:
            f1_shanghai.append(F1_res)
            IoU_shanghai.append(IoU_res)
        elif 'Khartoum' in img:
            f1_khartoum.append(F1_res)
            IoU_khartoum.append(IoU_res)
        else:
            print('not found', img)
        f1_results.append(F1_res)
        IoU_results.append(IoU_res)


    conf_dict = {'tps': sum(tps) / len(tps), 'fps': sum(fps) / len(fps), 'fns': sum(fns) / len(fns),
                 'tns': sum(tns) / len(tns)}

    with open(f'{base_path}/confusion_matrix_data.txt', 'w') as file:
        file.write(f"{conf_dict}\n")

    print("total f1 mean:", statistics.mean(f1_results))
    print("vegas f1 mean:", statistics.mean(f1_vegas))
    print("paris f1 mean:", statistics.mean(f1_paris))
    print("shanghai f1 mean:", statistics.mean(f1_shanghai))
    print("khartoum f1 mean:", statistics.mean(f1_khartoum))

    with open(f'{base_path}/F1_scores_{post_proc_state}.txt', 'w') as file:
        file.write(f"total f1 mean: {statistics.mean(f1_results)}\n")
        file.write(f"vegas f1 mean: {statistics.mean(f1_vegas)}\n")
        file.write(f"paris f1 mean: {statistics.mean(f1_paris)}\n")
        file.write(f"shanghai f1 mean: {statistics.mean(f1_shanghai)}\n")
        file.write(f"khartoum f1 mean: {statistics.mean(f1_khartoum)}\n")

    print("total IoU mean:", statistics.mean(IoU_results))
    print("vegas IoU mean:", statistics.mean(IoU_vegas))
    print("paris IoU mean:", statistics.mean(IoU_paris))
    print("shanghai IoU mean:", statistics.mean(IoU_shanghai))
    print("khartoum IoU mean:", statistics.mean(IoU_khartoum))

    with open(f'{base_path}/IoU_scores_{post_proc_state}.txt', 'w') as file:
        file.write(f"total IoU mean: {statistics.mean(IoU_results)}\n")
        file.write(f"vegas IoU mean: {statistics.mean(IoU_vegas)}\n")
        file.write(f"paris IoU mean: {statistics.mean(IoU_paris)}\n")
        file.write(f"shanghai IoU mean: {statistics.mean(IoU_shanghai)}\n")
        file.write(f"khartoum IoU mean: {statistics.mean(IoU_khartoum)}\n")

    return


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

    calc_F1_IoU = True
    calc_GED = True
    calc_topo = True

    postproc_states = ['stitched', 'stitched_postprocessed']
    if calc_F1_IoU:
        print('calculate F1 score')
        calc_F1_IoU_for_all(base_path=base_path,
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
        print('calculate similar path length score')
        compare_topology(gp_graphs_path=to_path_gp_graphs,
                         gt_graphs_path=from_path_gt_masks,
                         node_snapping_distance=30,
                         out_path=base_path)
