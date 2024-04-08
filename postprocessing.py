# https://github.com/SpaceNetChallenge/RoadDetector/blob/0a7391f546ab20c873dc6744920deef22c21ef3e/selim_sef-solution/tools/vectorize.py
import copy
import math
import os
import pickle
import time

import cv2
import geojson
import numpy as np
import sknw
from PIL import Image
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, Point
from simplification.cutil import simplify_coords
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, skeletonize
from tqdm import tqdm


def determine_overlap(img_size, wish_size):
    num_pics = int(np.ceil(img_size / wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i * (wish_size - applied_step), (i + 1) * wish_size - i * applied_step) for i in
                       range(num_pics)]
    return overlap_indices


def stitch_overlap_images(sorted_images, result_path, overlap_params, old_segmentation_result, for_visual_output):
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
        # print(base_name)

        for img in img_paths:
            names = img.replace('.png', '').split('/')[-1].split('_')[-2:]
            # print(names)
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
            # ret, bw_img = cv2.threshold(mask, street_color - 5, 1, cv2.THRESH_BINARY)
            arrays[ids] = ret_image

        # place arrays in big array
        it = 0
        for i, i_val in enumerate(overlap_params):
            for j, j_val in enumerate(overlap_params):
                if (i, j) in arrays.keys():
                    img = arrays[i, j]
                    # print(i_val[0],i_val[1], j_val[0],j_val[1])
                    # print(img.shape)
                    base[i_val[0]:i_val[1], j_val[0]:j_val[1]] += img
                it += 1
        base[base > 0] = output_color  # 1: for binary, 255: for visual
        cv2.imwrite(f'{result_path}/{base_name}', base[:1300, :1300])
    return


def sort_images(base_path):
    ret_dict = dict()
    # retrieve the image number from a string like this: 'AOI_2_Vegas_PS-RGB_img1_00_00.png' -> img1
    for name in tqdm(os.listdir(base_path)):
        ins_name = f'{name.split("_")[2]}_{name.split("_")[4]}'
        if ins_name in ret_dict.keys():
            ret_dict[ins_name].append(base_path + name)
        elif ins_name not in ret_dict.keys():
            ret_dict[ins_name] = [base_path + name]
    return ret_dict


def to_line_strings(mask, sigma=0.5, threashold=0.3, small_obj_size1=300, dilation=25,
                    return_ske=False):  # , dilation=6 treshhold = 0.3
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
                same_order = math.hypot(start[0] - coords[0][0], start[1] - coords[0][1]) <= math.hypot(
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
    # lengths = [shapely.length(l) for l in new_lines]
    lengths = [l.length for l in new_lines]

    # return skeleton too
    ske[ske > 0] = 255

    return line_strings, lengths, np.uint8(ske), graph, base  # , buffered_arr #ret_mask[8:1308, 8:1308]


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

        length = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

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
    dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
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


def skeletonize_segmentations(image_path, save_submissions, save_graph, save_skeleton, save_mask, geoRGB_path,
                              geoMS_path, plot=False, single=False):
    iterating = os.listdir(image_path)
    all_files_done = os.listdir(save_graph)
    skel1 = time.time()

    # create a list with all georefrenced imaegs
    geo_images = {}
    for image in iterating:
        image_name = image[:-10]
        folder_name = image_name.split('_PS-')[0]

        geo_images[
            image_name] = f'{geoRGB_path}/{folder_name}/PS-RGB_8bit/SN3_roads_train_{image_name.replace("MS", "RGB")}.tif'

    print(geo_images)

    """with Pool(processes=cpu_count()) as pool:
        pool.map(multi_skeletonize_segmentations, zip(image_path, save_submissions, save_graph, save_skeleton, save_mask, geoRGB_path, geoMS_path, plot, single, iterating, geo_images))"""

    for image in tqdm(iterating):
        image_name = image[:-10]
        if f'{image_name}.pickle' in os.listdir(save_graph):
            print()
            continue
        # if f'{image_name}.pickle' in all_files_done:
        #    continue
        img = np.asarray(Image.open(f'{image_path}/{image}'))
        # convert to skeleton and graph, apply morphologicals
        linestrings, lens, final_img, final_graph, postproc_image = to_line_strings(mask=img, sigma=0.5, threashold=0.3,
                                                                                    small_obj_size1=600, dilation=4,
                                                                                    return_ske=False)  # sigma=0.5 small_obj_size=350

        geo_img = geo_images[image_name]

        # # pickle graph
        # pickle.dump(final_graph, open(f'D:/SHollendonner/segmentation_results/1305_512_unet_densenet201_MS_150epochs_small/graphs_not_postprocessed/{image_name}.pickle', 'wb'))
        # continue
        # apply graph postprocessing
        # final_graph = graph_postprocessing(final_graph, final_img, geo_img, plot=plot)

        # save old submission csv
        with open(f'{save_submissions}/{image_name}.csv', 'w') as file:
            file.write('ImageId,WKT_Pix,length_m,travel_time_s\n')
            for line, leng in zip(linestrings, lens):
                file.write(f'{image.split("_")[4]},"{line}",{leng},{leng / 13.66}\n')

        # pickle graph
        pickle.dump(final_graph, open(f'{save_graph}/{image_name}.pickle', 'wb'))

        # save skeleton
        cv2.imwrite(f'{save_skeleton}/{image_name}.png', final_img)

        # save postprocessed image
        # cv2.imwrite(f'{save_mask}/{image[:-10]}.png', final_mask)

        if single:
            break
    skel2 = time.time()
    print(f'finished postprocessing in {round(skel2 - skel1, 2)}s')
    return


def convert_graph_to_geojson(G_g, edge_coordinate_feature_key='coords'):
    point_features, linestring_features = [], []

    for node in G_g.nodes(data=True):
        point = geojson.Point((node[1]['coords'][0], node[1]['coords'][1]))
        feature = geojson.Feature(geometry=point, properties={'id': node[0]})
        point_features.append(feature)

    for start, stop, attr_dict in G_g.edges(data=True):
        # ignore short edges
        if len(attr_dict['coords']) < 2:
            continue

        coords = attr_dict[edge_coordinate_feature_key]
        if edge_coordinate_feature_key == 'geometry':
            line = coords
        else:
            line = geojson.LineString(coords)

        feature = geojson.Feature(geometry=line, properties={})
        linestring_features.append(feature)

    return point_features, linestring_features


def getGeom(inputRaster, sourceSR='', geomTransform='', targetSR=''):
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
    '''From spacenet geotools'''
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

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

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


def convert_all_graphs_to_geojson(graph_path, RGB_image_path, MS_image_path, out_path, ms_bool,
                                  edge_coordinate_feature_key):
    all_pickles = os.listdir(graph_path)
    all_images = list()
    geosjon_time = time.time()

    # if RGB, access RG images and copy them to a single list
    if not ms_bool:
        img_type = 'PS-RGB_8bit'
    else:
        img_type = 'PS-MS'

    # if not ms_bool:
    image_path = RGB_image_path
    for folder in os.listdir(RGB_image_path):
        if 'AOI' in folder:
            for img in os.listdir(f'{RGB_image_path}/{folder}/{img_type}/'):
                all_images.append(f'{folder}/{img_type}/{img}')  # SN3_roads_train_
    # else:
    # image_path = MS_image_path
    # for img in os.listdir(MS_image_path):
    #     all_images.append(img)

    for img in tqdm(all_images):
        if not ms_bool:
            name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_', '')
        else:
            if os.path.splitext(img)[1] == '.tif':
                name = os.path.splitext(img)[0].split('/')[-1].replace('SN3_roads_train_',
                                                                       '')  # os.path.splitext(img)[0]
            else:
                continue
        if os.path.exists(f'{graph_path}{name}.pickle'):
            with open(f'{graph_path}{name}.pickle', "rb") as openfile:
                G = pickle.load(openfile)

                geom = getGeom(f'{image_path}{img}')

                for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
                    x_pix, y_pix = attr_dict['pts'][0][1], attr_dict['pts'][0][0]
                    x_WGS, y_WGS = pixelToGeoCoord(x_pix, y_pix, geomTransform=geom)
                    attr_dict["coords"] = (x_WGS, y_WGS)

                for start, stop, attr_dict in G.edges(data=True):
                    coords = list()
                    for point in attr_dict['pts']:
                        coords.append(pixelToGeoCoord(point[1], point[0], geomTransform=geom))
                    attr_dict['coords'] = coords

                point_features, linestring_features = convert_graph_to_geojson(G,
                                                                               edge_coordinate_feature_key=edge_coordinate_feature_key)  # f'{image_path}{name}.png')

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
    return


models = ['3105_MS_combined']
fp = 'D:/SHollendonner/'

srt = time.time()

for name in models:
    print(f'Postprocessing model {name}')
    base_path = f'{fp}/segmentation_results/{name}/'
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
    from_MS_img_root = f'{fp}/data_3/'

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

    stitch = True
    skeletonize = True
    to_geojson = True

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

    if skeletonize:
        print('skeletonise results, create graphs, apply postprocessing')
        skeletonize_segmentations(image_path=from_path_stitched,
                                  save_submissions=to_path_submissions,
                                  save_graph=to_path_gp_graphs,
                                  save_skeleton=to_path_skeletons,
                                  save_mask=to_path_stitched_postprocessed,
                                  geoRGB_path=from_RGB_img_root,
                                  geoMS_path=from_MS_img_root,
                                  plot=False,
                                  single=False)

    if to_geojson:
        print('converting graphs to geojsons')
        convert_all_graphs_to_geojson(graph_path=to_path_gp_graphs,
                                      RGB_image_path=from_RGB_img_root,
                                      MS_image_path=from_MS_img_root,
                                      out_path=to_path_geojsons,
                                      ms_bool=False,
                                      edge_coordinate_feature_key='geometry')

stp = time.time()
print(f"complete computation took: {round(stp - srt, 2)}s")
