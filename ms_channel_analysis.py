import numpy as np
import os
from tqdm import tqdm
import time
from itertools import combinations
from utils import read_ms_image


# channel selection calculation
def calculate_difference_between_channels(all_filenames, verbose=False):
    """
        receives: list of all images
        returns: sorted dict of maximum differences with channels as keys
        calculates the maximum difference between all channel combinations
    """
    """
    channel allocation
    0   5   Coastal: 397–454 nm 
    1   3   Blue: 445–517 nm 
    2   2   Green: 507–586 nm 
    3   6   Yellow: 580–629 nm 
    4   1   Red: 626–696 nm
    5   4   Red Edge: 698–749 nm
    6   7   Near-IR1: 765–899 nm
    7   8   Near-IR2: 857–1039 nm
    """

    all_imgs = np.empty((len(all_filenames), 325, 325, 8), dtype='uint16')

    for i, img in tqdm(enumerate(all_filenames)):
        if os.path.splitext(img)[1] == '.tif':
            img = read_ms_image(img)  # f'{img_paths}{img}'
            all_imgs[i, :, :, :] = img

    # calculate all channel combinations
    channel_triples = list(combinations(range(8), 3))

    # calculate max diff for ach channel combination
    diff_values = dict()
    for tripel in channel_triples:
        name = f'{tripel[0]}_{tripel[1]}_{tripel[2]}'
        value = np.mean(np.abs(np.diff(np.array([all_imgs[:, :, :, tripel[0]],
                                                 all_imgs[:, :, :, tripel[1]],
                                                 all_imgs[:, :, :, tripel[2]]]), axis=1)))
        diff_values[name] = value

    # sort dict after max diff
    sorted_dict = sorted(diff_values.items(), key=lambda x: x[1], reverse=True)
    if verbose:
        print(sorted_dict)

    return sorted_dict


start = time.time()

fp = 'D:/SHollendonner/'
base_path = f'{fp}/data_3/'
img_path_list = [f'{base_path}/AOI_2_Vegas/', f'{base_path}/AOI_4_Shanghai/', f'{base_path}/AOI_5_Khartoum/',
                 f'{base_path}/AOI_3_Paris/']

all_names_comp = list()
for folder in img_path_list:
    for file in os.listdir(folder + 'MS/'):
        if 'aux' not in file:
            all_names_comp.append(folder + 'MS/' + file)

sort_dict = calculate_difference_between_channels(all_names_comp, verbose=True)
stop = time.time()

print(f'time of script running [s]: {round(stop - start, 2)}')
