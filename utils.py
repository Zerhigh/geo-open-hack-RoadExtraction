import os
import numpy as np
import skimage
from PIL import Image


def read_ms_image(image_path):
    """
        receives: file path
        returns: 8 channel image
    """
    image = skimage.io.imread(image_path, plugin='tifffile')
    return image


def determine_overlap(img_size, to_size, verbose=False):
    """
        receives: image size to split, size image is split into
        returns: list of tuples describing the indices to split an image along
        calculates indices on which an image has to be split
    """

    num_pics = int(np.ceil(img_size / to_size))
    applied_step = int((num_pics * to_size - img_size) / (num_pics - 1))
    overlap_indices = [(i * (to_size - applied_step), (i + 1) * to_size - i * applied_step) for i in
                       range(num_pics)]
    if verbose:
        print(overlap_indices)

    return overlap_indices


def read_image(source):
    image = np.asarray(Image.open(source)).flatten()
    return image


def create_folder(folder_name, verbose=False):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

        if verbose:
            print(f'creating folder: {folder_name}')

    return


def flatten(l):
    """
    Flatten a list with sublists.
    returns a single list.
    """
    return [item for sublist in l for item in sublist]
