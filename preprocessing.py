# separating image channels
import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from utils import create_folder, determine_overlap

# change cv2s limitation on image size
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def square_image(image):
    squared_image = image[:min(image.shape[:2]), :min(image.shape[:2]), :]
    return squared_image


def tile_image(image, save_path, save_name, overlap_indices, to_size=512, verbose=False):
    """
        receives: uploadable image should be square starting in the lower left corner it will be squared, dict with
            parameters, saving path, list of rference images, boolean for augmentation
        returns: dict mapping the split images
        tiles all images and saves onto disk
    """

    # square image if necessary
    if image.shape[0] != image.shape[1]:
        image = square_image(image)

    # get either 3 or 1 image channels (image or mask)
    num_channels = image.shape[-1]
    no_images = int(len(overlap_indices))
    len_id = len(str(no_images))

    image_counter = 0

    # iterate over rows in image and create first naming convention
    for i in range(no_images):
        # create first index
        first_ind = list(str(i))
        while len(first_ind) <= len_id:
            first_ind.insert(0, '0')
        fi = ''.join(first_ind)

        # iterate over columns create second naming convention
        for j in range(no_images):
            second_ind = list(str(j))
            while len(second_ind) <= len_id:
                second_ind.insert(0, '0')
            si = ''.join(second_ind)

            # slice image
            add_image = image[overlap_indices[i][0]:overlap_indices[i][1], overlap_indices[j][0]:overlap_indices[j][1], :]

            # tiling into 256 can lead to rounding error, solved by assigning and cropping a larger base image
            if add_image.shape != (to_size, to_size, num_channels):
                base_img = np.zeros((to_size, to_size, num_channels))
                base_img[:add_image.shape[0], :add_image.shape[1], :] += add_image
                add_image = base_img

            cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}.png', add_image)
            image_counter += 1

    if verbose:
        print(f'Split image {save_name} into {image_counter} images.')

    return


def apply_tiling(source_name, folder_name, to_size, verbose=False):
    """
        receives: path, folder path name, size to tile into
        returns: True
        Tiles images and masks into the wish size with an overlap
    """

    # create folders
    create_folder(f'{source_name}{folder_name}/images', verbose=False)
    create_folder(f'{source_name}{folder_name}/rehashed', verbose=False)

    """
    if 'images' not in os.listdir(f'{source_name}{folder_name}'):
        os.mkdir(f'{source_name}{folder_name}/images')
    if 'rehashed' not in os.listdir(f'{source_name}{folder_name}'):
        os.mkdir(f'{source_name}{folder_name}/rehashed')"""

    # define parameters
    overlap_params = determine_overlap(img_size=1300, to_size=to_size)
    save_path_images = f'{source_name}/{folder_name}/images'
    save_path_masks = f'{source_name}/{folder_name}/rehashed_ones'

    # image files to split
    im_files = os.listdir(f'{source_name}/images')

    # tile images
    for img_name in tqdm(im_files):
        img = cv2.imread(f'{source_name}/images/{img_name}')
        tile_image(img, save_path_images, img_name, overlap_params, verbose=verbose)

    # tile masks
    for img_name in tqdm(os.listdir(f'{source_name}/rehashed')):
        # check if corresponding image exists
        if img_name in im_files:
            img = cv2.imread(f'{source_name}/rehashed/{img_name}')
            tile_image(img, save_path_masks, img_name, overlap_params, verbose=verbose)
        else:
            if verbose:
                print(f'Mask {img_name} was not created as corresponding image does not exist.')

    return


start = time.time()

fp = 'D:/SHollendonner/'
city_folders = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']

for city_folder in city_folders:
    base_source = f'{fp}/data/{city_folder}/'
    to_folder = 'tiled512/'
    wish_size = 512

    apply_tiling(base_source, folder_name=to_folder, to_size=wish_size, verbose=True)

stop = time.time()
print(f'time of script running [s]: {round(stop - start, 2)}')
