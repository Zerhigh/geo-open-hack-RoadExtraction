import os
from tqdm import tqdm
import time
import cv2
from utils import read_ms_image


def write_ms_image(image_name, out_path, channels, extract_channels=False):
    """
        receives: all image files, saving path, selected channels, boolean wether to rescale or not
        returns: True
        reads in all images, and converts them from 8 channel to 3 selected channel images
    """

    save_name = os.path.splitext(image_name)[0].split('/')[-1]
    if os.path.splitext(image_name)[1] == '.tif':
        image = read_ms_image(image_name)
        save_image = image[:, :, channels]

        """
        if extract_channels:
            save_image = image[:, :, channels]
            # (cv2.normalize(, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) * 255).astype('uint8')
        else:
            save_image = image[:, :, channels]"""

        cv2.imwrite(f'{out_path}/{save_name}.png', save_image)

    return


start = time.time()

fp = 'D:/SHollendonner/'

# decided on channel combination (2, 5, 7), to check in qgis select channel (3, 6, 8)
save_path = f'{fp}/multispectral/images_257'
out_path8BIT = f'{fp}/multispectral/images8Bit/'

for img in tqdm(os.listdir(out_path8BIT)):
    write_ms_image(image_name=out_path8BIT + img, out_path=save_path, channels=[2, 5, 7], extract_channels=True)

stop = time.time()

print(f'time of script running [s]: {round(stop - start, 2)}')

# all_names8Bit = [out_path8BIT+img for img in os.listdir(out_path8BIT)]
# write_channels_to_img(all_image_files=all_names8Bit, out_path=out_path, channels=[2, 5, 7], rescale=True)
