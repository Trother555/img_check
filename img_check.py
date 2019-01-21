import cv2
import numpy as np
from skimage.measure import compare_ssim
import argparse
import sys
import os
from glob import glob
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))


def read_transparent(filename):
    """Read image that may have alpha channel and convert alpha
    into white bg. Taken from
    https://stackoverflow.com/questions/3803888/opencv-how-to-load-png-images-with-4-channels

    Args:
        filename (str): path to the image
    """

    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_4channel.shape[2] < 4:
        return image_4channel
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor),
                                  axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def read_images(paths):
    """Read all images in paths

    Args:
        paths (list): list of image pathes

    Returns:
        list: list of tuples of image and its name (ndarray, str)
    """
    result = []
    for p in paths:
        result.append((cv2.cvtColor(read_transparent(p), cv2.COLOR_BGR2GRAY),
                       os.path.basename(p)))
    return result


def compare_images(ref_path, control_path, score_th=0.9):
    """Compares images in ref_path directory with images in control_path
        directory  with ssim algorithm. Images considered equal if their
        ssim score is greater than score_th

    Args:
        ref_path (str): path to refference images directory
        control_path (str): path to control images directory
        score_th (float): equality threshold

    Returns:
        dic: keys are ref file names and values are lists of equal images
            filenames
    """
    # get image pathes
    ref_images_path = (glob(os.path.join(ref_path, "*.jpg")) +
                       glob(os.path.join(ref_path, "*.png")) +
                       glob(os.path.join(ref_path, "*.gif")))

    ctrl_images_path = (glob(os.path.join(control_path, "*.jpg")) +
                        glob(os.path.join(control_path, "*.png")) +
                        glob(os.path.join(control_path, "*.gif")))

    # read images
    ref_images = read_images(ref_images_path)
    ctrl_images = read_images(ctrl_images_path)

    # comparation logic
    result = {}
    for rimage, rname in tqdm(ref_images):
        result[rname] = []
        for cimage, cname in ctrl_images:
            score, *rest = compare_ssim(rimage, cimage, full=True)
            if score > score_th:  # consider equal
                result[rname].append(cname)

    return result


def main(args):
    """The main function
    Args:
        args (list): cli args
    """

    # parse args
    parser = argparse.ArgumentParser(description='Compare images')
    parser.add_argument('-r', required=True,
                        help='reference images directory path')
    parser.add_argument('-c', default=current_dir,
                        help='control images directory path')
    parser.add_argument('-o', default=os.path.join(current_dir, 'log.txt'),
                        help='results file')
    args = parser.parse_args(args)

    # compare images
    res = compare_images(args.r, args.c)

    # print results
    with open(args.o, 'w') as f:
        for k, v in res.items():
            f.write(f'{k} : {v}\n')


if __name__ == '__main__':
    main(sys.argv[1:])
