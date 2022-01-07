from argparse import (
    ArgumentParser,
    Namespace,
)
import os
from os.path import join as pjoin
from typing import Optional
import sys

import numpy as np
import cv2
from skimage import exposure


# sys.path.append('Face_Detection')
# from align_warp_back_multiple_dlib import match_histograms


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image, src_mask=None, ref_mask=None):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    def rv(im):
        if ref_mask is None:
            return im.flatten()
        return im[ref_mask]

    def sv(im):
        if src_mask is None:
            return im.flatten()
        return im[src_mask]

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(sv(src_b), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(sv(src_g), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(sv(src_r), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(rv(ref_b), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(rv(ref_g), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(rv(ref_r), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def convert_to_BW(im, mode):
    if mode == "b":
        gray = im[..., 0]
    elif mode == "gb":
        gray = (im[..., 0].astype(float) + im[..., 1]) / 2.0
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    return np.stack([gray] * 3, axis=-1)


def parse_args(args=None, namespace: Optional[Namespace] = None):
    parser = ArgumentParser('match histogram of src to ref')
    parser.add_argument('src')
    parser.add_argument('ref')
    parser.add_argument('--out', default=None, help="converted src that matches ref")
    parser.add_argument('--src_mask', default=None, help="mask on which to match the histogram")
    parser.add_argument('--ref_mask', default=None, help="mask on which to match the histogram")
    parser.add_argument('--spectral_sensitivity', choices=['b', 'gb', 'g'], help="match the histogram of corresponding sensitive channel(s)")
    parser.add_argument('--crop', type=int, default=0, help="crop the boundary to match")
    return parser.parse_args(args=args, namespace=namespace)


def main(args):
    A = cv2.imread(args.ref)
    A = convert_to_BW(A, args.spectral_sensitivity)
    B = cv2.imread(args.src, 0)
    B = np.stack((B,) * 3, axis=-1)

    mask_A = cv2.resize(cv2.imread(args.ref_mask, 0), A.shape[:2][::-1],
                        interpolation=cv2.INTER_NEAREST) > 0 if args.ref_mask else None
    mask_B = cv2.resize(cv2.imread(args.src_mask, 0), B.shape[:2][::-1],
                        interpolation=cv2.INTER_NEAREST) > 0 if args.src_mask else None

    if args.crop > 0:
        c = args.crop
        bc = int(c / A.shape[0] * B.shape[0] + 0.5)
        A = A[c:-c, c:-c]
        B = B[bc:-bc, bc:-bc]

    B = match_histograms(B, A, src_mask=mask_B, ref_mask=mask_A)
    # B = exposure.match_histograms(B, A, multichannel=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        cv2.imwrite(args.out, B)

    return B


if __name__ == "__main__":
    main(parse_args())
