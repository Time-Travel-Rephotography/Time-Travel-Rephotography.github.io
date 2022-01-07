from argparse import ArgumentParser
import os
from os.path import join as pjoin
from subprocess import run

import numpy as np
import cv2
from tqdm import tqdm


def create_skin_mask(anno_dir, mask_dir, skin_thresh=13, include_hair=False):
    names = os.listdir(anno_dir)
    names = [n for n in names if n.endswith('.png')]
    os.makedirs(mask_dir, exist_ok=True)
    for name in tqdm(names):
        anno = cv2.imread(pjoin(anno_dir, name), 0)
        mask = np.logical_and(0 < anno, anno <= skin_thresh)
        if include_hair:
            mask |= anno == 17
        cv2.imwrite(pjoin(mask_dir, name), mask * 255)


def main(args):
    FACE_PARSING_DIR = 'third_party/face_parsing'

    main_env = os.getcwd()
    os.chdir(FACE_PARSING_DIR)
    tmp_parse_dir = pjoin(args.out_dir, 'face_parsing')
    cmd = [
        'python',
        'test.py',
        args.in_dir,
        tmp_parse_dir,
    ]
    print(' '.join(cmd))
    run(cmd)

    create_skin_mask(tmp_parse_dir, args.out_dir, include_hair=args.include_hair)

    os.chdir(main_env)


def parse_args(args=None, namespace=None):
    parser = ArgumentParser("Face Parsing and generate skin (& hair) mask")
    parser.add_argument('in_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--include_hair', action="store_true", help="include hair in the mask")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == "__main__":
    main(parse_args())



