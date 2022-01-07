import argparse
import json
import os
from os.path import join as pjoin
import sys
import bz2
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.utils import get_file
from utils.ffhq_dataset.face_alignment import image_align
from utils.ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


class SizePathMap(dict):
    """{size: {aligned_face_path0, aligned_face_path1, ...}, ...}"""
    def add_item(self, size, path):
        if size not in self:
            self[size] = set()
        self[size].add(path)

    def get_sizes(self):
        sizes = []
        for key, paths in self.items():
            sizes.extend([key,]*len(paths))
        return sizes

    def serialize(self):
        result = {}
        for key, paths in self.items():
            result[key] = list(paths)
        return result


def main(args):
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    face_sizes = SizePathMap()
    raw_img_dir = args.raw_image_dir
    img_names = [n for n in os.listdir(raw_img_dir) if os.path.isfile(pjoin(raw_img_dir, n))]
    aligned_image_dir = args.aligned_image_dir
    os.makedirs(aligned_image_dir, exist_ok=True)
    pbar = tqdm(img_names)
    for img_name in pbar:
        pbar.set_description(img_name)
        if os.path.splitext(img_name)[-1] == '.txt':
            continue
        raw_img_path = os.path.join(raw_img_dir, img_name)
        try:
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                aligned_face_path = os.path.join(aligned_image_dir, face_img_name)

                face_size = image_align(
                    raw_img_path, aligned_face_path, face_landmarks, resize=args.resize
                )
                face_sizes.add_item(face_size, aligned_face_path)
                pbar.set_description(f"{img_name}: {face_size}")

                if args.draw:
                    visual = LandmarksDetector.draw(cv2.imread(raw_img_path), face_landmarks)
                    cv2.imwrite(
                            pjoin(args.aligned_image_dir, os.path.splitext(face_img_name)[0] + "_landmarks.png"),
                            visual
                    )
        except Exception as e:
            print('[Error]', e, 'error happened when processing', raw_img_path)

    print(args.raw_image_dir, ':')
    sizes = face_sizes.get_sizes()
    results = {
            'mean_size': np.mean(sizes),
            'num_faces_detected': len(sizes),
            'num_images': len(img_names),
            'sizes': sizes,
            'size_path_dict': face_sizes.serialize(),
        }
    print('\t', results)
    if args.out_stats is not None:
        os.makedirs(os.path.dirname(args.out_stats), exist_ok=True)
        with open(out_stats, 'w') as f:
            json.dump(results, f)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="""
        Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
        python align_images.py /raw_images /aligned_images
        """
    )
    parser.add_argument('raw_image_dir')
    parser.add_argument('aligned_image_dir')
    parser.add_argument('--resize',
            help="True if want to resize to 1024",
            action='store_true')
    parser.add_argument('--draw',
            help="True if want to visualize landmarks",
            action='store_true')
    parser.add_argument('--out_stats',
            help="output_fn for statistics of faces", default=None)
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == "__main__":
    main(parse_args())
