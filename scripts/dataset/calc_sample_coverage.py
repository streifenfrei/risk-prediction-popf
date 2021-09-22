import os
from argparse import ArgumentParser

import yaml
import SimpleITK as sitk
import tensorflow as tf
import numpy as np

from data_loader import get_dataset_from_config
# https://stackoverflow.com/a/65523597
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


def main(config):
    dataset, _ = get_dataset_from_config(config, config["3D"])
    for i, (_, data_file, segmentation_file) in enumerate(dataset):
        directory = os.path.split(segmentation_file)[0]
        data_sitk = sitk.ReadImage(data_file)
        segmentation_sit = sitk.ReadImage(segmentation_file)
        segmentation = sitk.GetArrayFromImage(segmentation_sit).transpose()
        origin = segmentation_sit.TransformPhysicalPointToIndex(data_sitk.GetOrigin())
        size = data_sitk.GetSize()
        segmentation = segmentation[origin[0]:origin[0] + size[0],
                       origin[1]:origin[1] + size[1],
                       origin[2]:origin[2] + size[2]]
        sample_shape = config["data"]["sample"]["size"]
        if any(x > y for x, y in zip(sample_shape, segmentation.shape)):
            raise ValueError(f"Sample shape exceeds RoI of {segmentation_file}")
        seg_tf = tf.convert_to_tensor(segmentation, dtype=tf.float32)
        seg_tf = tf.reshape(seg_tf, [1, *list(seg_tf.shape), 1])
        density = tf.squeeze(tf.nn.conv3d(seg_tf, tf.ones([*sample_shape, 1, 1], dtype=tf.float32),
                                          strides=[1, 1, 1, 1, 1], padding="VALID") / np.product(sample_shape))
        roots = tf.where(density >= config["data"]["sample"]["min_coverage"])
        if tf.size(roots) == 0:
            raise ValueError(f"No valid samples in {segmentation_file}")
        np.save(os.path.join(directory, "sample_coverages.npy"), roots)
        print(f"\r{i + 1}/{len(dataset)}", end="")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    main(config)
