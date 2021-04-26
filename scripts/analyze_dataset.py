import json
import os
from argparse import ArgumentParser
from itertools import product

import SimpleITK as sitk
import numpy as np

from scripts.preprocess_dataset import HOUNSFIELD_BOUNDARIES, HOUNSFIELD_RANGE

LABELS = ["full", "fixed", "roi", "seg"]


def histogram(data):
    return np.histogram(data, bins=np.arange(HOUNSFIELD_BOUNDARIES[0], HOUNSFIELD_BOUNDARIES[1] + 1))[0]


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str, required=True)
    arg_parser.add_argument("--output", "-o", type=str, required=True)
    args = arg_parser.parse_args()

    dataset = []
    for directory in os.scandir(args.data):
        if directory.is_dir():
            try:
                int(directory.name)
            except ValueError:
                continue
            dataset.append(directory.path)

    # collect data (histograms, bounding boxes)
    histograms = [np.zeros(HOUNSFIELD_RANGE, dtype=np.int) for _ in range(len(LABELS))]
    rois = []
    min_bb_size = [np.inf, np.inf, np.inf]
    max_bb_size = [-np.inf, -np.inf, -np.inf]
    for i, directory in enumerate(dataset):
        data_sitk_list = [sitk.ReadImage(os.path.join(directory, "raw", f"{label}.nrrd")) for label in LABELS]
        data_np_list = [sitk.GetArrayFromImage(d_sitk) for d_sitk in data_sitk_list]
        for data_np, histo in zip(data_np_list, histograms):
            histo += histogram(data_np)
        segmentation_sitk = sitk.ReadImage(os.path.join(directory, "raw", "segmentation.seg.nrrd"))
        min_bb_size = np.array([min(x, y) for x, y in zip(min_bb_size, segmentation_sitk.GetSize())])
        max_bb_size = np.array([max(x, y + 1) for x, y in zip(max_bb_size, segmentation_sitk.GetSize())])
        rois.append(np.array(segmentation_sitk.GetSize()))
        print(f"\rCollect data: {i + 1}/{len(dataset)}", end="")
    print()
    for i, histo in enumerate(histograms):
        histograms[i] = (histo.astype(np.float) / np.sum(histo)).tolist()

    # calculate precision and recall for all meaningful bounding boxes
    rois = np.array(rois)
    rois_volumes = np.prod(rois, axis=1)
    search_space = list(product(*[range(lower, upper) for lower, upper in zip(min_bb_size, max_bb_size)]))
    precisions = np.zeros(max_bb_size - min_bb_size)
    precisions_std = np.zeros(max_bb_size - min_bb_size)
    precisions_min = np.zeros(max_bb_size - min_bb_size)
    recalls = np.zeros(max_bb_size - min_bb_size)
    recalls_std = np.zeros(max_bb_size - min_bb_size)
    recalls_min = np.zeros(max_bb_size - min_bb_size)
    for i, bounding_box in enumerate(search_space):
        x, y, z = bounding_box - min_bb_size
        bounding_box = np.array(bounding_box)
        bb_volume = np.prod(bounding_box)  # (= predicted positives)
        overlaps = np.prod(np.clip(rois, None, bounding_box), axis=1)  # (= true positives)
        current_precisions = overlaps / bb_volume
        precisions[x][y][z] = np.mean(current_precisions)
        precisions_std[x][y][z] = np.std(current_precisions)
        precisions_min[x][y][z] = np.min(current_precisions)
        current_recalls = overlaps / rois_volumes
        recalls[x][y][z] = np.mean(current_recalls)
        recalls_std[x][y][z] = np.std(current_recalls)
        recalls_min[x][y][z] = np.min(current_recalls)
        print(f"\rAnalyze bounding boxes in range [{min_bb_size}, {max_bb_size}]: "
              f"{i + 1}/{len(search_space)} ({int(np.ceil(100 * i / len(search_space)))}%)", end="")

    # write results to file
    results_file_path = os.path.join(args.output, "analysis_results.json")
    with open(results_file_path, "w") as results_file:
        results = {"histograms": histograms,
                   "bb_range": (tuple(min_bb_size.tolist()), tuple(max_bb_size.tolist())),
                   "precisions": precisions.tolist(),
                   "precisions_std": precisions_std.tolist(),
                   "precisions_min": precisions_min.tolist(),
                   "recalls": recalls.tolist(),
                   "recalls_std": recalls_std.tolist(),
                   "recalls_min": recalls_min.tolist()}
        json.dump(results, results_file, sort_keys=True, indent=4)
    print(f"\nWrote results to '{results_file_path}'")


if __name__ == '__main__':
    main()
