import json
import os
from argparse import ArgumentParser
from itertools import product

import SimpleITK as sitk
import numpy as np

from data_loader import scan_data_directory
from scripts.dataset.preprocess_dataset import HOUNSFIELD_BOUNDARIES, LABELS, get_segmentation_roi, fit_roi

ASPECT_RATIO_RANGE = [(1, 1, 1), (10, 10, 10)]

def histogram(data):
    return np.histogram(data, bins=np.arange(HOUNSFIELD_BOUNDARIES[0], HOUNSFIELD_BOUNDARIES[1] + 1))[0]


def analyze(dataset, do_histograms=True, do_bounding_boxes=True, labels=None):
    if labels is None:
        labels = LABELS
    labels = set(labels)
    labels.add("full")
    assert len(labels - set(LABELS)) == 0, f"invalid labels argument: {labels}"
    dataset = scan_data_directory(dataset, crop="all")
    # collect data (histograms, bounding boxes)
    histograms = [[] for _ in range(len(labels))]
    rois = []
    ids = []
    min_bb_size = np.array([np.inf, np.inf, np.inf])
    max_bb_size = np.array([-np.inf, -np.inf, -np.inf])
    for i, directory in enumerate(dataset):
        ids.append(int(os.path.basename(directory)))
        segmentation_sitk = sitk.ReadImage(os.path.join(directory, "raw", "segmentation.seg.nrrd"))
        if do_histograms:
            data_sitk_list = [sitk.ReadImage(os.path.join(directory, "raw", f"{label}.nrrd")) for label in labels]
            data_np_list = [sitk.GetArrayFromImage(d_sitk) for d_sitk in data_sitk_list]
            for data_np, histo in zip(data_np_list, histograms):
                histo.append(histogram(data_np).tolist())
        if do_bounding_boxes:
            roi_size = get_segmentation_roi(segmentation_sitk).GetSize()
            min_bb_size = np.array([min(x, y) for x, y in zip(min_bb_size, roi_size)])
            max_bb_size = np.array([max(x, y + 1) for x, y in zip(max_bb_size, roi_size)])
            rois.append(np.array(roi_size))
        print(f"\rCollect data: {i + 1}/{len(dataset)}", end="")
    print()
    results = {"ids": ids,
               "histograms": histograms,
               "rois": [r.tolist() for r in rois]}

    # analyze different roi aspect ratios
    ar_coverages = np.zeros(ASPECT_RATIO_RANGE[1])
    ar_coverages_std = np.zeros(ASPECT_RATIO_RANGE[1])
    ar_coverages_min = np.zeros(ASPECT_RATIO_RANGE[1])
    search_space = list(product(*[range(lower, upper) for lower, upper in zip(*ASPECT_RATIO_RANGE)]))
    for i, aspect_ratio in enumerate(search_space):
        x, y, z = aspect_ratio
        new_rois = [fit_roi(r, aspect_ratio) for r in rois]
        coverages = [np.prod(r) / np.prod(n_r) for r, n_r in zip(rois, new_rois)]
        ar_coverages[x][y][z] = np.mean(coverages)
        ar_coverages_std[x][y][z] = np.std(coverages)
        ar_coverages_min[x][y][z] = np.min(coverages)
        print(f"\rAnalyze aspect ratios in range {ASPECT_RATIO_RANGE}: "
              f"{int(np.ceil(100 * i / len(search_space)))}%", end="")
    print()
    results = {**results,
               "ar_coverages": ar_coverages.tolist(),
               "ar_coverages_std": ar_coverages_std.tolist(),
               "ar_coverages_min": ar_coverages_min.tolist(),}

    if do_bounding_boxes:
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
        coverage = np.zeros(max_bb_size - min_bb_size)
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
            coverage[x][y][z] = (np.count_nonzero(current_recalls == 1.)) / len(rois)
            print(f"\rAnalyze bounding boxes in range [{min_bb_size}, {max_bb_size}]: "
                  f"{i + 1}/{len(search_space)} ({int(np.ceil(100 * i / len(search_space)))}%)", end="")
        results = {**results,
                   "histograms": histograms,
                   "labels": list(labels),
                   "bb_range": (tuple(min_bb_size.tolist()), tuple(max_bb_size.tolist())),
                   "precisions": precisions.tolist(),
                   "precisions_std": precisions_std.tolist(),
                   "precisions_min": precisions_min.tolist(),
                   "recalls": recalls.tolist(),
                   "recalls_std": recalls_std.tolist(),
                   "recalls_min": recalls_min.tolist(),
                   "coverage": coverage.tolist()}
    return results


def main(data, output):
    results = analyze(data)
    results_file_path = os.path.join(output, "analysis_results.json")
    with open(results_file_path, "w") as results_file:
        json.dump(results, results_file, sort_keys=True, indent=4)
    print(f"\nWrote results to '{results_file_path}'")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str, required=True)
    arg_parser.add_argument("--output", "-o", type=str, required=True)
    args = arg_parser.parse_args()
    main(args.data, args.output)
