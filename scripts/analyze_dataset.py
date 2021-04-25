import json
import os
from argparse import ArgumentParser

import SimpleITK as sitk
import numpy as np

from scripts.preprocess_dataset import HOUNSFIELD_BOUNDARIES

HOUNSFIELD_RANGE = HOUNSFIELD_BOUNDARIES[1] - HOUNSFIELD_BOUNDARIES[0]
FILE_NAMES = ["full.nrrd", "fixed.nrrd", "roi.nrrd", "seg.nrrd"]


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

    # histograms
    histograms = [np.zeros(HOUNSFIELD_RANGE, dtype=np.int) for _ in range(len(FILE_NAMES))]
    for i, directory in enumerate(dataset):
        data_sitk_list = [sitk.ReadImage(os.path.join(directory, "raw", file_name)) for file_name in FILE_NAMES]
        data_np_list = [sitk.GetArrayFromImage(d_sitk) for d_sitk in data_sitk_list]
        for data_np, histo in zip(data_np_list, histograms):
            histo += histogram(data_np)
        print(f"\rAccumulate histograms: {i + 1}/{len(dataset)}", end="")
    for i, histo in enumerate(histograms):
        histograms[i] = (histo.astype(np.float) / np.sum(histo)).tolist()

    # write results to file
    with open(os.path.join(args.output, "analysis_results.json"), "w") as result_file:
        results = {"histograms": histograms}
        json.dump(results, result_file, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
