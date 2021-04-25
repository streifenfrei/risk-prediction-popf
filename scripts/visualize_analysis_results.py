import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from scripts.preprocess_dataset import HOUNSFIELD_BOUNDARIES, HOUNSFIELD_RANGE


def _crop_histogram(histogram, coverage):
    mean = int(np.sum(histogram * np.arange(0, HOUNSFIELD_RANGE)))
    cropped_histogram = np.array([histogram[mean]])
    offset_left = 0
    offset_right = 0
    while np.sum(cropped_histogram) < coverage and len(cropped_histogram) < len(histogram):
        offset_left += 1 if mean - offset_left > 0 else 0
        offset_right += 1 if mean + offset_right < len(histogram) else 0
        cropped_histogram = histogram[mean - offset_left:mean + offset_right]
    return cropped_histogram, mean - offset_left, len(histogram) - mean - offset_right


def visualize_histograms(histograms, coverage):
    rows = int(np.ceil(len(histograms) / 2))
    plt.figure(figsize=(8, 8))
    for i, histogram in enumerate(histograms):
        histogram = np.array(histogram)
        histogram, cutoff_left, cutoff_right = _crop_histogram(histogram.astype(np.float) / np.sum(histogram), coverage)
        plt.subplot(rows, 2, i+1)
        plt.plot(np.arange(HOUNSFIELD_BOUNDARIES[0] + cutoff_left, HOUNSFIELD_BOUNDARIES[1] - cutoff_right), histogram)
    plt.show()


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file", "-f", type=str, required=True)
    arg_parser.add_argument("--hist_coverage", "-hc", type=float, default=0.95)
    args = arg_parser.parse_args()

    with open(args.file, "r") as file:
        results = json.load(file)

    visualize_histograms(results["histograms"], args.hist_coverage)


if __name__ == '__main__':
    main()
