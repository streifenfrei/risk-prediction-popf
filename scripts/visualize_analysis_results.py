import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from scripts.analyze_dataset import LABELS
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
    print(f"Histograms (coverage = {coverage}):")
    rows = int(np.ceil(len(histograms) / 2))
    fig, axs = plt.subplots(rows, 2, constrained_layout=True)
    for i, (histogram, label) in enumerate(zip(histograms, LABELS)):
        histogram = np.array(histogram)
        histogram, cutoff_left, cutoff_right = _crop_histogram(histogram.astype(np.float) / np.sum(histogram), coverage)
        row = int(i / 2)
        col = i % 2
        lower_boundary = HOUNSFIELD_BOUNDARIES[0] + cutoff_left
        upper_boundary = HOUNSFIELD_BOUNDARIES[1] - cutoff_right
        axs[row][col].fill_between(np.arange(lower_boundary, upper_boundary), histogram)
        axs[row][col].plot(np.arange(lower_boundary, upper_boundary), histogram)
        axs[row][col].set_xticks([lower_boundary, 0, upper_boundary])
        axs[row][col].set_ylim([0., None])
        axs[row][col].set_title(label)
        print(f"\t{label}: [{lower_boundary}, {upper_boundary}] ({upper_boundary - lower_boundary})")
    plt.show()
    print()


def f_score(precision, recall, beta):
    beta **= 2
    return ((1 + beta) * precision * recall) / ((beta * precision) + recall)


def _3d_visualization(array, title="42"):
    sitk_image = sitk.GetImageFromArray(array.transpose())
    sitk.Show(sitk_image, title=title)


def visualize_bb_analysis(precisions, precisions_std, precisions_min,
                          recalls, recalls_std, recalls_min, bb_range, f_beta):
    precisions = np.array(precisions)
    precisions_std = np.array(precisions_std)
    precisions_min = np.array(precisions_min)
    recalls = np.array(recalls)
    recalls_std = np.array(recalls_std)
    recalls_min = np.array(recalls_min)
    f_scores = f_score(precisions, recalls, f_beta)
    f_scores_std = f_score(precisions + precisions_std, recalls + recalls_std, f_beta) - f_scores
    f_scores_min = f_score(precisions - precisions_min, recalls - recalls_min, f_beta)
    _3d_visualization(precisions, title="precisions")
    _3d_visualization(recalls, title="recalls")
    _3d_visualization(f_scores, title="f_scores")
    print(f"Best bounding boxes (f_beta = {f_beta}):")

    def _print_results(index):
        print(f"\t\tF-score (mean): {f_scores.flatten()[index]}\n"
              f"\t\tF-score (standard deviation): {f_scores_std.flatten()[index]}\n"
              f"\t\tF-score (min): {f_scores_min.flatten()[index]}\n"
              f"\t\tPrecision (mean): {precisions.flatten()[index]}\n"
              f"\t\tPrecision (standard deviation): {precisions_std.flatten()[index]}\n"
              f"\t\tPrecision (min): {precisions_min.flatten()[index]}\n"
              f"\t\tRecall (mean): {recalls.flatten()[index]}\n"
              f"\t\tRecall (standard deviation): {recalls_std.flatten()[index]}\n"
              f"\t\tRecall (min): {recalls_min.flatten()[index]}")

    i = np.argmax(f_scores)
    i_full = np.unravel_index(i, f_scores.shape) + np.array(bb_range[0])
    print(f"\tBest mean F-score for {i_full}:")
    _print_results(i)
    i = np.argmax(f_scores_min)
    i_full = np.unravel_index(i, f_scores_min.shape) + np.array(bb_range[0])
    print(f"\tBest minimal F-score for {i_full}:")
    _print_results(i)
    print()


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file", "-f", type=str, required=True)
    arg_parser.add_argument("--hist_coverage", "-hc", type=float, default=0.95)
    arg_parser.add_argument("--f_beta", "-fb", type=float, default=1.)
    args = arg_parser.parse_args()

    with open(args.file, "r") as file:
        results = json.load(file)

    visualize_histograms(results["histograms"], args.hist_coverage)
    visualize_bb_analysis(results["precisions"], results["precisions_std"], results["precisions_min"],
                          results["recalls"], results["recalls_std"],  results["recalls_min"],
                          results["bb_range"], args.f_beta)


if __name__ == '__main__':
    main()