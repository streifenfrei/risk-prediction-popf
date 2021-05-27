import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from scripts.dataset.preprocess_dataset import HOUNSFIELD_BOUNDARIES, HOUNSFIELD_RANGE, LABELS


def _get_mean_of_histo(histogram):
    if not isinstance(histogram, np.ndarray):
        histogram = np.array(histogram)
    return int(np.sum(histogram * np.arange(0, HOUNSFIELD_RANGE)))


def crop_histogram(histogram, coverage):
    mean = _get_mean_of_histo(histogram)
    cropped_histogram = np.array([histogram[mean]])
    offset_left = 0
    offset_right = 0
    while np.sum(cropped_histogram) < coverage and len(cropped_histogram) < len(histogram):
        offset_left += 1 if mean - offset_left > 0 else 0
        offset_right += 1 if mean + offset_right < len(histogram) else 0
        cropped_histogram = histogram[mean - offset_left:mean + offset_right]
    return cropped_histogram, mean - offset_left, len(histogram) - mean - offset_right


def visualize_histograms(histograms, coverage, ids):
    print(f"Histograms (coverage = {coverage}):")
    means = [[_get_mean_of_histo(np.array(h) / sum(h)) for h in c] for c in histograms]
    histograms = [np.array(h).sum(0) for h in histograms]
    histograms = [h.astype(float) / h.sum() for h in histograms]

    rows = int(np.ceil(len(histograms) / 2))
    fig, axs = plt.subplots(rows, 2, constrained_layout=True)
    for i, (histogram, label) in enumerate(zip(histograms, LABELS)):
        histogram, cutoff_left, cutoff_right = crop_histogram(histogram.astype(np.float) / np.sum(histogram), coverage)
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

    fig, axs = plt.subplots(rows, 2, constrained_layout=True)
    fig.set_dpi(500)
    ids = [str(i) for i in ids]
    for i, (mean, label) in enumerate(zip(means, LABELS)):
        row = int(i / 2)
        col = i % 2
        axs[row][col].bar(ids, mean)
        axs[row][col].set_title(label)
    plt.show()
    for id_and_mean in [list(zip(ids, c)) for c in means]:
        id_and_mean.sort(key=lambda x: x[1])
        print(id_and_mean)
    print()



def f_score(precision, recall, beta):
    beta **= 2
    return ((1 + beta) * precision * recall) / ((beta * precision) + recall)


def _3d_visualization(array, title="42"):
    sitk_image = sitk.GetImageFromArray(array.transpose())
    sitk.Show(sitk_image, title=title)


def visualize_bb_analysis(precisions, precisions_std, precisions_min, recalls,
                          recalls_std, recalls_min, bb_range, coverage, f_beta):
    precisions = np.array(precisions)
    precisions_std = np.array(precisions_std)
    precisions_min = np.array(precisions_min)
    recalls = np.array(recalls)
    recalls_std = np.array(recalls_std)
    recalls_min = np.array(recalls_min)
    coverage = np.array(coverage)
    f_scores = f_score(precisions, recalls, f_beta)
    f_scores_std = f_score(precisions + precisions_std, recalls + recalls_std, f_beta) - f_scores
    f_scores_min = f_score(precisions - precisions_min, recalls - recalls_min, f_beta)
    try:
        _3d_visualization(precisions, title="precisions")
        _3d_visualization(recalls, title="recalls")
        _3d_visualization(f_scores, title="f_scores")
    except RuntimeError:
        pass
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
              f"\t\tRecall (min): {recalls_min.flatten()[index]}\n"
              f"\t\tCoverage: {coverage.flatten()[index]}")

    i = np.argmax(f_scores)
    i_full = np.unravel_index(i, f_scores.shape) + np.array(bb_range[0])
    print(f"\tBest mean F-score for {i_full}:")
    _print_results(i)
    i = np.argmax(f_scores_min)
    i_full = np.unravel_index(i, f_scores_min.shape) + np.array(bb_range[0])
    print(f"\tBest minimal F-score for {i_full}:")
    _print_results(i)
    print()


def main(file, hist_coverage, f_beta):
    with open(file, "r") as file:
        results = json.load(file)

    rois = results["rois"]
    rois.sort(key=lambda x: x[0] * x[1] * x[2])
    print(f"Minimal ROI: {rois[0]}")
    print(f"Maximal ROI: {rois[-1]}")
    print(f"Bounding box range: {results['bb_range']}\n")

    visualize_histograms(results["histograms"], hist_coverage, results["ids"])
    visualize_bb_analysis(results["precisions"], results["precisions_std"], results["precisions_min"],
                          results["recalls"], results["recalls_std"],  results["recalls_min"],
                          results["bb_range"], results["coverage"], f_beta)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file", "-f", type=str, required=True)
    arg_parser.add_argument("--hist_coverage", "-hc", type=float, default=0.98)
    arg_parser.add_argument("--f_beta", "-fb", type=float, default=2.35)
    args = arg_parser.parse_args()
    main(args.file, args.hist_coverage, args.f_beta)