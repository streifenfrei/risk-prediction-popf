import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from scripts.preprocess_dataset import HOUNSFIELD_BOUNDARIES


def visualize_histograms(histograms):
    rows = int(np.ceil(len(histograms) / 2))
    plt.figure(figsize=(8, 8))
    for i, histo in enumerate(histograms):
        histo = np.array(histo)
        histo = histo.astype(np.float) / np.sum(histo)
        plt.subplot(rows, 2, i+1)
        plt.plot(np.arange(HOUNSFIELD_BOUNDARIES[0], HOUNSFIELD_BOUNDARIES[1]), histo)
    plt.show()


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file", "-f", type=str, required=True)
    args = arg_parser.parse_args()

    with open(args.file, "r") as file:
        results = json.load(file)

    visualize_histograms(results["histograms"])


if __name__ == '__main__':
    main()
