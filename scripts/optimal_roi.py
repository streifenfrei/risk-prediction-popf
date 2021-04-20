import os
from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np

from util import data_iterator


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str)
    args = arg_parser.parse_args()
    roi_root = [np.inf, np.inf, np.inf]
    min_roi_size = [-np.inf, -np.inf, -np.inf]
    count = 0
    for patient_id, data_file, segmentation_file in data_iterator(args.data):
        data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
        segmentation_sitk = sitk.ReadImage(segmentation_file, imageIO="NrrdImageIO")
        segmentation_root = segmentation_sitk.TransformIndexToPhysicalPoint((0, 0, 0))
        segmentation_root = data_sitk.TransformPhysicalPointToIndex(segmentation_root)
        roi_root = [min(x, y) for x, y in zip(roi_root, segmentation_root)]
        min_roi_size = [max(x, y) for x, y in zip(min_roi_size, segmentation_sitk.GetSize())]
        count += 1
        print(f"{count} - Processed {patient_id}")
    print(f"Optimal ROI root: {roi_root}\n"
          f"Optimal ROI size: {min_roi_size}")


if __name__ == '__main__':
    main()
