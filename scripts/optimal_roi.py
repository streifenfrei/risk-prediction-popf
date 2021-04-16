import os
from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str)
    args = arg_parser.parse_args()
    roi_root = [np.inf, np.inf, np.inf]
    min_roi_size = [-np.inf, -np.inf, -np.inf]
    count = 0
    for root, _, files in os.walk(args.data):
        try:
            int(os.path.basename(root))
        except ValueError:
            continue
        nrrd_files = set([x for x in files if x[-4:] == "nrrd"])
        segmentation_file = set([x for x in nrrd_files if x[-8:] == "seg.nrrd"])
        data_file = nrrd_files - segmentation_file
        if len(segmentation_file) != 1 or len(data_file) != 1:
            print(f"Invalid data directory {root}. Ignoring")
            continue
        data_file = os.path.join(root, list(data_file)[0])
        segmentation_file = os.path.join(root, list(segmentation_file)[0])
        data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
        segmentation_sitk = sitk.ReadImage(segmentation_file, imageIO="NrrdImageIO")
        segmentation_root = segmentation_sitk.TransformIndexToPhysicalPoint((0, 0, 0))
        segmentation_root = data_sitk.TransformPhysicalPointToIndex(segmentation_root)
        roi_root = [min(x, y) for x, y in zip(roi_root, segmentation_root)]
        min_roi_size = [max(x, y) for x, y in zip(min_roi_size, segmentation_sitk.GetSize())]
        count += 1
        print(f"{count} - Processed {root}")
    print(f"Optimal ROI root: {roi_root}\n"
          f"Optimal ROI size: {min_roi_size}")


if __name__ == '__main__':
    main()
