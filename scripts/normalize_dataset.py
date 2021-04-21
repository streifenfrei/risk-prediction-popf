import os
from argparse import ArgumentParser
import SimpleITK as sitk
import yaml

from util import data_iterator
import numpy as np


def normalize(data, segmentation, intensity_range, new_size, new_spacing):
    # intensities
    data_np = sitk.GetArrayFromImage(data)
    data_np = np.clip(data_np, intensity_range[0], intensity_range[1])
    data_np -= intensity_range[0]
    data_np = data_np.astype(np.float)
    data_np /= intensity_range[1] - intensity_range[0]
    data_normalized = sitk.GetImageFromArray(data_np)
    data_normalized.SetOrigin(data.GetOrigin())
    data_normalized.SetSpacing(data.GetSpacing())
    # spacing/size
    center = data_normalized.TransformContinuousIndexToPhysicalPoint([sz / 2.0 for sz in data_normalized.GetSize()])
    new_origin = [c - c_index * n_spc for c, c_index, n_spc in zip(center, [sz / 2.0 for sz in new_size], new_spacing)]
    data_normalized = sitk.Resample(data_normalized, size=new_size, outputOrigin=new_origin, outputSpacing=new_spacing,
                                    interpolator=sitk.sitkLinear, defaultPixelValue=0)
    new_size = [int(o_sz * o_spc / n_spc) for o_sz, o_spc, n_spc in
                zip(segmentation.GetSize(), segmentation.GetSpacing(), new_spacing)]
    segmentation = sitk.Resample(segmentation, size=new_size, outputOrigin=segmentation.GetOrigin(),
                                 outputSpacing=new_spacing, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
    return data_normalized, segmentation


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str, required=True)
    arg_parser.add_argument("--config", "-c", type=str, required=True)
    arg_parser.add_argument("--output", "-o", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)["data"]["normalization"]
    target_size = config["size"]
    target_spacing = config["spacing"]
    intensity_range = config["intensity_range"]
    assert intensity_range[0] < intensity_range[1]
    for patient_id, data_file, segmentation_file in data_iterator(args.data):
        data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
        segmentation_sitk = sitk.ReadImage(segmentation_file, imageIO="NrrdImageIO")
        data_sitk, segmentation_sitk = normalize(data_sitk, segmentation_sitk,
                                                 intensity_range, target_size, target_spacing)
        output_directory = os.path.join(args.output, str(patient_id))
        os.mkdir(output_directory)
        sitk.WriteImage(data_sitk, os.path.join(output_directory, "data.nrrd"))
        sitk.WriteImage(segmentation_sitk, os.path.join(output_directory, "data.seg.nrrd"))
        print(f"Created: {output_directory}")


if __name__ == '__main__':
    main()
