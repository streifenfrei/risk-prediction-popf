import os
import shutil
from argparse import ArgumentParser
from logging import warning
import re

import SimpleITK as sitk
import yaml
import numpy as np

PATIENT_ID_PATTERN = "UKD_{id}$"
DATA_FILE_PATTERN = "CT/NRRD/CT_{id}.nrrd"
SEGMENTATION_FILE_PATTERN = "RoI/combined.nrrd"

LABELS = ["full", "fixed", "roi", "seg"]
MASKING_VALUE = -2000  # just has to be smaller than lower hounsfield boundary
HOUNSFIELD_BOUNDARIES = [-1024, 3071]
HOUNSFIELD_RANGE = HOUNSFIELD_BOUNDARIES[1] - HOUNSFIELD_BOUNDARIES[0]
INTERPOLATION_MAPPING = {
    "linear": sitk.sitkLinear,
    "spline": sitk.sitkBSpline,
    "gaussian": sitk.sitkGaussian
}


def data_iterator(data_directory):
    pattern = PATIENT_ID_PATTERN.format(id="([0-9]{4})")
    for root, _, files in os.walk(data_directory):
        match = re.search(pattern, os.path.basename(root))
        if match:
            patient_id = int(match.group(1))
        else:
            continue
        data_file = os.path.join(root, DATA_FILE_PATTERN.format(id=patient_id))
        segmentation_file = os.path.join(root, SEGMENTATION_FILE_PATTERN.format(id=patient_id))
        if not (os.path.exists(data_file) and os.path.isfile(data_file)) or \
                not (os.path.exists(segmentation_file) and os.path.isfile(data_file)):
            print(f"Invalid data directory {root}. Ignoring")
            continue
        yield patient_id, data_file, segmentation_file


def _np_to_sitk(data_np, reference_data_sitk):
    data_sitk = sitk.GetImageFromArray(data_np)
    data_sitk.SetOrigin(reference_data_sitk.GetOrigin())
    data_sitk.SetSpacing(reference_data_sitk.GetSpacing())
    data_sitk.SetDirection(reference_data_sitk.GetDirection())
    return data_sitk


def clip_intensities(data):
    data_np = sitk.GetArrayFromImage(data)
    data_np = np.clip(data_np, HOUNSFIELD_BOUNDARIES[0], HOUNSFIELD_BOUNDARIES[1])
    return _np_to_sitk(data_np, data)


def resample(data, segmentation, new_size, new_spacing, interpolator):
    center = data.TransformContinuousIndexToPhysicalPoint([sz / 2.0 for sz in data.GetSize()])
    new_origin = [c - c_index * n_spc for c, c_index, n_spc in zip(center, [sz / 2.0 for sz in new_size], new_spacing)]
    data = sitk.Resample(data, size=new_size,
                         outputOrigin=new_origin,
                         outputSpacing=new_spacing,
                         interpolator=interpolator,
                         defaultPixelValue=HOUNSFIELD_BOUNDARIES[0])
    new_size = [int(o_sz * o_spc / n_spc) for o_sz, o_spc, n_spc in
                zip(segmentation.GetSize(), segmentation.GetSpacing(), new_spacing)]
    segmentation = sitk.Resample(segmentation, size=new_size,
                                 outputOrigin=segmentation.GetOrigin(),
                                 outputSpacing=new_spacing,
                                 interpolator=sitk.sitkNearestNeighbor,
                                 defaultPixelValue=0)
    return data, segmentation


class Crop:
    def __init__(self, data: sitk.Image, segmentation: sitk.Image, bb_size):
        self.segmentation = segmentation
        # place bb so that the segmentation is centered
        self.offset = ((np.array(bb_size) - np.array(self.segmentation.GetSize())) / 2).astype(int)
        # adjust offset if resulting bb is out of bounds
        segmentation_origin_in_data = data.TransformPhysicalPointToIndex(segmentation.GetOrigin())
        self.offset = [seg_or if seg_or - off < 0 else off
                       for seg_or, off in zip(segmentation_origin_in_data, self.offset)]
        self.offset = [off + ((seg_or - off + bb_sz) - data_sz) if seg_or - off + bb_sz > data_sz else off
                       for seg_or, off, bb_sz, data_sz in
                       zip(segmentation_origin_in_data, self.offset, bb_size, data.GetSize())]
        cropped_origin = np.array(segmentation_origin_in_data) - self.offset
        assert all(cr_or >= 0 for cr_or in cropped_origin), \
            f"Data size: {data.GetSize()}, BB size: {bb_size}, BB origin: {cropped_origin}"
        assert all(cr_or + bb_s <= data_s for cr_or, bb_s, data_s in zip(cropped_origin, bb_size, data.GetSize())), \
            f"Data size: {data.GetSize()}, BB size: {bb_size}, BB origin: {cropped_origin}"
        self.data = data[cropped_origin[0]:cropped_origin[0] + bb_size[0],
                         cropped_origin[1]:cropped_origin[1] + bb_size[1],
                         cropped_origin[2]:cropped_origin[2] + bb_size[2]]

    def fixed(self):
        return self.data

    def roi(self):
        data_np = sitk.GetArrayFromImage(self.data).transpose()
        segmentation_size = self.segmentation.GetSize()
        mask = np.ones_like(data_np, dtype=bool)
        clipped_offset = np.clip(self.offset, 0, None)
        mask[clipped_offset[0]:clipped_offset[0] + segmentation_size[0],
             clipped_offset[1]:clipped_offset[1] + segmentation_size[1],
             clipped_offset[2]:clipped_offset[2] + segmentation_size[2]] = False
        data_np_roi = np.where(mask, MASKING_VALUE, data_np)
        assert all(sitk_sz == np_sz for sitk_sz, np_sz in zip(self.data.GetSize(), data_np_roi.shape))
        return _np_to_sitk(data_np_roi.transpose(), self.data)

    def seg(self):
        data_np = sitk.GetArrayFromImage(self.data).transpose()
        segmentation_np = sitk.GetArrayFromImage(self.segmentation).transpose()
        segmentation_np = segmentation_np.sum(axis=0).astype(bool)
        inner_offset = [max(-off, 0) for off in self.offset]
        data_size = self.data.GetSize()
        segmentation_np = segmentation_np[inner_offset[0]:inner_offset[0] + data_size[0],
                                          inner_offset[1]:inner_offset[1] + data_size[1],
                                          inner_offset[2]:inner_offset[2] + data_size[2]]
        outer_offset = [max(off, 0) for off in self.offset]
        padding = list((off, data_sz - seg_sz - off)
                       for seg_sz, off, data_sz in
                       zip(segmentation_np.shape, outer_offset, data_size))
        mask = np.pad(segmentation_np, padding, constant_values=False)
        data_np_seg = np.where(mask, data_np, MASKING_VALUE)
        assert all(sitk_sz == np_sz for sitk_sz, np_sz in zip(self.data.GetSize(), data_np_seg.shape))
        return _np_to_sitk(data_np_seg.transpose(), self.data)


def _update_intensity_range(ir, data, masked=False):
    data_np = sitk.GetArrayFromImage(data)
    # for masked data (roi and seg) the effective minimum is the second lowest value as the masked surroundings
    # have the value MASKING_VALUE. In that case also subtract 1 from the minimum to make the voxels with
    # MASKING_VALUE distinguishable from the lowest HE values when clipped during normalization.
    minimum = np.partition(np.unique(data_np), 1, axis=None)[1] - 1 if masked else np.min(data_np)
    ir[0] = min(ir[0], minimum)
    ir[1] = max(ir[1], np.max(data_np))
    return ir


def normalize(data, intensity_range, normalization_range):
    data_np = np.array(sitk.GetArrayFromImage(data))
    data_np = np.clip(data_np, intensity_range[0], intensity_range[1])
    data_np -= intensity_range[0]
    data_np = data_np.astype(float)
    data_np /= intensity_range[1] - intensity_range[0]  # values are now between 0 and 1
    data_np *= normalization_range[1] - normalization_range[0]
    data_np += normalization_range[0]  # values are now in the normalization range
    return _np_to_sitk(data_np, data)


def main(config, data, out, do_resample=True, do_crop=True, do_normalize=True, crops=None):
    if crops is None:
        crops = LABELS
    crops = set(crops)
    assert len(crops - set(LABELS)) == 0, f"invalid crops argument: {crops}"
    target_size = config["resampling"]["size"]
    target_spacing = config["resampling"]["spacing"]
    interpolator = INTERPOLATION_MAPPING[config["resampling"]["interpolation"]]
    bb_size, calculate_bb = ([-np.inf, -np.inf, -np.inf], True) \
        if config["cropping"]["bb_size"] == "auto" else (config["cropping"]["bb_size"], False)
    intensity_ranges = {}
    for crop in crops:
        intensity_ranges[crop] = [[np.inf, -np.inf], True] \
            if config["normalization"][f"ir_{crop}"] == "auto" \
            else [config["normalization"][f"ir_{crop}"], False]
    normalization_range = config["normalization"]["target_range"]
    labels_file = config["labels"]

    dataset = []
    for patient_id, data_file, segmentation_file in data_iterator(data):
        dataset.append((patient_id, data_file, segmentation_file))

    # Resample (and calculate bounding box if required)
    invalid_data = []
    for i, entry in enumerate(dataset):
        patient_id, data_file, segmentation_file = entry
        try:
            data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
            segmentation_sitk = sitk.ReadImage(segmentation_file, imageIO="NrrdImageIO")
        except RuntimeError:
            invalid_data.append(entry)
            continue
        if do_resample:
            data_sitk, segmentation_sitk = resample(data_sitk, segmentation_sitk, target_size, target_spacing, interpolator)
            data_sitk = clip_intensities(data_sitk)
            output_directory = os.path.join(out, str(patient_id), "raw")
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                sitk.WriteImage(data_sitk, os.path.join(output_directory, "full.nrrd"))
                sitk.WriteImage(segmentation_sitk, os.path.join(output_directory, "segmentation.seg.nrrd"))
            print(f"\rResampling: {i + 1}/{len(dataset)}", end="")
        if calculate_bb:
            bb_size = [max(x, y) for x, y in zip(bb_size, segmentation_sitk.GetSize())]
    for entry in invalid_data:
        dataset.remove(entry)
    if len(invalid_data) != 0:
        print(f"\nBroken data in: {','.join([str(i[0]) for i in invalid_data])}")
    else:
        print()

    # Generate cropped data (and calculate intensity ranges if required)
    for i, (patient_id, _, _) in enumerate(dataset):
        output_directory = os.path.join(out, str(patient_id), "raw")
        data_sitk = sitk.ReadImage(os.path.join(output_directory, "full.nrrd"))
        if do_crop:
            segmentation_sitk = sitk.ReadImage(os.path.join(output_directory, "segmentation.seg.nrrd"))
            crop = Crop(data_sitk, segmentation_sitk, bb_size)
            # fixed bb crop
            if "fixed" in crops:
                data_sitk = crop.fixed()
                sitk.WriteImage(data_sitk, os.path.join(output_directory, "fixed.nrrd"))
            # roi crop
            if "roi" in crops:
                data_sitk = crop.roi()
                sitk.WriteImage(data_sitk, os.path.join(output_directory, "roi.nrrd"))
            # segmentation crop
            if "seg" in crops:
                data_sitk = crop.seg()
                sitk.WriteImage(crop.seg(), os.path.join(output_directory, "seg.nrrd"))
            print(f"\rCropping: {i + 1}/{len(dataset)}", end="")
        for crop in crops:
            if intensity_ranges[crop][1]:
                intensity_ranges[crop][0] = _update_intensity_range(intensity_ranges[crop][0], data_sitk)
    print()
    # Normalize intensities
    if do_normalize:
        for i, (patient_id, _, _) in enumerate(dataset):
            raw_directory = os.path.join(out, str(patient_id), "raw")
            output_directory = os.path.join(out, str(patient_id))
            for crop in crops:
                input_path = os.path.join(raw_directory, f"{crop}.nrrd")
                sitk.WriteImage(normalize(sitk.ReadImage(input_path), intensity_ranges[crop][0], normalization_range),
                                os.path.join(output_directory, f"{crop}.nrrd"))
            print(f"\rNormalizing: {i + 1}/{len(dataset)}", end="")

    if os.path.exists(labels_file):
        shutil.copyfile(labels_file, os.path.join(out, "labels.csv"))
    else:
        warning(f"{labels_file} not found.")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", "-i", type=str, required=True)
    arg_parser.add_argument("--config", "-c", type=str, required=True)
    arg_parser.add_argument("--output", "-o", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    main(config, args.input, args.output)
