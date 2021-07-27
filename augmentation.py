from abc import ABC

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.spatial_transformations import augment_resize

from batchgenerators.transforms import SpatialTransform, RandomShiftTransform, AbstractTransform


ROTATION_P = 0.75
ROTATION_ANGLE = 5
SHIFT_P = 0.75
SHIFT_MU = 0
SHIFT_SIGMA = 3
MAX_ZOOM = 1.2
ZOOM_P = 0.75


class RealZoomTransform(AbstractTransform, ABC):
    def __init__(self, max_zoom, p_per_sample):
        assert max_zoom > 1
        self.max_zoom = max_zoom
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict["data"]
        seg = data_dict.get("seg", None)
        dimensionality = len(data.shape) - 2
        size = np.array(data.shape[-dimensionality:], dtype=int)
        zoom = 1 + (np.random.random() * (self.max_zoom - 1))
        samples = []
        segs = []
        split_data = np.split(data, data.shape[0], 0)
        split_seg = np.split(seg, seg.shape[0], 0) if seg is not None else None
        for i in range(len(split_data)):
            sample = split_data[i]
            seg = split_seg[i] if split_seg is not None else None
            if np.random.random() < self.p_per_sample:
                sample, seg = crop(sample, seg=seg, crop_size=(size / zoom).astype(int), crop_type="random")
                sample, seg = augment_resize(sample.squeeze(0), sample_seg=seg.squeeze(0), target_size=size.tolist())
            else:
                sample = sample.squeeze(0)
                if seg is not None:
                    seg = seg.squeeze(0)
            samples.append(sample)
            segs.append(seg)
        data = np.stack(samples)
        data_dict["data"] = data
        if seg is not None:
            segs = np.stack(segs)
            data_dict["seg"] = segs
        return data_dict


def get_transforms():
    rot_angle = np.radians(ROTATION_ANGLE)
    return [
        SpatialTransform(
            patch_size=None,
            do_elastic_deform=False,
            do_rotation=True,
            p_rot_per_sample=ROTATION_P,
            angle_x=(0, 0),
            angle_y=(0, 0),
            angle_z=(-rot_angle, rot_angle),
            border_mode_data="constant",
            border_cval_data=0,
            do_scale=False,
            random_crop=False
        ),
        RandomShiftTransform(
            shift_mu=SHIFT_MU,
            shift_sigma=SHIFT_SIGMA,
            p_per_sample=SHIFT_P,
            p_per_channel=1
        ),
        RealZoomTransform(
            max_zoom=MAX_ZOOM,
            p_per_sample=ZOOM_P
        )
    ]
