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
        dimensionality = len(data.shape) - 2
        size = np.array(data.shape[-dimensionality:], dtype=int)
        zoom = 1 + (np.random.random() * (self.max_zoom - 1))
        samples = []
        for sample in np.split(data, data.shape[0], 0):
            if np.random.random() < self.p_per_sample:
                sample = crop(sample, crop_size=(size / zoom).astype(int), crop_type="random")[0]
                sample = augment_resize(sample.squeeze(0), sample_seg=None, target_size=size.tolist())[0]
            else:
                sample = sample.squeeze(0)
            samples.append(sample)
        data = np.stack(samples)
        data_dict["data"] = data
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
