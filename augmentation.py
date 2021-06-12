import numpy as np

from batchgenerators.transforms import SpatialTransform, RandomShiftTransform

ROTATION_P = 0.5
ROTATION_ANGLE = 5
SHIFT_P = 0.5
SHIFT_MU = 0
SHIFT_SIGMA = 3


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
        )
    ]
