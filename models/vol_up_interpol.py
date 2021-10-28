import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.layers.base import InputSpec
# stolen from https://github.com/tensorflow/tensorflow/issues/46609#issuecomment-774573667


def linear_interpolate(x_fix, y_fix, x_var):
    '''
        Functionality:
            1D linear interpolation
        Author:
            Michael Osthege
        Link:
            https://gist.github.com/michaelosthege/e20d242bc62a434843b586c78ebce6cc
    '''

    x_repeat = np.tile(x_var[:, None], (len(x_fix),))
    distances = np.abs(x_repeat - x_fix)

    x_indices = np.searchsorted(x_fix, x_var)

    weights = np.zeros_like(distances)
    idx = np.arange(len(x_indices))
    weights[idx, x_indices] = distances[idx, x_indices - 1]
    weights[idx, x_indices - 1] = distances[idx, x_indices]
    weights /= np.sum(weights, axis=1)[:, None] + 10e-7

    y_var = np.dot(weights, y_fix.T)

    return y_var


def Interpolate1D(x, y, xx, method='nearest'):
    '''
        Functionality:
            1D interpolation with various methods
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    n = len(x)
    nn = len(xx)
    yy = np.zeros(nn)

    # Nearest neighbour interpolation
    if method == 'nearest':
        for i in range(0, nn):
            xi = np.abs(xx[i] - x).argmin()
            yy[i] = y[xi]

    # Linear interpolation
    elif method == 'linear':

        # # slower version
        # if n == 1:
        #     yy[:-1] = y[0]

        # else:
        #     for i in range(0, nn):

        #         if xx[i] < x[0]:
        #             t = (xx[i] - x[0]) / (x[1] - x[0])
        #             yy[i] = (1.0 - t) * y[0] + t * y[1]

        #         elif x[n - 1] <= xx[i]:
        #             t = (xx[i] - x[n - 2]) / (x[n - 1] - x[n - 2])
        #             yy[i] = (1.0 - t) * y[n - 2] + t * y[n - 1]

        #         else:
        #             for k in range(1, n):
        #                 if x[k - 1] <= xx[i] and xx[i] < x[k]:
        #                     t = (xx[i] - x[k - 1]) / (x[k] - x[k - 1])
        #                     yy[i] = (1.0 - t) * y[k - 1] + t * y[k]
        #                     break

        # # faster version
        yy = linear_interpolate(x, y, xx)

    return yy


def Interpolate3D(x, y, z, f, xx, yy, zz, method='nearest'):
    '''
        Functionality:
            3D interpolation implemented in a separable fashion
            There are methods that do real 3D non-separable interpolation, which are
                more difficult to implement.
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    n1 = len(x)
    n2 = len(y)
    n3 = len(z)
    nn1 = len(xx)
    nn2 = len(yy)
    nn3 = len(zz)

    w1 = np.zeros((nn1, n2, n3))
    w2 = np.zeros((nn1, nn2, n3))
    ff = np.zeros((nn1, nn2, nn3))

    # Interpolate along the 1st dimension
    for k in range(0, n3):
        for j in range(0, n2):
            w1[:, j, k] = Interpolate1D(x, f[:, j, k], xx, method)

    # Interpolate along the 2nd dimension
    for k in range(0, n3):
        for i in range(0, nn1):
            w2[i, :, k] = Interpolate1D(y, w1[i, :, k], yy, method)

    # Interpolate along the 3rd dimension
    for j in range(0, nn2):
        for i in range(0, nn1):
            ff[i, j, :] = Interpolate1D(z, w2[i, j, :], zz, method)

    return ff


def UpInterpolate3D(x,
                    size=(2, 2, 2),
                    interpolation='nearest',
                    data_format='channels_first',
                    align_corners=True):
    '''
        Functionality:
            3D upsampling interpolation for tf
        Author:
            Kai Gao <nebulaekg@gmail.com>
    '''

    x = x.numpy()

    if data_format == 'channels_last':
        nb, nr, nc, nd, nh = x.shape
    elif data_format == 'channels_first':
        nb, nh, nr, nc, nd = x.shape

    r = size[0]
    c = size[1]
    d = size[2]
    ir = np.linspace(0.0, nr - 1.0, num=nr)
    ic = np.linspace(0.0, nc - 1.0, num=nc)
    id = np.linspace(0.0, nd - 1.0, num=nd)

    if align_corners:
        # align_corners=True assumes that values are sampled at discrete points
        iir = np.linspace(0.0, nr - 1.0, num=nr * r)
        iic = np.linspace(0.0, nc - 1.0, num=nc * c)
        iid = np.linspace(0.0, nd - 1.0, num=nd * d)
    else:
        # aling_corners=False assumes that values are sampled at centers of discrete blocks
        iir = np.linspace(0.0 - 0.5 + 0.5 / r, nr - 1.0 + 0.5 - 0.5 / r, num=nr * r)
        iic = np.linspace(0.0 - 0.5 + 0.5 / c, nc - 1.0 + 0.5 - 0.5 / c, num=nc * c)
        iid = np.linspace(0.0 - 0.5 + 0.5 / d, nd - 1.0 + 0.5 - 0.5 / d, num=nd * d)
        iir = np.clip(iir, 0.0, nr - 1.0)
        iic = np.clip(iic, 0.0, nc - 1.0)
        iid = np.clip(iid, 0.0, nd - 1.0)

    if data_format == 'channels_last':
        xx = np.zeros((nb, nr * r, nc * c, nd * d, nh))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, :, :, :, j], (nr, nc, nd))
                xx[i, :, :, :, j] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)

    elif data_format == 'channels_first':
        xx = np.zeros((nb, nh, nr * r, nc * c, nd * d))
        for i in range(0, nb):
            for j in range(0, nh):
                t = np.reshape(x[i, j, :, :, :], (nr, nc, nd))
                xx[i, j, :, :, :] = Interpolate3D(ir, ic, id, t, iir, iic, iid, interpolation)

    return tf.convert_to_tensor(xx, dtype=x.dtype)


class UpSampling3D(tf.keras.layers.Layer):
    def __init__(self,
                 size=(2, 2, 2),
                 data_format=None,
                 interpolation='nearest',
                 align_corners=True,
                 **kwargs):
        super(UpSampling3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)
        self.interpolation = interpolation
        if interpolation not in {'nearest', 'trilinear', 'linear'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                             'or `"trilinear"` '
                             'or `"linear"` ')
        if self.interpolation == 'trilinear':
            self.interpolation = 'linear'
        self.align_corners = align_corners

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            dim1 = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            dim2 = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            dim3 = self.size[2] * input_shape[4] if input_shape[4] is not None else None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], dim1, dim2, dim3])
        else:
            dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
            return tensor_shape.TensorShape([input_shape[0], dim1, dim2, dim3, input_shape[4]])

    def call(self, inputs):
        return UpInterpolate3D(inputs,
                               self.size,
                               data_format=self.data_format,
                               interpolation=self.interpolation,
                               align_corners=self.align_corners)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
