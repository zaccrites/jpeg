"""Convert between RGB and YCbCr color spaces."""

import numpy as np


def rgb_to_ycbcr(img_rgb_data):
    # Transformation procedure derived from chapter 3 of
    # "Video Demystified: A Handbook for the Digital Engineer"
    # (section "YCbCr Color Space - Computer System Considerations")
    T = np.array([
        [ 0.183,  0.614,  0.062],
        [-0.101, -0.338,  0.439],
        [ 0.439, -0.399, -0.040],
    ])

    xdim, ydim, num_channels = img_rgb_data.shape
    img_ycbcr_data = np.copy(img_rgb_data)

    img_ycbcr_data = img_ycbcr_data.reshape((xdim * ydim, num_channels))
    img_ycbcr_data = np.transpose(img_ycbcr_data)
    img_ycbcr_data = T @ img_ycbcr_data
    img_ycbcr_data = np.transpose(img_ycbcr_data)
    img_ycbcr_data = img_ycbcr_data.reshape(img_rgb_data.shape)

    img_ycbcr_data = img_ycbcr_data + [16, 128, 128]
    img_ycbcr_data = img_ycbcr_data.clip(0, 255)
    return img_ycbcr_data.astype('uint8')


def ycbcr_to_rgb(img_ycbcr_data):
    T = np.array([
        [1.164,  0.000,  1.793],
        [1.164, -0.213, -0.534],
        [1.164,  2.115,  0.000],
    ])

    xdim, ydim, num_channels = img_ycbcr_data.shape
    img_rgb_data = np.copy(img_ycbcr_data)
    img_rgb_data = img_rgb_data - [16, 128, 128]

    img_rgb_data = img_rgb_data.reshape((xdim * ydim, num_channels))
    img_rgb_data = np.transpose(img_rgb_data)
    img_rgb_data = T @ img_rgb_data
    img_rgb_data = np.transpose(img_rgb_data)
    img_rgb_data = img_rgb_data.reshape(img_ycbcr_data.shape)

    img_rgb_data = img_rgb_data.clip(0, 255)
    return img_rgb_data.astype('uint8')
