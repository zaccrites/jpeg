
import numpy as np


def rgb_to_ycbcr(img_rgb_data):
    # Equations from the Chapter 3 book (RGB - YCbCr Equations: Computer System Considerations)
    T = np.array([
        # SDTV
        # [ 0.299,  0.587,  0.114],
        # [-0.172, -0.339,  0.511],
        # [ 0.511, -0.428, -0.083],

        # HDTV
        # [ 0.213,  0.715,  0.072],
        # [-0.117, -0.394,  0.511],
        # [ 0.511, -0.464, -0.047],

        [ 0.183,  0.614,  0.062],
        [-0.101, -0.338,  0.439],
        [ 0.439, -0.399, -0.040],
    ])

    # TODO: There is probably a more elegant way to do this:
    img_ycbcr_data = np.empty_like(img_rgb_data)
    for x in range(img_rgb_data.shape[0]):
        for y in range(img_rgb_data.shape[1]):
            rgb_px_data = img_rgb_data[x, y, :]
            img_ycbcr_data[x, y, :] = np.clip((T @ rgb_px_data) + [16, 128, 128], 0, 255)
            # img_ycbcr_data[x, y, :] = (T @ rgb_px_data) + [16, 128, 128]
    return img_ycbcr_data


def ycbcr_to_rgb(img_ycbcr_data):
    T = np.array([
        [1.164,  0.000,  1.793],
        [1.164, -0.213, -0.534],
        [1.164,  2.115,  0.000],
    ])

    # TODO: There is probably a more elegant way to do this:
    img_rgb_data = np.empty_like(img_ycbcr_data)
    for x in range(img_ycbcr_data.shape[0]):
        for y in range(img_ycbcr_data.shape[1]):
            ycbcr_px_data = img_ycbcr_data[x, y, :]
            img_rgb_data[x, y, :] = np.clip(T @ (ycbcr_px_data - [16, 128, 128]), 0, 255)
            # img_rgb_data[x, y, :] = T @ (ycbcr_px_data - [16, 128, 128])
    return img_rgb_data
