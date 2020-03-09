
import sys
import os
import shutil
import itertools
import argparse

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from jpeg import Jpeg
from ycbcr import ycbcr_to_rgb


def iter_channels(ycbcr_img_data):
    channel_names = ['Luminance', 'Blue Chrominance', 'Red Chrominance']
    for channel_index, channel_name in enumerate(channel_names):
        channel_data = ycbcr_img_data[:, :, channel_index]
        yield channel_index, channel_name, channel_data


def get_image_data(path):
    data = imageio.imread(path)
    xdim, ydim, num_channels = data.shape

    if num_channels < 3:
        raise RuntimeError('Image must have all three color channels.')
    if xdim % 8 != 0 or ydim % 8 != 0:
        raise RuntimeError('Image dimensions must be a multiple of 8')

    # Remove alpha channel, if one exists
    return data[:, :, :3]


def run(args):
    # Read in RGB data and convert to YCbCr
    print('Reading source image')
    rgb_img_data = get_image_data(args.img_path)

    print('Encoding data as JPEG')
    jpeg = Jpeg.from_rgb(rgb_img_data, args.quality)

    ycbcr_img_data = jpeg.ycbcr_data
    quantized_ycbcr_img_data = jpeg.get_quantized_ycbcr_data()
    final_rgb_img_data = ycbcr_to_rgb(quantized_ycbcr_img_data)

    # Clean output directory
    output_dir = f'output/quality{args.quality}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    copy_source_image(args.img_path, output_dir)
    save_split_ycbcr_plots('source', ycbcr_img_data, output_dir)
    save_split_ycbcr_plots('quantized', quantized_ycbcr_img_data, output_dir)
    save_difference_plots(ycbcr_img_data, quantized_ycbcr_img_data, output_dir)
    save_comparison_plots(rgb_img_data, final_rgb_img_data, output_dir)
    save_output_image(final_rgb_img_data, output_dir)

    print('Done!')


def copy_source_image(img_path, output_dir):
    _, image_ext = os.path.splitext(img_path)
    dest_path = os.path.join(output_dir, 'source_image' + image_ext)
    shutil.copyfile(img_path, dest_path)


def save_split_ycbcr_plots(name, ycbcr_img_data, output_dir):
    """Split each channel into its own plot.

    Dark areas indicate a lack of that color component (or overall darkness,
    for luminance), while lighter areas indicate an abundance of that color
    component (or an overall bright image, for luminance).

    """
    print(f'Saving split YCbCr plots ({name})')
    fig = plt.figure(figsize=(16, 6))

    for channel_index, channel_name, channel_data in iter_channels(ycbcr_img_data):
        ax = plt.subplot(1, 3, channel_index + 1)
        ax.set_title(channel_name)
        plt.imshow(channel_data, cmap='Greys_r')

    filename = os.path.join(output_dir, f'{name}_split_ycbcr_plots.png')
    plt.savefig(filename)


def save_difference_plots(ycbcr_img_data, quantized_ycbcr_img_data, output_dir):
    """Show differences between the original and quantized image channel data."""
    print('Saving difference plots')
    def plot_difference(index, channel_name, original, quantized):
        fmts = ['Original {}', 'Quantized {}', 'Difference']
        data_sets = [original, quantized, np.abs(original - quantized)]
        for i, (fmt, data) in enumerate(zip(fmts, data_sets)):
            ax = plt.subplot(3, 3, 3 * index + i + 1)
            ax.set_title(fmt.format(channel_name))
            plt.imshow(data, cmap='Greys_r')

    fig = plt.figure(figsize=(16, 18))
    for channel_index, channel_name, channel_data in iter_channels(ycbcr_img_data):
        quantized_channel_data = quantized_ycbcr_img_data[:, :, channel_index]
        plot_difference(channel_index, channel_name, channel_data, quantized_channel_data)

    filename = os.path.join(output_dir, 'difference_plots.png')
    plt.savefig(filename)


def save_comparison_plots(rgb_img_data, final_rgb_img_data, output_dir, slice_coords=None):
    fig = plt.figure(figsize=(16, 6))

    if slice_coords is None:
        print('Saving comparison plots (all)')
        sample_slice = np.index_exp[:, :]
        interpolation = None
        filename_suffix = 'all'
    else:
        print('Saving comparison plots (partial)')
        slice_x, slice_y = slice_coords
        sample_slice = np.index_exp[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1]]
        interpolation = 'nearest'
        filename_suffix = '_{}_to_{}__and__{}_to_{}'.format(*slice_x, *slice_y)

    ax = plt.subplot(1, 2, 1)
    ax.set_title('Original Image')
    plt.imshow(rgb_img_data[sample_slice], interpolation=interpolation)

    ax = plt.subplot(1, 2, 2)
    ax.set_title('JPEG Compressed Image')
    plt.imshow(final_rgb_img_data[sample_slice], interpolation=interpolation)

    filename = os.path.join(output_dir, 'comparison_plots__{}.png'.format(filename_suffix))
    plt.savefig(filename)


def save_output_image(rgb_img_data, output_dir):
    print('Saving output image')
    filename = os.path.join(output_dir, 'output_image.png')
    imageio.imsave(filename, rgb_img_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('quality', type=int)
    # parser.add_argument('output_dir')  # TODO
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
