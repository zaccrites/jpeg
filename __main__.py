
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
    # ycbcr_img_data = rgb_to_ycbcr(rgb_img_data)

    print('Encoding data as JPEG')
    jpeg = Jpeg.from_rgb(rgb_img_data, args.quality)


    from ycbcr import ycbcr_to_rgb
    ycbcr_img_data = jpeg.ycbcr_data
    quantized_ycbcr_img_data = jpeg.get_quantized_ycbcr_data()
    final_rgb_img_data = ycbcr_to_rgb(quantized_ycbcr_img_data)


    # print('Creating quantization tables')
    # luminance_quantization_table = create_quantization_table(args.quality, QuantizationTableType.luminance)
    # chrominance_quantization_table = create_quantization_table(args.quality, QuantizationTableType.chrominance)

    # # Quantize YCbCr data
    # print(f'Quantizing YCbCr data at {args.quality}% quality')
    # quantized_ycbcr_img_data = np.empty_like(ycbcr_img_data)
    # channel_quantization_tables = [luminance_quantization_table, chrominance_quantization_table, chrominance_quantization_table]
    # for i, quantization_table in enumerate(channel_quantization_tables):
    #     channel_data = ycbcr_img_data[:, :, i]
    #     quantized_ycbcr_img_data[:, :, i] = quantize(channel_data, quantization_table)

    # # Convert back to RGB and compare with the original image
    # print('Converting back to RGB')
    # final_rgb_img_data = ycbcr_to_rgb(quantized_ycbcr_img_data)

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
    # save_comparison_plots(rgb_img_data, final_rgb_img_data, output_dir, slice_coords=((0,200),   (0,200)))
    # save_comparison_plots(rgb_img_data, final_rgb_img_data, output_dir, slice_coords=((150,250), (250,350)))

    # y_img_data = ycbcr_img_data[:, :, 0]
    # save_quantization_process_steps(y_img_data, luminance_quantization_table, output_dir)

    save_output_image(final_rgb_img_data, output_dir)

    # save_compression_stats(quantized_ycbcr_img_data, output_dir)

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


# def save_compression_stats(quantized_ycbcr_img_data, output_dir):
#     print('Computing compression stats')
#     filename = os.path.join(output_dir, 'compression_stats.txt')
#     with open(filename, 'w') as f:

#         # Uncompressed, the file requires one byte per channel per pixel.
#         uncompressed_size = 3 * quantized_ycbcr_img_data.size
#         fmt = 'Uncompressed Size: {:.2f} kB \n'
#         print(fmt.format(uncompressed_size / 1024), file=f)

#         run_length_encoded_size = 0
#         huffman_coded_size = 0

#         for channel_index, channel_name, channel_data in iter_channels(quantized_ycbcr_img_data):
#             # For run-length encoding, each run requires one byte for the value
#             # and one byte for the length of the run.
#             run_length_encoded_data = []
#             for run in compress_image_data(channel_data):
#                 run_length_encoded_data.append(run.value)
#                 run_length_encoded_data.append(run.length)
#             run_length_encoded_size += len(run_length_encoded_data)

#             # Huffman coding goes further, assigning a variable-length code
#             # to each value. More common values will receive shorter codes,
#             # which can save space.
#             huffman_code_table, huffman_coded_data = huffman_encode(run_length_encoded_data)

#             # Each Huffman-coded value takes up only as many bits as the code used
#             # to represent it.
#             huffman_coded_size += sum(len(encoded_value) for encoded_value in huffman_coded_data) / 8

#             # The total Huffman encoding size must store the code table with it.
#             # Each entry in the table requires one byte for the associated value,
#             # one byte for the length of the code in bits, and then the code itself.
#             huffman_code_table_size = 2 * len(huffman_code_table)
#             for value, huffman_code in huffman_code_table.items():
#                 huffman_code_table_size += len(huffman_code) / 8
#             huffman_coded_size += huffman_code_table_size

#         fmt = '{} Encoded Size: {:.2f} kB   (compression ratio = {:.3f})'

#         compression_ratio = uncompressed_size / run_length_encoded_size
#         print(fmt.format('Run-length', run_length_encoded_size / 1024, compression_ratio), file=f)

#         compression_ratio = uncompressed_size / huffman_coded_size
#         print(fmt.format('   Huffman', huffman_coded_size / 1024, compression_ratio), file=f)


def scale_block(block_data, scale_factor):
    xres, yres = block_data.shape
    large_block_data = np.empty((xres * scale_factor, yres * scale_factor))
    for x, y in itertools.product(range(xres), range(yres)):
        large_block_data[(x*scale_factor):((x+1)*scale_factor), (y*scale_factor):((y+1)*scale_factor)] = block_data[x, y]
    return large_block_data


# def save_quantization_process_steps(channel_data, quantization_table, output_dir):
#     print('Saving quantization process steps')
#     block_data = channel_data[136:136+8, 198:198+8]
#     quantizer = BlockQuantizer(block_data, quantization_table)

#     # Save sample input image
#     SCALE_FACTOR = 64
#     filename = os.path.join(output_dir, 'quantization_process_steps_input_image.png')
#     imageio.imsave(filename, scale_block(block_data, SCALE_FACTOR))

#     # Setup numpy printing
#     def float_formatter(x):
#         raw = '{:.2f}'.format(x)
#         return raw.rjust(len('-999.99'))
#     np.set_printoptions(formatter={'float_kind': float_formatter})

#     steps_to_write = [
#         ('Input Block', block_data),
#         ('Shifted Block', quantizer.shifted_block),
#         ('DCT-II Coefficients', quantizer.dct_block),
#         ('Quantization Table', quantization_table),
#         ('Divided Coefficients', quantizer.divided_block),
#         ('Quantized', quantizer.rounded_block),
#         ('DCT-III Coefficients', quantizer.multiplied_block),
#         ('Output Shifted Block', quantizer.idct_block),
#         ('Output Block', quantizer.quantized_block),
#     ]

#     filename = os.path.join(output_dir, 'quantization_process_steps.txt')
#     with open(filename, 'w') as f:
#         for name, data in steps_to_write:
#             print(name, file=f)
#             print(data, file=f)
#             print('\n', file=f)

#     # Save sample output image
#     filename = os.path.join(output_dir, 'quantization_process_steps_output_image.png')
#     imageio.imsave(filename, scale_block(quantizer.quantized_block, SCALE_FACTOR))


# From the output directory:
# python ..\implementation ..\images\Pluto.png <quality>

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('quality', type=int)
    # parser.add_argument('output_dir')  # TODO
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
