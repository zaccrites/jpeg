"""Run-length encoding."""

import itertools
from collections import namedtuple

from jpeg import iterate_blocks


Run = namedtuple('Run', ['value', 'length'])


def run_length_encode(data):
    for key, values in itertools.groupby(data):
        yield Run(value=key, length=len(list(values)))


def compress_image_data(channel_data):
    # Zig-zag pattern starting from top-left
    # https://en.wikipedia.org/wiki/JPEG#Entropy_coding

    # Since the blocks are always 8x8, we can hard-code the traversal order
    traversal_order = [
        0,
        1, 8,
        16, 9, 2,
        3, 10, 17, 24,
        32, 25, 18, 11, 4,
        5, 12, 19, 26, 33, 40,
        48, 41, 34, 27, 20, 13, 6,
        7, 14, 21, 28, 35, 42, 49, 56,
        57, 50, 43, 36, 29, 22, 15,
        23, 30, 37, 44, 51, 58,
        59, 52, 45, 38, 31,
        39, 46, 53, 60,
        61, 54, 47,
        55, 62,
        63
    ]
    assert list(sorted(traversal_order)) == list(range(64))

    for block_slice, block_data in iterate_blocks(channel_data):
        # Linearize the block data for easier indexing
        block_data = block_data.reshape(64)
        pixel_data = (block_data[i] for i in traversal_order)
        yield from run_length_encode(pixel_data)
