"""Perform JPEG compression steps."""

import struct
from enum import Enum
from collections import namedtuple

import numpy as np
from scipy.fftpack import dct, idct

from ycbcr import rgb_to_ycbcr, ycbcr_to_rgb


class QuantizationTable(object):

    def __init__(self, coefficients):
        if coefficients.shape != (8, 8):
            raise ValueError('Table must be of shape 8x8')
        self.coefficients = coefficients

    def quantize(self, block_data):
        divided_block = np.divide(block_data, self.coefficients)
        rounded_block = divided_block.round().astype(int)
        return np.multiply(rounded_block, self.coefficients)

    @classmethod
    def _create(cls, base_table, quality):
        # From "RTP Payload Format for JPEG-compressed video" section 4.2, rfc2435
        # quality must be between 1 and 99
        if quality < 50:
            scale_factor = 5000 / quality
        else:
            scale_factor = 200 - 2 * quality
        # The scale factor is a percentage.
        # At quality=50%, the quantization table is used as given.
        coefficients = base_table * (scale_factor / 100)
        return cls(coefficients)

    @classmethod
    def create_luma(cls, quality):
        # From JPEG standard, IEC 10918-1
        base_table = np.array([
            [16, 11, 10, 16, 124, 140, 151, 161],
            [12, 12, 14, 19, 126, 158, 160, 155],
            [14, 13, 16, 24, 140, 157, 169, 156],
            [14, 17, 22, 29, 151, 187, 180, 162],
            [18, 22, 37, 56, 168, 109, 103, 177],
            [24, 35, 55, 64, 181, 104, 113, 192],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 199],
        ])
        return cls._create(base_table, quality)

    @classmethod
    def create_chroma(cls, quality):
        # From JPEG standard, IEC 10918-1
        base_table = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ])
        return cls._create(base_table, quality)


class Jpeg(object):

    def __init__(self, ycbcr_data, luma_table, chroma_table):
        self.luma_table = luma_table
        self.chroma_table = chroma_table
        self.ycbcr_data = ycbcr_data

    @classmethod
    def from_rgb(cls, rgb_data, quality):
        luma_table = QuantizationTable.create_luma(quality)
        chroma_table = QuantizationTable.create_chroma(quality)
        ycbcr_data = rgb_to_ycbcr(rgb_data)
        return cls(ycbcr_data, luma_table, chroma_table)

    def get_quantized_ycbcr_data(self):
        quantized_data = np.empty_like(self.ycbcr_data)
        tables = [self.luma_table, self.chroma_table, self.chroma_table]
        for i, quantization_table in enumerate(tables):
            channel_data = self.ycbcr_data[:, :, i]
            quantized_data[:, :, i] = quantize_channel(channel_data, quantization_table)
        return quantized_data

    @property
    def compressed_size(self):
        # TODO: Calculate run-length encoding size.
        # FUTURE: Add Huffman coding to further compress
        raise NotImplementedError


def quantize_channel(channel_data, quantization_table):
    quantized_data = np.empty_like(channel_data)
    for block_slice, block_data in iterate_blocks(channel_data):
        # Shift [0,255] pixel data to signed [-128,127]
        shifted_block = block_data.astype(int) - 128
        # Compute the 2D discrete cosine transform to get frequency domain coefficients
        dct_block = dct2(shifted_block.astype(float))
        # Quantize frequency coefficients according to the table
        quantized_block = quantization_table.quantize(dct_block)
        # Compute inverse DCT to get back signed pixel data
        idct_block = idct2(quantized_block.astype(float))
        # Shift back into unsigned range
        unshifted_block = idct_block.astype(int) + 128
        quantized_data[block_slice] = unshifted_block
    return quantized_data


def iterate_blocks(channel_data):
    """Iterate over the channel data in 8x8 blocks."""
    xres, yres = channel_data.shape
    assert xres % 8 == 0
    assert yres % 8 == 0

    for block_x in range(xres // 8):
        for block_y in range(yres // 8):
            block_slice = np.index_exp[(block_x)*8:(block_x+1)*8, (block_y)*8:(block_y+1)*8]
            block_data = channel_data[block_slice]
            yield block_slice, block_data


# http://stackoverflow.com/questions/34890585/in-scipy-why-doesnt-idctdcta-equal-to-a
def dct2(block):
    """Compute a 2D discrete cosine transform."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    """Compute a 2D inverse discrete cosine transform."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')
