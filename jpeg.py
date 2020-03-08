
from enum import Enum

import numpy as np
from scipy.fftpack import dct, idct


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
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


class BlockQuantizer(object):

    def __init__(self, block_data, quantization_table):
        self.block_data = block_data
        self.quantization_table = quantization_table

    @property
    def shifted_block(self):
        # Make sure that it is signed!
        return self.block_data.astype(int) - 128

    @property
    def dct_block(self):
        return dct2(self.shifted_block.astype(float))

    @property
    def divided_block(self):
        # return self.dct_block
        return np.divide(self.dct_block, self.quantization_table)

    @property
    def rounded_block(self):
        return self.divided_block.round().astype(int)

    @property
    def multiplied_block(self):
        # return self.idct_block
        return np.multiply(self.rounded_block, self.quantization_table)

    @property
    def idct_block(self):
        return idct2(self.multiplied_block.astype(float))

    @property
    def unshifted_block(self):
        return self.idct_block + 128

    @property
    def quantized_block(self):
        return self.unshifted_block.astype(int)

    def quantize(self):
        return self.quantized_block


# From JPEG standard, IEC 10918-1
BASE_LUMINANCE_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 124, 140, 151, 161],
    [12, 12, 14, 19, 126, 158, 160, 155],
    [14, 13, 16, 24, 140, 157, 169, 156],
    [14, 17, 22, 29, 151, 187, 180, 162],
    [18, 22, 37, 56, 168, 109, 103, 177],
    [24, 35, 55, 64, 181, 104, 113, 192],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 199],
])
BASE_CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
])


class QuantizationTableType(Enum):
    luminance = 1
    chrominance = 2


def create_quantization_table(quality, table_type):
    # From "RTP Payload Format for JPEG-compressed video" section 4.2, rfc2435
    # quality must be between 1 and 99
    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality

    # The scale factor is a percentage. At quality=50%, the quantization table is used as given.
    base_table = {
        QuantizationTableType.luminance: BASE_LUMINANCE_QUANTIZATION_TABLE,
        QuantizationTableType.chrominance: BASE_CHROMINANCE_QUANTIZATION_TABLE,
    }[table_type]
    return base_table * (scale_factor / 100)


def quantize(channel_data, quantization_table):
    quantized_data = np.empty_like(channel_data)
    for block_slice, block_data in iterate_blocks(channel_data):
        quantizer = BlockQuantizer(block_data, quantization_table)
        quantized_data[block_slice] = quantizer.quantize()
    return quantized_data
