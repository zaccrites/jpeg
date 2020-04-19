
# JPEG Encoder

A JPEG encoder implemented in Python

![Q=20 Comparison](images/quality20/comparison_plots__all.png "Q=20 Comparison")

The script does not generate usable `.jpg` files in truth,
but it will perform the transforms needed to convert e.g. a PNG
image to JPEG at a requested quality level, showing the difference
in each luma and chroma channel from the original source image
and the JPEG-ified output.

## Encoding Steps

1. Convert image RGB pixel values to the [YCbCr](https://en.wikipedia.org/wiki/YCbCr) (luma+chroma) color space.
2. Compute the [discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) to find coefficients for high and low frequency content in 8x8 pixel blocks.
3. Compute a quantization table using a specified quality value, then divide each coefficient by the corresponding entry in the table. Divisors increase in magnitude for higher frequency image content.
4. Round resulting quotients to the nearest integer. This typically rounds high frequency coefficients to zero. Multiply by the original divisor to get the quantized coefficients, many of which will now be zero.


## Decoding Steps

1. Perform an inverse discrete cosine transform to obtain quantized YCbCr pixel values.
2. Convert YCbCr color data to RGB to display the image.


## Compression

The strength of the JPEG encoding process lies in the quantization and removal of high frequency content.
High frequency content is less noticeable than low frequency content,
so it can be removed with little noticeable impact on many images
(notably [photographs](https://en.wikipedia.org/wiki/Joint_Photographic_Experts_Group)).
By employing [run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding) and traversing DCT coefficients from lowest frequency to highest, a long string of zeroes in the high frequency section may be represented compactly. Applying a further layer of [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding) can further compress the data.


# Results

## Quality = 90

![Q=90 Output](images/quality90/output_image.png "Q=90 Output")

![Q=90 Comparison](images/quality90/comparison_plots__all.png "Q=90 Output")

![Q=90 Difference Plots](images/quality90/difference_plots.png "Q=90 Difference Plots")

## Quality = 50

![Q=50 Output](images/quality50/output_image.png "Q=50 Output")

![Q=50 Comparison](images/quality50/comparison_plots__all.png "Q=50 Output")

![Q=50 Difference Plots](images/quality50/difference_plots.png "Q=50 Difference Plots")

## Quality = 20

![Q=20 Output](images/quality20/output_image.png "Q=20 Output")

![Q=20 Comparison](images/quality20/comparison_plots__all.png "Q=20 Output")

![Q=20 Difference Plots](images/quality20/difference_plots.png "Q=20 Difference Plots")

## Quality = 10

![Q=10 Output](images/quality10/output_image.png "Q=10 Output")

![Q=10 Comparison](images/quality10/comparison_plots__all.png "Q=10 Output")

![Q=10 Difference Plots](images/quality10/difference_plots.png "Q=10 Difference Plots")
