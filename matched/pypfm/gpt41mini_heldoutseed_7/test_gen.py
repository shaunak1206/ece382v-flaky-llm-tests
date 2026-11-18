import io
import os
import sys
import tempfile
import pytest
from pypfm import PFMLoader

# Since pypfm.py source was requested but not provided,
# tests are written assuming typical behavior of a PFM loader:
# - loading PFM files in little and big endian
# - loading both color and grayscale
# - handling invalid files and inputs properly

def create_pfm_file(contents: bytes) -> str:
    """Helper to write bytes to a temporary file and return its name."""
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.write(contents)
    tf.flush()
    tf.close()
    return tf.name

def test_load_valid_little_endian_color_pfm():
    # Example minimal PFM file: header + scale + 1x1 pixel RGB float data (little endian)
    contents = (
        b'PF\n'          # color image
        b'1 1\n'         # width height
        b'-1.0\n'        # scale (negative for little endian)
        + b'\x00\x00\x80?' * 3  # RGB pixel = (1.0, 1.0, 1.0) as float32 little endian
    )
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    image, scale = loader.load(filename)
    os.remove(filename)

    assert image.shape == (1, 1, 3)
    assert pytest.approx(image[0,0,0], 1e-6) == 1.0
    assert pytest.approx(scale, 1e-6) == -1.0

def test_load_valid_big_endian_grayscale_pfm():
    # Example PFM grayscale file with big endian
    contents = (
        b'Pf\n'          # grayscale image
        b'1 1\n'         # width height
        b'1.0\n'         # scale (positive for big endian)
        + b'?\x80\x00\x00'  # 1.0 float32 big endian
    )
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    image, scale = loader.load(filename)
    os.remove(filename)

    assert image.shape == (1, 1)
    assert pytest.approx(image[0,0], 1e-6) == 1.0
    assert pytest.approx(scale, 1e-6) == 1.0

def test_load_invalid_header_raises():
    contents = b'PZ\n1 1\n1.0\n' + b'\x00'*4
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(filename)
    os.remove(filename)

def test_load_invalid_dimensions_raises():
    contents = b'Pf\nx y\n1.0\n' + b'\x00'*4
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(filename)
    os.remove(filename)

def test_load_invalid_scale_raises():
    contents = b'PF\n1 1\nabc\n' + b'\x00'*12
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(filename)
    os.remove(filename)

def test_load_short_data_raises():
    # Header valid but not enough data bytes for image
    contents = b'PF\n1 1\n-1.0\n' + b'\x00\x00\x80?' * 2  # Missing 1 float (3 needed)
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(filename)
    os.remove(filename)

def test_load_nonexistent_file_raises():
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load("this_file_does_not_exist.pfm")

def test_load_file_object_instead_of_path():
    # Using a file-like object should raise or be unsupported
    contents = (
        b'PF\n1 1\n-1.0\n' + b'\x00\x00\x80?' * 3
    )
    fileobj = io.BytesIO(contents)
    loader = PFMLoader()
    with pytest.raises((AttributeError, TypeError)):
        loader.load(fileobj)

def test_load_image_data_is_float32():
    contents = (
        b'PF\n1 1\n-1.0\n' + b'\x00\x00\x00@'*3  # RGB with (2.0, 2.0, 2.0)
    )
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    image, _ = loader.load(filename)
    os.remove(filename)
    assert image.dtype.name == "float32"

def test_version_is_correct():
    # The version string is from __init__.py
    import pypfm
    assert pypfm.__version__ == "1.0.0"

def test_load_realistic_3x2_color_image():
    # Minimal test for multiple pixels
    header = b'PF\n3 2\n-1.0\n'
    # 3*2*3 float32 pixels little endian = 18 float32 values
    # We'll fill with increasing values starting from 0.0, 1.0, ...
    import struct
    pixels = b''.join(struct.pack('<f', float(i)) for i in range(18))
    contents = header + pixels
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    img, scale = loader.load(filename)
    os.remove(filename)

    assert img.shape == (2, 3, 3)
    assert pytest.approx(img[0,0,0], 1e-6) == 0.0
    assert pytest.approx(img[1,2,2], 1e-6) == 17.0
    assert scale == -1.0

def test_load_realistic_2x3_grayscale_image():
    header = b'Pf\n2 3\n1.0\n'
    import struct
    pixels = b''.join(struct.pack('>f', float(i)) for i in range(6))
    contents = header + pixels
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    img, scale = loader.load(filename)
    os.remove(filename)

    assert img.shape == (3, 2)
    assert pytest.approx(img[0,0], 1e-6) == 0.0
    assert pytest.approx(img[2,1], 1e-6) == 5.0
    assert scale == 1.0

def test_load_grayscale_image_returns_2d_array():
    contents = (
        b'Pf\n1 1\n-1.0\n' + b'\x00\x00\x80?'
    )
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    img, _ = loader.load(filename)
    os.remove(filename)

    assert img.ndim == 2

def test_load_color_image_returns_3d_array_with_3_channels():
    contents = (
        b'PF\n1 1\n-1.0\n' + b'\x00\x00\x80?'*3
    )
    filename = create_pfm_file(contents)
    loader = PFMLoader()
    img, _ = loader.load(filename)
    os.remove(filename)

    assert img.ndim == 3
    assert img.shape[2] == 3

def test_loading_empty_file_raises():
    filename = create_pfm_file(b'')
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(filename)
    os.remove(filename)
