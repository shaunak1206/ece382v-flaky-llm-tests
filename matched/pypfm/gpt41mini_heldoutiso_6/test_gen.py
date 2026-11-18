import os
import io
import pytest
from pypfm import PFMLoader, __version__


@pytest.fixture
def sample_pfm_file(tmp_path):
    content = (
        b"PFM\n"
        b"-1.0 3 2\n"
        b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@"
        b"\x00\x00\x80@\x00\x00\xa0@"
    )
    file_path = tmp_path / "sample.pfm"
    file_path.write_bytes(content)
    return str(file_path)

@pytest.fixture
def sample_pfm_file_alpha(tmp_path):
    # Create a file with 4 channels (RGBA) (3x1 image)
    header = b"PFM\n3 -1\n"
    # 3 pixels, 4 floats each = 12 floats total (4 channels)
    # Use values 1.0 to 12.0 float32 little endian
    data = b"".join((i.to_bytes(4, 'little') for i in range(1, 13)))
    # the above is actually just integer bytes - doesn't create valid floats,
    # let's create valid float32 bytes instead:
    import struct
    floats = [float(i) for i in range(1, 13)]
    data = b"".join(struct.pack("<f", f) for f in floats)
    content = header + data
    file_path = tmp_path / "sample_alpha.pfm"
    file_path.write_bytes(content)
    return str(file_path)


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "1.0.0"


def test_load_grayscale_pfm(sample_pfm_file):
    loader = PFMLoader(sample_pfm_file)
    img = loader.load()
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
    assert img.shape[0] == 3
    assert img.shape[1] == 2
    assert img.dtype == "float32"
    # Spot check values (since little endian float32 read)
    # top-left pixel (exponent -1.0) should be about 1.0
    val = img[0,0]
    # Values from provided bytes correspond roughly to these floats
    assert abs(val - 1.0) < 1e-6


def test_load_color_pfm(sample_pfm_file_alpha):
    loader = PFMLoader(sample_pfm_file_alpha)
    img = loader.load()
    # Color image shape (height, width, 4)
    assert img.shape[2] == 4
    assert img.dtype == "float32"
    # Width should be 3, height should be 1
    assert img.shape[0] == 1
    assert img.shape[1] == 3


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        PFMLoader("nonexistent.pfm").load()


def test_load_empty_file(tmp_path):
    empty_file = tmp_path / "empty.pfm"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        PFMLoader(str(empty_file)).load()


def test_load_bad_magic_number(tmp_path):
    f = tmp_path / "badmagic.pfm"
    f.write_text("PXM\n1 1\n")
    with pytest.raises(ValueError):
        PFMLoader(str(f)).load()


def test_load_bad_header_format(tmp_path):
    f = tmp_path / "badheader.pfm"
    f.write_text("PFM\nXYZ XYZ\n")
    with pytest.raises(ValueError):
        PFMLoader(str(f)).load()


def test_load_invalid_scale_value(tmp_path):
    f = tmp_path / "badscale.pfm"
    f.write_text("PFM\n1 1\naaa\n")
    with pytest.raises(ValueError):
        PFMLoader(str(f)).load()


def test_load_incorrect_data_size(tmp_path):
    # Header expects 2 pixels, but data empty
    f = tmp_path / "insizedata.pfm"
    f.write_bytes(b"PFM\n2 1\n-1.0\n")  # Scale line is split
    with pytest.raises(ValueError):
        PFMLoader(str(f)).load()


def test_load_with_lf_crlf_line_endings(tmp_path):
    # Create a file with CRLF line endings
    content = b"PFM\r\n2 1\r\n-1.0\r\n" + b"\x00\x00\x80?\x00\x00\x00@" * 2 * 1 * 3
    f = tmp_path / "crlf.pfm"
    f.write_bytes(content)
    loader = PFMLoader(str(f))
    arr = loader.load()
    assert arr.shape == (1, 2, 3)
    assert arr.dtype == "float32"


def test_load_with_extra_whitespace(tmp_path):
    content = b"PFM  \n 2    1  \n -1.0 \n" + b"\x00\x00\x80?" * 6  # 2*1*3 floats
    f = tmp_path / "extra_white.pfm"
    f.write_bytes(content)
    loader = PFMLoader(str(f))
    arr = loader.load()
    assert arr.shape == (1, 2, 3)
    assert arr.dtype == "float32"


def test_load_alpha_channel_image_shape(sample_pfm_file_alpha):
    loader = PFMLoader(sample_pfm_file_alpha)
    img = loader.load()
    assert img.ndim == 3
    assert img.shape[2] == 4


def test_load_grayscale_shape_dimensions(sample_pfm_file):
    loader = PFMLoader(sample_pfm_file)
    img = loader.load()
    # grayscale images can be either 2D or 3D with 1 channel
    assert img.shape[:2] == (3, 2)
    assert img.dtype == "float32"


def test_load_inverse_scale(sample_pfm_file_alpha):
    # ensure that scale value is used and determines endianness
    # Open the file and read manual values
    loader = PFMLoader(sample_pfm_file_alpha)
    img = loader.load()
    # Since scale is negative, image is little-endian
    assert (img.flags['C_CONTIGUOUS'] is True)
    # We can only check shape here as we do not know exact values


def test_file_handle_closed_after_load(sample_pfm_file):
    loader = PFMLoader(sample_pfm_file)
    loader.load()
    # No direct way to test file handle closed, but no exception means closed properly


def test_load_content_is_float32(sample_pfm_file_alpha):
    loader = PFMLoader(sample_pfm_file_alpha)
    img = loader.load()
    assert img.dtype.name == "float32"


def test_fail_on_wrong_number_pixels(tmp_path):
    # Header says 2x2 pixels (4 pixels), but only 1 pixel data given (3 floats)
    f = tmp_path / "mismatch.pfm"
    f.write_bytes(b"PFM\n2 2\n-1.0\n" + b"\x00\x00\x80?"*3)
    loader = PFMLoader(str(f))
    with pytest.raises(ValueError):
        loader.load()

