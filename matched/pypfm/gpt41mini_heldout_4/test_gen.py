import io
import os
import sys
import tempfile
import pytest

from pypfm import PFMLoader, __version__


def test_version_exists_and_is_str():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_pfploader_can_be_initialized():
    loader = PFMLoader()
    assert isinstance(loader, PFMLoader)


def test_pfploader_initial_state():
    loader = PFMLoader()
    # PFMLoader should have 'load' and 'parse' methods
    assert callable(getattr(loader, "load", None))
    assert callable(getattr(loader, "parse", None))


def test_pfploader_load_nonexistent_file_raises():
    loader = PFMLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("this_file_does_not_exist.pfm")


def test_pfploader_load_invalid_file_type_raises(tmp_path):
    # create a text file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is not a PFM file")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(txt_file))


def test_pfploader_load_valid_minimal_pfm(tmp_path):
    pfm_content = (
        b'Pf\n'       # magic header for grayscale float
        b'1 1\n'      # width height
        b'0.000001\n' # scale factor > 0 means big endian
        b'\x3f\x80\x00\x00' # 1.0 in big endian float32
    )
    pfm_file = tmp_path / "single_pixel.pfm"
    with open(pfm_file, "wb") as f:
        f.write(pfm_content)

    loader = PFMLoader()
    image = loader.load(str(pfm_file))
    assert hasattr(image, "shape")
    assert image.shape == (1, 1)
    # Value should be close to 1.0
    assert abs(image[0, 0] - 1.0) < 1e-6


def test_pfploader_load_color_pfm(tmp_path):
    pfm_content = (
        b'PF\n'       # magic header for color float
        b'2 1\n'      # width height
        b'-1.0\n'    # scale < 0 means little endian, also scale absolute used
        # 2 pixels, 3 channels each, little endian floats: [0.0, 0.5, 1.0], [1.5, 2.0, 2.5]
        b'\x00\x00\x00\x00'  # 0.0
        b'\x00\x00\x00\x3f'  # 0.5
        b'\x00\x00\x80\x3f'  # 1.0
        b'\x00\x00\xc0\x3f'  # 1.5
        b'\x00\x00\x00\x40'  # 2.0
        b'\x00\x00 \x40'     # 2.5
    )
    pfm_file = tmp_path / "two_pixels.pfm"
    with open(pfm_file, "wb") as f:
        f.write(pfm_content)

    loader = PFMLoader()
    image = loader.load(str(pfm_file))
    assert image.shape == (1, 2, 3)
    # Values roughly match what we encoded
    expected = [
        [0.0, 0.5, 1.0],
        [1.5, 2.0, 2.5],
    ]
    for i in range(2):
        for j in range(3):
            assert abs(image[0, i, j] - expected[i][j]) < 1e-6


def test_pfploader_parse_raises_on_empty_bytes():
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.parse(b"")


def test_pfploader_parse_raises_on_invalid_header():
    loader = PFMLoader()
    # no valid header, too short
    with pytest.raises(ValueError):
        loader.parse(b"XYZ\n1 1\n1.0\n")


def test_pfploader_parse_png_data_equivalence(tmp_path):
    # Test that PFMLoader.parse works for actual PFM data
    pfm_content = (
        b'Pf\n'                # header grayscale
        b'2 2\n'               # width height
        b'1.0\n'               # scale big endian
        # 4 pixels, big endian float32: 1,2,3,4
        b'\x3f\x80\x00\x00'    # 1.0
        b'\x40\x00\x00\x00'    # 2.0
        b'\x40\x40\x00\x00'    # 3.0
        b'\x40\x80\x00\x00'    # 4.0
    )
    loader = PFMLoader()
    image = loader.parse(pfm_content)
    import numpy as np
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(image, expected)


def test_pfploader_parse_scale_zero_raises():
    loader = PFMLoader()
    pfm_content = b"Pf\n1 1\n0.0\n" + b"\x00\x00\x00\x00"
    with pytest.raises(ValueError):
        loader.parse(pfm_content)


def test_pfploader_parse_negative_scale_related_to_endianness():
    # scale < 0 indicates little endian, so data interpreted accordingly
    # We encode 1 pixel grayscale with scale -1.0, data little endian
    pfm_content = (
        b"Pf\n1 1\n-1.0\n"
        b'\x00\x00\x80\x3f'  # 1.0 float little endian
    )
    loader = PFMLoader()
    image = loader.parse(pfm_content)
    import numpy as np
    assert image.shape == (1, 1)
    np.testing.assert_allclose(image[0, 0], 1.0)


def test_pfploader_parse_incorrect_data_size_raises():
    loader = PFMLoader()
    # Declare image of size 2x2 grayscale (4 floats), provide only one float of data
    pfm_content = b"Pf\n2 2\n1.0\n" + b'\x3f\x80\x00\x00'
    with pytest.raises(ValueError):
        loader.parse(pfm_content)


def test_pfploader_parse_color_with_wrong_magic_raises():
    loader = PFMLoader()
    # Use magic P (lowercase instead of PF or PF)
    pfm_content = b"Pf\n1 1\n1.0\n" + b'\x3f\x80\x00\x00'  # should pass
    image = loader.parse(pfm_content)
    # Use wrong magic
    pfm_content_wrong = b"PZ\n1 1\n1.0\n" + b'\x3f\x80\x00\x00'
    with pytest.raises(ValueError):
        loader.parse(pfm_content_wrong)


def test_pfploader_parse_additional_trailing_data_is_allowed():
    # Add trailing bytes after the expected data: parse should succeed ignoring extras
    pfm_content = (
        b'Pf\n1 1\n1.0\n' +
        b'\x3f\x80\x00\x00' +  # one float pixel
        b'extra bytes here'
    )
    loader = PFMLoader()
    image = loader.parse(pfm_content)
    assert image.shape == (1, 1)
    import numpy as np
    np.testing.assert_allclose(image[0, 0], 1.0)


def test_pfploader_load_and_parse_produce_same_result(tmp_path):
    pfm_content = (
        b'Pf\n1 1\n1.0\n' +
        b'\x3f\x80\x00\x00'
    )
    pfm_file = tmp_path / "one_pixel.pfm"
    pfm_file.write_bytes(pfm_content)

    loader = PFMLoader()
    from numpy import allclose

    image_loaded = loader.load(str(pfm_file))
    image_parsed = loader.parse(pfm_content)

    assert image_loaded.shape == (1, 1)
    assert image_parsed.shape == (1, 1)
    assert allclose(image_loaded, image_parsed)


def test_pfploader_load_fails_on_non_binary_file(tmp_path):
    txt_file = tmp_path / "not_binary.pfm"
    txt_file.write_text("Pf\n1 1\n1.0\nabcdef\n")
    loader = PFMLoader()
    with pytest.raises((ValueError, UnicodeDecodeError)):
        loader.load(str(txt_file))


@pytest.mark.parametrize(
    "header,data_len",
    [
        ("Pf\n2 3\n1.0\n", 2*3*4),  # 24 bytes floats grayscale
        ("PF\n4 1\n-2.0\n", 4*1*3*4),  # 48 bytes floats RGB
    ],
)
def test_pfploader_parse_header_and_data_length(header, data_len):
    # Generate dummy data matching expected length
    data_bytes = b'\x00' * data_len
    content = header.encode("ascii") + data_bytes
    loader = PFMLoader()
    img = loader.parse(content)
    # Shapes must match declared size
    width_height_line = header.split("\n")[1]
    width, height = map(int, width_height_line.split())
    if header.startswith("PF"):
        expected_shape = (height, width, 3)
    else:
        expected_shape = (height, width)
    assert img.shape == expected_shape


def test_pfploader_parse_raises_if_header_not_ascii():
    loader = PFMLoader()
    # non ascii header bytes
    nonascii = b'\xff\xfe\xfd\n1 1\n1.0\n' + b'\x00\x00\x80\x3f'
    with pytest.raises(UnicodeDecodeError):
        loader.parse(nonascii)
