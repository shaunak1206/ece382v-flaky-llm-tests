import io
import os
import pytest
from pypfm import PFMLoader


# Helper function to create a minimal valid PFM file content in bytes
def minimal_pfm_content(width=2, height=2, scale=1.0, color=True, data=None):
    """
    Create a minimal valid PFM file content as bytes.
    """
    header = 'PF\n' if color else 'Pf\n'
    header += f"{width} {height}\n"
    # Negative scale means little endian, positive scale big endian (most commonly used is negative)
    header += f"{-abs(scale)}\n"

    # if no data supplied, create dummy data (width*height*3 or *1 floats)
    num_channels = 3 if color else 1
    count = width * height * num_channels
    if data is None:
        data = (b'\x00\x00\x80\x3f' * count)  # float32 1.0 little endian
    return header.encode('ascii') + data


def test_version_string():
    import pypfm
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_create_pfploader_instance():
    loader = PFMLoader()
    assert isinstance(loader, PFMLoader)


def test_read_color_pfm_from_bytes(tmp_path):
    # Write minimal color PFM file
    path = tmp_path / "color.pfm"
    content = minimal_pfm_content()
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (2, 2, 3)
    assert img.dtype.name == "float32"
    assert abs(scale - 1.0) < 1e-6  # scale stored as abs of header value


def test_read_greyscale_pfm_from_bytes(tmp_path):
    # Write minimal greyscale PFM file
    path = tmp_path / "grey.pfm"
    content = minimal_pfm_content(color=False)
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (2, 2)
    assert img.dtype.name == "float32"
    assert abs(scale - 1.0) < 1e-6


def test_invalid_magic_number(tmp_path):
    path = tmp_path / "invalid_magic.pfm"
    content = b"XX\n2 2\n-1.0\n" + b"\x00" * 24
    path.write_bytes(content)

    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))


def test_invalid_dimension_line(tmp_path):
    path = tmp_path / "invalid_dim.pfm"
    content = b"PF\nnotnumbers\n-1.0\n" + b"\x00" * 24
    path.write_bytes(content)

    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))


def test_invalid_scale_line(tmp_path):
    path = tmp_path / "invalid_scale.pfm"
    content = b"PF\n2 2\nnotafloat\n" + b"\x00" * 24
    path.write_bytes(content)

    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))


def test_endianness_big_endian(tmp_path):
    # scale positive means big endian
    width, height = 2, 2
    header = "PF\n2 2\n1.0\n"
    # float32 1.0 in big endian is \x3f\x80\x00\x00
    data = b'\x3f\x80\x00\x00' * (width * height * 3)
    content = header.encode("ascii") + data
    path = tmp_path / "big_endian.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (height, width, 3)
    assert img.dtype.name == "float32"
    assert abs(scale - 1.0) < 1e-6


def test_endianness_little_endian(tmp_path):
    # scale negative means little endian
    width, height = 2, 2
    header = "PF\n2 2\n-1.0\n"
    # float32 1.0 in little endian is \x00\x00\x80\x3f
    data = b'\x00\x00\x80\x3f' * (width * height * 3)
    content = header.encode("ascii") + data
    path = tmp_path / "little_endian.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (height, width, 3)
    assert img.dtype.name == "float32"
    assert abs(scale - 1.0) < 1e-6


def test_load_from_file_like_object():
    content = minimal_pfm_content()
    fileobj = io.BytesIO(content)

    loader = PFMLoader()
    img, scale = loader.load(fileobj)
    assert img.shape == (2, 2, 3)
    assert abs(scale - 1.0) < 1e-6


def test_load_with_non_existent_file():
    loader = PFMLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path.pfm")


def test_load_with_closed_fileobj():
    content = minimal_pfm_content()
    fileobj = io.BytesIO(content)
    fileobj.close()
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(fileobj)


def test_data_content_accuracy(tmp_path):
    # Check that actual pixel values read are correct
    width, height = 1, 1
    color = False
    # data for one pixel = float32 little endian 0.5 (=0x3f000000)
    data = b'\x00\x00\x00\x3f'
    content = minimal_pfm_content(width=width, height=height, color=color, data=data)
    path = tmp_path / "single_pixel.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (height, width)
    # Numpy float32 of 0.5 approx
    assert abs(img[0, 0] - 0.5) < 1e-6


def test_load_image_non_contiguous(tmp_path):
    # Load a valid color PFM and check that returned numpy array is writable and contiguous
    path = tmp_path / "test.pfm"
    content = minimal_pfm_content()
    path.write_bytes(content)

    loader = PFMLoader()
    img, _ = loader.load(str(path))
    assert img.flags.writeable
    assert img.flags.c_contiguous


def test_large_image(tmp_path):
    # Create a moderately large image for test
    width, height = 100, 100
    data = b'\x00\x00\x80\x3f' * (width * height * 3)
    content = minimal_pfm_content(width=width, height=height, data=data)
    path = tmp_path / "large.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, _ = loader.load(str(path))
    assert img.shape == (height, width, 3)


def test_load_with_extra_whitespace(tmp_path):
    # Check if loader correctly handles extra whitespace in headers
    content = b"PF\n 2  2 \n -1.00 \n" + b"\x00\x00\x80\x3f" * 12
    path = tmp_path / "extra_ws.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (2, 2, 3)
    assert abs(scale - 1.0) < 1e-6


def test_load_with_windows_line_endings(tmp_path):
    # Use Windows style line endings \r\n
    content = b"PF\r\n2 2\r\n-1.0\r\n" + b"\x00\x00\x80\x3f" * 12
    path = tmp_path / "windows_line_end.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    img, scale = loader.load(str(path))
    assert img.shape == (2, 2, 3)


def test_load_with_incorrect_data_size(tmp_path):
    # Write header for 2x2 color but only 4 floats instead of 12 floats
    content = b"PF\n2 2\n-1.0\n" + b'\x00\x00\x80\x3f' * 4
    path = tmp_path / "short_data.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))
