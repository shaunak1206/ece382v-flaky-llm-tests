import io
import os
import sys
import tempfile
import pytest

from pypfm import PFMLoader


def test_version_string():
    import pypfm
    assert hasattr(pypfm, "__version__")
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_pfm_loader_exists():
    assert PFMLoader is not None
    # PFMLoader should be a callable class/type
    assert callable(PFMLoader)


def test_load_valid_ascii_pfm_file(tmp_path):
    # Minimal valid ASCII PFM content: header, size, scale, and 1 pixel
    # We make a 1x1 image with a pixel value of 0.5
    content = b"PFM\n1 1\n1.0\n" + (b"\x00\x00\x00\x00" if sys.byteorder == "big" else b"\x00\x00\x00\x00")
    # Instead of zero, write float32 0.5
    import struct
    content = b"PFM\n1 1\n1.0\n" + struct.pack("<f", 0.5)

    path = tmp_path / "test_ascii.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    image, scale = loader.load(str(path))

    import numpy as np
    assert isinstance(image, np.ndarray)
    assert image.shape == (1, 1)
    assert abs(image[0, 0] - 0.5) < 1e-6
    assert scale == 1.0


def test_load_valid_binary_pfm_file(tmp_path):
    # PFM binary: header PFM\n width height\n scale\n then float32 pixels
    width, height = 2, 2
    scale = -1.0  # negative scale => little endian
    header = f"PFM\n{width} {height}\n{scale}\n".encode("ascii")

    import numpy as np
    data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    # Flip vertically because PFM is stored bottom to top
    data_bytes = data[::-1].tobytes()

    content = header + data_bytes

    path = tmp_path / "test_binary.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    image, file_scale = loader.load(str(path))

    np.testing.assert_array_almost_equal(image, data)
    assert file_scale == scale


def test_load_invalid_header(tmp_path):
    path = tmp_path / "invalid_header.pfm"
    path.write_text("XYZ\n1 1\n1.0\n")
    loader = PFMLoader()

    with pytest.raises(RuntimeError):
        loader.load(str(path))


def test_load_invalid_size_line(tmp_path):
    path = tmp_path / "invalid_size.pfm"
    path.write_text("PFM\nnotanint 2\n1.0\n")
    loader = PFMLoader()
    with pytest.raises(RuntimeError):
        loader.load(str(path))


def test_load_invalid_scale_line(tmp_path):
    path = tmp_path / "invalid_scale.pfm"
    path.write_text("PFM\n2 2\nnotafloat\n")
    loader = PFMLoader()
    with pytest.raises(RuntimeError):
        loader.load(str(path))


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.pfm"
    path.write_bytes(b"")
    loader = PFMLoader()
    with pytest.raises(RuntimeError):
        loader.load(str(path))


def test_load_with_file_object(monkeypatch, tmp_path):
    # Same as test_load_valid_binary_pfm_file but use open file object

    width, height = 1, 1
    scale = 1.0
    header = f"PFM\n{width} {height}\n{scale}\n".encode("ascii")
    import numpy as np
    data = np.array([[9.0]], dtype=np.float32)
    content = header + data.tobytes()
    path = tmp_path / "file_obj.pfm"
    path.write_bytes(content)

    loader = PFMLoader()
    with open(path, "rb") as f:
        image, file_scale = loader.load(f)

    np.testing.assert_array_almost_equal(image, data)
    assert file_scale == scale


def test_loader_repr():
    loader = PFMLoader()
    r = repr(loader)
    assert isinstance(r, str)
    assert "PFMLoader" in r


def test_load_invalid_file_type():
    # Pass something other than str or file-like should raise TypeError
    loader = PFMLoader()
    with pytest.raises(TypeError):
        loader.load(42)


def test_load_bad_pixel_data_size(tmp_path):
    # Create pfm with too few pixel bytes
    path = tmp_path / "bad_pixels.pfm"
    contents = b"PFM\n1 1\n1.0\n"  # header
    # No pixel data
    path.write_bytes(contents)

    loader = PFMLoader()
    with pytest.raises(RuntimeError):
        loader.load(str(path))


# Test line ending variations
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_load_with_different_line_endings(tmp_path, line_ending):
    # Compose header with given line ending
    header = b"PFM" + line_ending + b"1 1" + line_ending + b"1.0" + line_ending
    import struct
    content = header + struct.pack("<f", 7.25)
    path = tmp_path / f"line_ending_{line_ending!r}.pfm"
    path.write_bytes(content)
    loader = PFMLoader()
    image, scale = loader.load(str(path))
    import numpy as np
    assert image.shape == (1, 1)
    assert abs(image[0, 0] - 7.25) < 1e-6
    assert scale == 1.0


def test_load_multi_channel_image(tmp_path):
    # PFM can contain multi-channel float data (e.g., 3 channels)
    width, height, channels = 2, 2, 3
    scale = -1.0
    header = f"PFM\n{width} {height}\n{scale}\n".encode("ascii")
    import numpy as np
    data = np.arange(width * height * channels, dtype=np.float32).reshape((height, width, channels))
    # Flip vertically
    data_bytes = data[::-1].tobytes()
    content = header + data_bytes
    path = tmp_path / "multi_channel.pfm"
    path.write_bytes(content)
    loader = PFMLoader()
    image, file_scale = loader.load(str(path))
    np.testing.assert_array_almost_equal(image, data)
    assert file_scale == scale


def test_load_after_reading_partial_file(tmp_path):
    # Test loading a pfm from an opened file object partially read before
    width, height = 1, 1
    scale = 1.0
    header = f"PFM\n{width} {height}\n{scale}\n".encode("ascii")
    import numpy as np
    data = np.array([[4.0]], dtype=np.float32)
    content = header + data.tobytes()
    path = tmp_path / "partial_read.pfm"
    path.write_bytes(content)
    loader = PFMLoader()

    with open(path, "rb") as f:
        # read part of the file (to shift pointer)
        partial = f.read(5)
        with pytest.raises(RuntimeError):
            loader.load(f)


def test_load_unicode_path(tmp_path):
    # Use a unicode path to ensure no issues with path types
    p = tmp_path / "unicodé.pfm"
    width, height = 1, 1
    scale = 1.0
    header = f"PFM\n{width} {height}\n{scale}\n".encode("ascii")
    import numpy as np
    data = np.array([[2.0]], dtype=np.float32)
    content = header + data.tobytes()
    p.write_bytes(content)

    loader = PFMLoader()
    image, file_scale = loader.load(str(p))
    import numpy as np
    np.testing.assert_array_almost_equal(image, data)
    assert file_scale == scale
