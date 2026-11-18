import io
import os
import pytest
from pypfm import PFMLoader


@pytest.fixture
def sample_pfm_file(tmp_path):
    content = (
        "PF\n"
        "4 3\n"
        "-0.5\n"
        + ("\x00" * (4 * 3 * 3 * 4))  # 4x3 pixels, 3 channels (RGB), float32 (4 bytes)
    )
    file_path = tmp_path / "sample.pfm"
    with open(file_path, "wb") as f:
        f.write(content.encode("latin1"))  # Write the header as text with latin1, then binary for pixels
        # Actually pixel data should be binary floats, so overwrite last part with binary floats
    # Overwrite pixel data with correct values (float32 little endian)
    import struct
    with open(file_path, "r+b") as f:
        f.seek(len("PF\n4 3\n-0.5\n"))
        for _ in range(4*3*3):
            f.write(struct.pack("<f", 1.0))
    return file_path


@pytest.fixture
def sample_pfm_gray_file(tmp_path):
    content = (
        "Pf\n"
        "2 2\n"
        "0.25\n"
    )
    file_path = tmp_path / "gray.pfm"
    # create with correct binary float pixel data
    import struct
    with open(file_path, "wb") as f:
        f.write(content.encode("latin1"))
        # 2*2 grayscale pixels = 4 pixels * 4 bytes each = 16 bytes
        for v in (0.1, 0.2, 0.3, 0.4):
            f.write(struct.pack(">f", v))  # write big endian floats (normal end, scale positive)
    return file_path


def test_version_string():
    import pypfm
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_import_pfploader_class():
    assert hasattr(PFMLoader, "__init__")
    loader = PFMLoader()
    assert loader is not None


def test_load_valid_pfm_rgb(sample_pfm_file):
    loader = PFMLoader()
    data = loader.load(str(sample_pfm_file))
    # Should be a 3D numpy array with shape height x width x 3 (3 channels)
    assert data.ndim == 3
    assert data.shape[2] == 3
    assert data.shape[0] == 3
    assert data.shape[1] == 4
    import numpy as np
    np.testing.assert_allclose(data, 1.0, atol=1e-6)


def test_load_valid_pfm_gray(sample_pfm_gray_file):
    loader = PFMLoader()
    data = loader.load(str(sample_pfm_gray_file))
    # Gray PFM should have 2D shape
    assert data.ndim == 2
    assert data.shape == (2, 2)
    # Values near the written values (0.1, 0.2, 0.3, 0.4)
    import numpy as np
    assert abs(data[0, 0] - 0.1) < 1e-6
    assert abs(data[0, 1] - 0.2) < 1e-6
    assert abs(data[1, 0] - 0.3) < 1e-6
    assert abs(data[1, 1] - 0.4) < 1e-6


def test_load_non_existing_file():
    loader = PFMLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existing_file.pfm")


def test_load_invalid_header(tmp_path):
    file_path = tmp_path / "invalid.pfm"
    with open(file_path, "w") as f:
        f.write("InvalidHeader\n")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_invalid_dimensions(tmp_path):
    file_path = tmp_path / "invalid_dim.pfm"
    with open(file_path, "w") as f:
        f.write("PF\ninvalid dimensions\n0.25\n")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_invalid_scale(tmp_path):
    file_path = tmp_path / "invalid_scale.pfm"
    with open(file_path, "w") as f:
        f.write("PF\n1 1\nnot_a_float\n")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_file_with_incomplete_data(tmp_path):
    file_path = tmp_path / "short.pfm"
    with open(file_path, "wb") as f:
        # header for 1x1 image, RGB (3 channels)
        f.write(b"PF\n1 1\n-1.0\n")
        # but no pixel data: should raise due to incomplete data
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_file_with_extra_whitespace(tmp_path):
    file_path = tmp_path / "whitespace.pfm"
    import struct
    with open(file_path, "wb") as f:
        f.write(b"PF   \n 2   1 \n 1.0\n")
        # 2x1 RGB float pixels
        f.write(struct.pack("<f", 0.5) * 3 * 2)
    loader = PFMLoader()
    data = loader.load(str(file_path))
    assert data.shape == (1, 2, 3)
    import numpy as np
    np.testing.assert_allclose(data, 0.5, atol=1e-6)


def test_load_scale_positive_big_endian(tmp_path):
    file_path = tmp_path / "big_endian.pfm"
    import struct
    with open(file_path, "wb") as f:
        f.write(b"Pf\n1 1\n0.5\n")
        # big endian float pixel 0.1234
        f.write(struct.pack(">f", 0.1234))
    loader = PFMLoader()
    data = loader.load(str(file_path))
    assert data.shape == (1, 1)
    import numpy as np
    np.testing.assert_allclose(data, 0.1234, atol=1e-6)


def test_load_scale_negative_little_endian(tmp_path):
    file_path = tmp_path / "little_endian.pfm"
    import struct
    with open(file_path, "wb") as f:
        f.write(b"Pf\n1 1\n-1.0\n")
        # little endian float pixel 0.5678
        f.write(struct.pack("<f", 0.5678))
    loader = PFMLoader()
    data = loader.load(str(file_path))
    assert data.shape == (1, 1)
    import numpy as np
    np.testing.assert_allclose(data, 0.5678, atol=1e-6)


def test_load_empty_file(tmp_path):
    file_path = tmp_path / "empty.pfm"
    file_path.write_text("")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_file_with_wrong_magic_number(tmp_path):
    file_path = tmp_path / "wrong_magic.pfm"
    file_path.write_text("PX\n2 2\n1.0\n")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(file_path))


def test_load_non_string_path():
    loader = PFMLoader()
    with pytest.raises(TypeError):
        loader.load(None)


def test_load_file_like_object(tmp_path):
    # If PFMLoader.load supports path-like objects, test that
    import pathlib
    loader = PFMLoader()
    file_path = tmp_path / "sample.pfm"
    file_path.write_bytes(b"PF\n1 1\n-1.0\n" + b"\x00" * 12)
    result = loader.load(file_path)  # pass Path object
    assert result.shape == (1, 1, 3)


def test_load_file_with_extra_lines(tmp_path):
    file_path = tmp_path / "extra_lines.pfm"
    import struct
    # Write correct header plus extra lines after scale before pixel data
    with open(file_path, "wb") as f:
        f.write(b"PF\n2 1\n-1.0\n\n\n")
        # 2 RGB pixels
        f.write(struct.pack("<f", 0.2) * 6)
    loader = PFMLoader()
    data = loader.load(str(file_path))
    import numpy as np
    np.testing.assert_allclose(data, 0.2, atol=1e-6)
    assert data.shape == (1, 2, 3)
