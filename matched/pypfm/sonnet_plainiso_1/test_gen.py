import io
import sys

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader
from pypfm.pypfm import HEADER_SIZE, LENDIAN_SIZE, BENDIAN_SIZE


def _color_image(h=3, w=4):
    return np.arange(h * w * 3, dtype=np.float32).reshape(h, w, 3)


def test_version_and_export():
    assert pypfm.__version__ == "1.0.0"
    assert pypfm.PFMLoader is PFMLoader


def test_size_constants():
    assert HEADER_SIZE == 3
    assert LENDIAN_SIZE == len("-1.000000\n")
    assert BENDIAN_SIZE == len("1.000000\n")


def test_init_defaults_without_dim():
    loader = PFMLoader()
    assert loader.color is True
    assert loader.endian == "<"
    assert loader.dim is None
    assert loader.compress is False
    assert not hasattr(loader, "shape")


def test_init_with_dim_color_little_endian():
    loader = PFMLoader(dim=(4, 3))
    assert loader.dim == (4, 3)
    assert loader.width == 4
    assert loader.height == 3
    assert loader.shape == (3, 4, 3)
    assert loader.dim_size == len("4 3\n")
    assert loader.data_size == HEADER_SIZE + len("4 3\n") + LENDIAN_SIZE


def test_init_with_dim_gray_big_endian():
    loader = PFMLoader(dim=(5, 2), color=False, endian=">")
    assert loader.shape == (2, 5, 1)
    assert loader.data_size == HEADER_SIZE + len("5 2\n") + BENDIAN_SIZE


def test_build_size_updates_state():
    loader = PFMLoader()
    loader.build_size(10, 20, False, ">")
    assert loader.color is False
    assert (loader.width, loader.height) == (10, 20)
    assert loader.shape == (20, 10, 1)
    assert loader.dim_size == len("10 20\n")
    assert loader.data_size == HEADER_SIZE + len("10 20\n") + BENDIAN_SIZE


def test_extract_dim_color_little_endian():
    loader = PFMLoader()
    loader.extract_dim(io.BytesIO(b"PF\n4 3\n-1.000000\n"))
    assert loader.color is True
    assert (loader.width, loader.height) == (4, 3)
    assert loader.shape == (3, 4, 3)
    assert loader.data_size == HEADER_SIZE + len("4 3\n") + LENDIAN_SIZE


def test_extract_dim_gray_big_endian():
    loader = PFMLoader()
    loader.extract_dim(io.BytesIO(b"Pf\n6 2\n1.000000\n"))
    assert loader.color is False
    assert loader.shape == (2, 6, 1)
    assert loader.data_size == HEADER_SIZE + len("6 2\n") + BENDIAN_SIZE


def test_extract_dim_rejects_bad_magic():
    loader = PFMLoader()
    with pytest.raises(Exception, match="Not a PFM file"):
        loader.extract_dim(io.BytesIO(b"P6\n4 3\n-1.000000\n"))


def test_extract_dim_rejects_malformed_dimensions():
    loader = PFMLoader()
    with pytest.raises(Exception, match="Malformed PFM header"):
        loader.extract_dim(io.BytesIO(b"PF\nabc def\n-1.000000\n"))


def test_save_rejects_non_float32(tmp_path):
    loader = PFMLoader()
    image = np.zeros((3, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        loader.save_pfm(str(tmp_path / "bad.pfm"), image)


def test_save_color_writes_expected_header_and_size(tmp_path):
    loader = PFMLoader()
    image = _color_image()
    path = tmp_path / "color.pfm"
    returned = loader.save_pfm(str(path), image)
    assert returned is image
    raw = path.read_bytes()
    expected_scale = b"-1.000000\n" if sys.byteorder == "little" else b"1.000000\n"
    header = b"PF\n4 3\n" + expected_scale
    assert raw.startswith(header)
    assert len(raw) == len(header) + image.size * 4


def test_save_gray_uses_pf_lowercase_header(tmp_path):
    loader = PFMLoader(color=False)
    image = np.ones((3, 4), dtype=np.float32)
    path = tmp_path / "gray.pfm"
    loader.save_pfm(str(path), image)
    assert path.read_bytes().startswith(b"Pf\n4 3\n")


def test_save_scale_magnitude_in_header(tmp_path):
    loader = PFMLoader()
    path = tmp_path / "scaled.pfm"
    loader.save_pfm(str(path), _color_image(), scale=2)
    lines = path.read_bytes().split(b"\n", 3)
    value = float(lines[2])
    assert abs(value) == 2.0
    assert (value < 0) == (sys.byteorder == "little")


def test_save_big_endian_array_keeps_positive_scale(tmp_path):
    loader = PFMLoader()
    image = _color_image().astype(">f4")
    path = tmp_path / "be.pfm"
    loader.save_pfm(str(path), image)
    lines = path.read_bytes().split(b"\n", 3)
    assert float(lines[2]) == 1.0


def test_roundtrip_color_without_dim(tmp_path):
    image = _color_image()
    path = str(tmp_path / "rt.pfm")
    PFMLoader().save_pfm(path, image)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (3, 4, 3)
    np.testing.assert_array_equal(loaded, image)


def test_roundtrip_color_with_dim(tmp_path):
    image = _color_image(h=5, w=7)
    path = str(tmp_path / "rt_dim.pfm")
    PFMLoader().save_pfm(path, image)
    loaded = PFMLoader(dim=(7, 5)).load_pfm(path)
    assert loaded.shape == (5, 7, 3)
    np.testing.assert_array_equal(loaded, image)


def test_roundtrip_gray_adds_channel_axis(tmp_path):
    image = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    path = str(tmp_path / "gray.pfm")
    PFMLoader(color=False).save_pfm(path, image)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (3, 4, 1)
    np.testing.assert_array_equal(loaded[:, :, 0], image)


def test_load_big_endian_file_with_dim(tmp_path):
    values = np.arange(12, dtype=">f4").reshape(2, 2, 3)
    path = tmp_path / "be_in.pfm"
    path.write_bytes(b"PF\n2 2\n1.000000\n" + values.tobytes())
    loaded = PFMLoader(dim=(2, 2), endian=">").load_pfm(str(path))
    assert loaded.shape == (2, 2, 3)
    np.testing.assert_array_equal(loaded, values.astype(np.float32))


def test_load_invalid_file_raises(tmp_path):
    path = tmp_path / "junk.pfm"
    path.write_bytes(b"not a pfm\n1 1\n1.0\n")
    with pytest.raises(Exception, match="Not a PFM file"):
        PFMLoader().load_pfm(str(path))


def test_clean_values_replaces_out_of_range_and_counts():
    image = np.array([0, 1, 5, 10, 0])
    cleaned, changed = PFMLoader.clean_values(image, (2, 8))
    np.testing.assert_array_equal(cleaned, [0, 0, 5, 0, 0])
    assert changed == 2


def test_clean_values_boundaries_inclusive_and_range_order():
    image = np.array([2, 5, 8])
    cleaned, changed = PFMLoader.clean_values(image, (8, 2))
    np.testing.assert_array_equal(cleaned, [0, 5, 0])
    assert changed == 2


def test_clean_values_custom_placeholder_modifies_in_place():
    image = np.array([-1, 3, 7, 12])
    cleaned, changed = PFMLoader.clean_values(image, (0, 10), placeholder=-1)
    assert cleaned is image
    np.testing.assert_array_equal(image, [-1, 3, 7, -1])
    assert changed == 1
