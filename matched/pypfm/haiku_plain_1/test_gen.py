"""Unit tests for pypfm.PFMLoader (pypfm/pypfm.py) and the pypfm package.

Known limitations of the code under test, deliberately not asserted as correct:
  * extract_dim() never sets self.endian, so big-endian PFM files loaded
    without an explicit dim are read as little-endian.
  * clean_values() applies bitwise ~ to the selected values, which raises
    TypeError for float arrays, and its changed-count is wrong when the
    placeholder is not 0.
"""

import io
import struct

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader
from pypfm.pypfm import BENDIAN_SIZE, HEADER_SIZE, LENDIAN_SIZE


def test_init_defaults():
    loader = PFMLoader()
    assert loader.dim is None
    assert loader.color is True
    assert loader.endian == '<'
    assert loader.compress is False


def test_init_with_dim_sets_geometry():
    loader = PFMLoader(dim=(3, 2))
    assert loader.dim == (3, 2)
    assert loader.width == 3
    assert loader.height == 2
    assert loader.shape == (2, 3, 3)
    # 'PF\n' + '3 2\n' + '-1.000000\n'
    assert loader.data_size == 17


def test_init_grayscale_shape():
    loader = PFMLoader(dim=(4, 5), color=False)
    assert loader.color is False
    assert loader.shape == (5, 4, 1)


def test_build_size_little_endian_data_size():
    loader = PFMLoader()
    loader.build_size(3, 2, True, '<')
    assert loader.data_size == HEADER_SIZE + len('3 2\n') + LENDIAN_SIZE
    assert loader.data_size == 17


def test_build_size_big_endian_data_size():
    loader = PFMLoader()
    loader.build_size(3, 2, True, '>')
    assert loader.data_size == HEADER_SIZE + len('3 2\n') + BENDIAN_SIZE
    assert loader.data_size == 16


def test_extract_dim_color_header():
    loader = PFMLoader()
    loader.extract_dim(io.BytesIO(b'PF\n3 2\n-1.000000\n'))
    assert loader.color is True
    assert (loader.width, loader.height) == (3, 2)
    assert loader.shape == (2, 3, 3)
    assert loader.data_size == 17


def test_extract_dim_grayscale_header():
    loader = PFMLoader()
    loader.extract_dim(io.BytesIO(b'Pf\n4 1\n-1.0\n'))
    assert loader.color is False
    assert (loader.width, loader.height) == (4, 1)
    assert loader.shape == (1, 4, 1)


def test_extract_dim_positive_scale_uses_big_endian_size():
    loader = PFMLoader()
    loader.extract_dim(io.BytesIO(b'PF\n2 2\n1.0\n'))
    assert loader.data_size == HEADER_SIZE + len('2 2\n') + BENDIAN_SIZE
    assert loader.data_size == 16


def test_extract_dim_rejects_non_pfm_magic():
    loader = PFMLoader()
    with pytest.raises(Exception, match='Not a PFM file'):
        loader.extract_dim(io.BytesIO(b'P6\n3 2\n255\n'))


def test_extract_dim_rejects_malformed_size_line():
    loader = PFMLoader()
    with pytest.raises(Exception, match='Malformed PFM header'):
        loader.extract_dim(io.BytesIO(b'PF\n3x2\n-1.0\n'))


def test_save_pfm_writes_expected_header(tmp_path):
    img = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    path = tmp_path / 'header.pfm'
    PFMLoader(color=True).save_pfm(str(path), img, scale=2)
    data = path.read_bytes()
    lines = data.split(b'\n', 3)
    assert lines[:3] == [b'PF', b'3 2', b'-2.000000']
    # header is 3 + 4 + 10 bytes, then the raw float32 payload
    assert len(data) == 17 + img.nbytes


def test_save_pfm_rejects_non_float32(tmp_path):
    img = np.zeros((2, 3, 3), dtype=np.float64)
    path = tmp_path / 'bad.pfm'
    with pytest.raises(ValueError, match='float32'):
        PFMLoader().save_pfm(str(path), img)


def test_color_roundtrip_via_files(tmp_path):
    img = (np.arange(18, dtype=np.float32).reshape(2, 3, 3) * 0.5 - 2.0).astype(np.float32)
    path = tmp_path / 'color.pfm'
    PFMLoader(color=True).save_pfm(str(path), img)
    out = PFMLoader().load_pfm(str(path))
    assert out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out, img)


def test_grayscale_roundtrip_via_files(tmp_path):
    img = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3, 1)
    path = tmp_path / 'gray.pfm'
    PFMLoader(color=False).save_pfm(str(path), img)
    assert path.read_bytes().split(b'\n', 1)[0] == b'Pf'
    out = PFMLoader().load_pfm(str(path))
    assert out.shape == (2, 3, 1)
    np.testing.assert_array_equal(out, img)


def test_load_with_dim_skips_header(tmp_path):
    img = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    path = tmp_path / 'dim.pfm'
    PFMLoader(color=True).save_pfm(str(path), img)
    out = PFMLoader(dim=(3, 2)).load_pfm(str(path))
    np.testing.assert_array_equal(out, img)


def test_load_with_mismatched_dim_raises(tmp_path):
    img = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    path = tmp_path / 'mismatch.pfm'
    PFMLoader(color=True).save_pfm(str(path), img)
    # 4x2 colour needs 24 floats, the file only holds 18
    with pytest.raises(ValueError):
        PFMLoader(dim=(4, 2)).load_pfm(str(path))


def test_load_pfm_preserves_negative_and_fractional_values(tmp_path):
    img = np.array([[[-1.5, 0.25, 1.0e6], [1e-3, -7.0, 7.0]]], dtype=np.float32)
    path = tmp_path / 'values.pfm'
    PFMLoader(color=True).save_pfm(str(path), img)
    out = PFMLoader().load_pfm(str(path))
    assert out.shape == (1, 2, 3)
    np.testing.assert_array_equal(out, img)


def test_compressed_roundtrip_approximate(tmp_path):
    img = np.linspace(0.0, 4.0, 192, dtype=np.float32).reshape(8, 8, 3)
    path = tmp_path / 'compressed.pfm'
    PFMLoader(compress=True, color=True).save_pfm(str(path), img)
    out = PFMLoader(compress=True).load_pfm(str(path))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32
    # ZFP fixed-accuracy mode with tolerance=0.5 is lossy, so allow slack
    assert np.allclose(out, img, atol=1.0)


def test_clean_values_counts_changed_pixels():
    image = np.array([1, 5, 10, 20, 30], dtype=np.int64)
    out, changed = PFMLoader.clean_values(image, (2, 25))
    np.testing.assert_array_equal(out, [0, 5, 10, 20, 0])
    assert changed == 2

    # the order of the valid range does not matter
    image2 = np.array([1, 5, 10, 20, 30], dtype=np.int64)
    out2, changed2 = PFMLoader.clean_values(image2, (25, 2))
    np.testing.assert_array_equal(out2, [0, 5, 10, 20, 0])
    assert changed2 == 2


def test_clean_values_in_range_untouched():
    image = np.array([3, 4, 5], dtype=np.int64)
    out, changed = PFMLoader.clean_values(image, (1, 10))
    np.testing.assert_array_equal(out, [3, 4, 5])
    assert changed == 0


def test_clean_values_custom_placeholder_replaces_values():
    image = np.array([-5, 2, 8, 9, 0], dtype=np.int64)
    out, _ = PFMLoader.clean_values(image, (0, 8), placeholder=-1)
    assert out.tolist() == [-1, 2, -1, -1, -1]


def test_package_exports_and_version():
    assert pypfm.__version__ == '1.0.0'
    assert PFMLoader is pypfm.pypfm.PFMLoader
    assert HEADER_SIZE == 3
    assert LENDIAN_SIZE == 10
    assert BENDIAN_SIZE == 9


def test_load_handcrafted_little_endian_file(tmp_path):
    path = tmp_path / 'hand.pfm'
    values = struct.pack('<6f', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    path.write_bytes(b'PF\n2 1\n-1.0\n' + values)
    out = PFMLoader().load_pfm(str(path))
    assert out.shape == (1, 2, 3)
    expected = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_load_pfm_rejects_non_pfm_file(tmp_path):
    path = tmp_path / 'not_pfm.txt'
    path.write_bytes(b'hello world\n')
    with pytest.raises(Exception, match='Not a PFM file'):
        PFMLoader().load_pfm(str(path))
