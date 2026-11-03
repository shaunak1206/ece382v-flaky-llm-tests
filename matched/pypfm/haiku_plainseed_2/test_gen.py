"""Unit tests for pypfm.PFMLoader and the pypfm package interface."""

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader
from pypfm.pypfm import PFMLoader as CoreLoader


def test_default_constructor_state():
    loader = PFMLoader()
    assert loader.color is True
    assert loader.endian == '<'
    assert loader.dim is None
    assert loader.compress is False


def test_constructor_with_dim_computes_layout():
    loader = PFMLoader(dim=(4, 3))
    assert loader.dim == (4, 3)
    assert (loader.width, loader.height) == (4, 3)
    assert loader.shape == (3, 4, 3)
    # 'PF\n' + '4 3\n' + '-1.000000\n'
    assert loader.data_size == 3 + 4 + 10


def test_constructor_big_endian_dim_uses_shorter_scale_field():
    loader = PFMLoader(dim=(4, 3), endian='>')
    # positive scale field '1.000000\n' is 9 bytes, not 10
    assert loader.data_size == 3 + 4 + 9


def test_build_size_sets_header_length_and_shape():
    loader = PFMLoader()
    loader.build_size(10, 7, False, '<')
    assert loader.color is False
    assert loader.shape == (7, 10, 1)
    assert loader.dim_size == len('10 7\n')
    assert loader.data_size == 3 + 5 + 10


def test_extract_dim_parses_color_header(tmp_path):
    path = tmp_path / 'color.pfm'
    path.write_bytes(b'PF\n6 2\n-1.000000\n')
    loader = PFMLoader()
    with open(path, 'rb') as f:
        loader.extract_dim(f)
        # header is 3 + 4 + 10 bytes; the reader must be left at the payload
        assert f.tell() == 17
    assert loader.color is True
    assert (loader.width, loader.height) == (6, 2)
    assert loader.shape == (2, 6, 3)


def test_extract_dim_parses_grayscale_header(tmp_path):
    path = tmp_path / 'gray.pfm'
    path.write_bytes(b'Pf\n3 5\n-1.000000\n')
    loader = PFMLoader()
    with open(path, 'rb') as f:
        loader.extract_dim(f)
    assert loader.color is False
    assert loader.shape == (5, 3, 1)


def test_extract_dim_positive_scale_selects_big_endian_layout(tmp_path):
    path = tmp_path / 'big.pfm'
    path.write_bytes(b'PF\n6 2\n1.000000\n')
    loader = PFMLoader()
    with open(path, 'rb') as f:
        loader.extract_dim(f)
    assert loader.data_size == 3 + 4 + 9
    assert loader.shape == (2, 6, 3)


def test_extract_dim_rejects_bad_magic(tmp_path):
    path = tmp_path / 'bad.pfm'
    path.write_bytes(b'P6\n6 2\n255\n')
    with open(path, 'rb') as f:
        with pytest.raises(Exception, match='Not a PFM file'):
            PFMLoader().extract_dim(f)


def test_extract_dim_rejects_malformed_dimensions(tmp_path):
    path = tmp_path / 'malformed.pfm'
    path.write_bytes(b'PF\n6x2\n-1.000000\n')
    with open(path, 'rb') as f:
        with pytest.raises(Exception, match='Malformed PFM header'):
            PFMLoader().extract_dim(f)


def test_save_then_load_round_trip_color(tmp_path):
    img = np.arange(36, dtype=np.float32).reshape(3, 4, 3) + 0.25
    path = str(tmp_path / 'rgb.pfm')
    PFMLoader().save_pfm(path, img)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (3, 4, 3)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, img)


def test_save_then_load_round_trip_grayscale(tmp_path):
    img = np.linspace(-3.0, 3.0, 10, dtype=np.float32).reshape(2, 5, 1)
    path = str(tmp_path / 'gray_rt.pfm')
    PFMLoader(color=False).save_pfm(path, img)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (2, 5, 1)
    np.testing.assert_array_equal(loaded, img)


def test_save_pfm_writes_little_endian_header_and_size(tmp_path):
    img = np.zeros((2, 5, 3), dtype=np.float32)
    path = tmp_path / 'hdr.pfm'
    PFMLoader().save_pfm(str(path), img)
    raw = path.read_bytes()
    # width is shape[1], height is shape[0]; native float32 gets scale -1
    assert raw.startswith(b'PF\n5 2\n-1.000000\n')
    assert len(raw) == 17 + img.nbytes


def test_save_pfm_rejects_non_float32(tmp_path):
    img = np.zeros((2, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='float32'):
        PFMLoader().save_pfm(str(tmp_path / 'wrong_dtype.pfm'), img)


def test_save_big_endian_array_writes_positive_scale(tmp_path):
    values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    img = values.astype('>f4')
    path = tmp_path / 'be_save.pfm'
    PFMLoader().save_pfm(str(path), img)
    raw = path.read_bytes()
    assert raw.startswith(b'PF\n3 2\n1.000000\n')
    assert len(raw) == 16 + img.nbytes
    loaded = PFMLoader(endian='>').load_pfm(str(path))
    np.testing.assert_array_equal(loaded, values)


def test_load_big_endian_file_with_big_endian_loader(tmp_path):
    values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    path = tmp_path / 'be_hand.pfm'
    path.write_bytes(b'PF\n3 2\n1.000000\n' + values.astype('>f4').tobytes())
    loaded = PFMLoader(endian='>').load_pfm(str(path))
    assert loaded.shape == (2, 3, 3)
    np.testing.assert_array_equal(loaded, values)


def test_load_with_preset_dim_skips_header(tmp_path):
    values = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    path = tmp_path / 'preset.pfm'
    path.write_bytes(b'PF\n4 3\n-1.000000\n' + values.tobytes())
    loaded = PFMLoader(dim=(4, 3)).load_pfm(str(path))
    assert loaded.shape == (3, 4, 3)
    np.testing.assert_array_equal(loaded, values)


def test_loader_reusable_across_different_sizes(tmp_path):
    a = np.full((3, 2, 3), 2.0, dtype=np.float32)
    b = np.full((1, 5, 3), -7.5, dtype=np.float32)
    path_a = str(tmp_path / 'a.pfm')
    path_b = str(tmp_path / 'b.pfm')
    PFMLoader().save_pfm(path_a, a)
    PFMLoader().save_pfm(path_b, b)

    loader = PFMLoader()
    got_a = loader.load_pfm(path_a)
    got_b = loader.load_pfm(path_b)
    assert got_a.shape == (3, 2, 3)
    assert got_b.shape == (1, 5, 3)
    np.testing.assert_array_equal(got_a, a)
    np.testing.assert_array_equal(got_b, b)
    # loading without a preset dim must not stick a dim on the loader
    assert loader.dim is None


def test_load_with_mismatched_preset_dim_raises(tmp_path):
    # the preset dim (4, 3) needs 36 floats; the payload only has 12
    path = tmp_path / 'short.pfm'
    path.write_bytes(b'PF\n2 2\n-1.000000\n' + np.zeros(12, dtype=np.float32).tobytes())
    with pytest.raises(ValueError):
        PFMLoader(dim=(4, 3)).load_pfm(str(path))


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PFMLoader().load_pfm(str(tmp_path / 'does_not_exist.pfm'))


def test_compressed_round_trip_within_tolerance(tmp_path):
    img = np.linspace(0.0, 10.0, num=8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)
    path = str(tmp_path / 'compressed.pfm')
    PFMLoader(compress=True).save_pfm(path, img)
    loaded = PFMLoader(compress=True).load_pfm(path)
    assert loaded.shape == (8, 12, 3)
    assert loaded.dtype == np.float32
    # codec is run with tolerance=0.5; allow a generous margin above that
    assert float(np.max(np.abs(loaded - img))) <= 1.0


def test_clean_values_zeroes_out_of_range_and_counts():
    arr = np.array([0, 1, 2, 5, 8, 10], dtype=np.int32)
    out, changed = PFMLoader.clean_values(arr, (2, 8))
    assert out is arr
    np.testing.assert_array_equal(out, [0, 0, 0, 5, 0, 0])
    # 1, 2, 8 and 10 were replaced with the placeholder
    assert changed == 4


def test_clean_values_accepts_reversed_range():
    arr = np.array([0, 1, 2, 5, 8, 10], dtype=np.int32)
    out, changed = PFMLoader.clean_values(arr, (8, 2))
    np.testing.assert_array_equal(out, [0, 0, 0, 5, 0, 0])
    assert changed == 4


def test_clean_values_no_out_of_range_values():
    arr = np.array([3, 4, 5, 6], dtype=np.int64)
    out, changed = PFMLoader.clean_values(arr, (2, 8))
    assert out is arr
    np.testing.assert_array_equal(out, [3, 4, 5, 6])
    assert changed == 0


def test_package_exports_pfmloader():
    assert pypfm.PFMLoader is CoreLoader
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__
