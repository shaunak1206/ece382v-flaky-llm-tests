import io

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def _make_image(height=3, width=4, channels=3):
    n = height * width * channels
    arr = np.arange(n, dtype=np.float32) * 0.5 + 0.25
    return arr.reshape(height, width, channels)


def test_version_and_export():
    assert pypfm.__version__ == "1.0.0"
    assert pypfm.PFMLoader is PFMLoader


def test_init_defaults():
    loader = PFMLoader()
    assert loader.color is True
    assert loader.endian == '<'
    assert loader.dim is None
    assert loader.compress is False


def test_init_with_dim_builds_size_little_endian():
    loader = PFMLoader(dim=(4, 3))
    assert loader.dim == (4, 3)
    assert loader.width == 4
    assert loader.height == 3
    assert loader.shape == (3, 4, 3)
    # 'PF\n' (3) + '4 3\n' (4) + '-1.000000\n' (10)
    assert loader.data_size == 17


def test_init_with_dim_big_endian_data_size():
    loader = PFMLoader(dim=(4, 3), endian='>')
    # 'PF\n' (3) + '4 3\n' (4) + '1.000000\n' (9)
    assert loader.data_size == 16


def test_init_grayscale_shape():
    loader = PFMLoader(dim=(5, 2), color=False)
    assert loader.shape == (2, 5, 1)
    assert loader.color is False


def test_build_size_updates_color_and_shape():
    loader = PFMLoader()
    loader.build_size(10, 20, False, '<')
    assert loader.color is False
    assert loader.shape == (20, 10, 1)
    assert loader.width == 10
    assert loader.height == 20
    # 3 + len('10 20\n') + 10
    assert loader.data_size == 3 + 6 + 10


def test_extract_dim_color_little_endian():
    loader = PFMLoader()
    reader = io.BytesIO(b'PF\n6 5\n-1.0\n')
    loader.extract_dim(reader)
    assert loader.color is True
    assert loader.width == 6
    assert loader.height == 5
    assert loader.shape == (5, 6, 3)
    assert loader.data_size == 3 + len('6 5\n') + 10


def test_extract_dim_grayscale_big_endian():
    loader = PFMLoader()
    reader = io.BytesIO(b'Pf\n7 2\n1.0\n')
    loader.extract_dim(reader)
    assert loader.color is False
    assert loader.shape == (2, 7, 1)
    assert loader.data_size == 3 + len('7 2\n') + 9


def test_extract_dim_rejects_bad_magic():
    loader = PFMLoader()
    with pytest.raises(Exception, match='Not a PFM file'):
        loader.extract_dim(io.BytesIO(b'P6\n4 3\n-1.0\n'))


def test_extract_dim_rejects_malformed_dimensions():
    loader = PFMLoader()
    with pytest.raises(Exception, match='Malformed PFM header'):
        loader.extract_dim(io.BytesIO(b'PF\nfoo bar\n-1.0\n'))


def test_save_writes_expected_header_and_size(tmp_path):
    path = tmp_path / 'img.pfm'
    image = _make_image()
    PFMLoader().save_pfm(str(path), image)
    raw = path.read_bytes()
    assert raw.startswith(b'PF\n4 3\n-1.000000\n')
    assert len(raw) == 17 + image.size * 4


def test_save_grayscale_header(tmp_path):
    path = tmp_path / 'gray.pfm'
    image = _make_image(channels=1)[:, :, 0]
    PFMLoader(color=False).save_pfm(str(path), image)
    raw = path.read_bytes()
    assert raw.startswith(b'Pf\n4 3\n')


def test_save_custom_scale_in_header(tmp_path):
    path = tmp_path / 'scaled.pfm'
    PFMLoader().save_pfm(str(path), _make_image(), scale=2)
    assert path.read_bytes().startswith(b'PF\n4 3\n-2.000000\n')


def test_save_rejects_non_float32(tmp_path):
    path = tmp_path / 'bad.pfm'
    image = np.zeros((3, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='float32'):
        PFMLoader().save_pfm(str(path), image)
    assert not path.exists()


def test_save_returns_image_unchanged_without_compress(tmp_path):
    image = _make_image()
    result = PFMLoader().save_pfm(str(tmp_path / 'x.pfm'), image)
    assert result is image


def test_roundtrip_color_without_dim(tmp_path):
    path = str(tmp_path / 'color.pfm')
    image = _make_image()
    PFMLoader().save_pfm(path, image)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (3, 4, 3)
    np.testing.assert_array_equal(loaded, image)


def test_roundtrip_grayscale_without_dim(tmp_path):
    path = str(tmp_path / 'gray.pfm')
    image = _make_image(channels=1)[:, :, 0]
    PFMLoader(color=False).save_pfm(path, image)
    loaded = PFMLoader().load_pfm(path)
    assert loaded.shape == (3, 4, 1)
    np.testing.assert_array_equal(loaded[:, :, 0], image)


def test_roundtrip_with_known_dim(tmp_path):
    path = str(tmp_path / 'dim.pfm')
    image = _make_image()
    PFMLoader().save_pfm(path, image)
    loaded = PFMLoader(dim=(4, 3)).load_pfm(path)
    assert loaded.shape == (3, 4, 3)
    np.testing.assert_array_equal(loaded, image)


def test_load_big_endian_file(tmp_path):
    path = tmp_path / 'be.pfm'
    values = np.array([1.5, -2.0, 3.25, 4.0, 5.0, 6.0], dtype='>f4')
    with open(str(path), 'wb') as f:
        f.write(b'PF\n2 1\n1.000000\n')
        f.write(values.tobytes())
    loaded = PFMLoader(endian='>').load_pfm(str(path))
    assert loaded.shape == (1, 2, 3)
    np.testing.assert_array_equal(loaded.reshape(-1), values.astype(np.float32))


def test_load_invalid_file_raises(tmp_path):
    path = tmp_path / 'junk.pfm'
    path.write_bytes(b'XX\n1 1\n-1.0\n')
    with pytest.raises(Exception, match='Not a PFM file'):
        PFMLoader().load_pfm(str(path))


def test_clean_values_replaces_out_of_range_and_counts():
    image = np.array([0, 1, 2, 3, 4])
    result, changed = PFMLoader.clean_values(image, (1, 3))
    np.testing.assert_array_equal(result, np.array([0, 0, 2, 0, 0]))
    assert changed == 3


def test_clean_values_range_order_and_placeholder():
    image = np.array([5, 6, 7, 8, 9])
    result, changed = PFMLoader.clean_values(image, (8, 6), placeholder=-1)
    # values <= 6 or >= 8 become -1
    np.testing.assert_array_equal(result, np.array([-1, -1, 7, -1, -1]))
    assert isinstance(changed, (int, np.integer))
    assert changed >= 0
