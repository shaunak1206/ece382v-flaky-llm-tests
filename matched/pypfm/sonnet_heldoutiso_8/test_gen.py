import os
import struct

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(color=False, compress=False):
    return PFMLoader(color=color, compress=compress)


def write_raw_pfm(path, header, payload):
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert pypfm.PFMLoader is PFMLoader


def test_loader_can_be_instantiated_default_flags():
    loader = make_loader()
    assert loader is not None


def test_loader_has_public_api():
    loader = make_loader()
    assert callable(getattr(loader, "load_pfm", None))
    assert callable(getattr(loader, "save_pfm", None))


def test_save_creates_file(tmp_path):
    path = str(tmp_path / "gray.pfm")
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    make_loader().save_pfm(path, image)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_roundtrip_grayscale(tmp_path):
    path = str(tmp_path / "gray.pfm")
    image = (np.arange(20, dtype=np.float32).reshape(4, 5) * 0.5) + 0.25
    loader = make_loader(color=False)
    loader.save_pfm(path, image)
    loaded = loader.load_pfm(path)
    assert loaded.shape == image.shape
    np.testing.assert_allclose(loaded, image)


def test_roundtrip_color(tmp_path):
    path = str(tmp_path / "color.pfm")
    rng = np.random.RandomState(0)
    image = rng.rand(3, 4, 3).astype(np.float32)
    loader = make_loader(color=True)
    loader.save_pfm(path, image)
    loaded = loader.load_pfm(path)
    assert loaded.shape == image.shape
    np.testing.assert_allclose(loaded, image)


def test_roundtrip_compressed(tmp_path):
    path = str(tmp_path / "compressed.pfm")
    image = np.ones((6, 6), dtype=np.float32) * 3.5
    loader = make_loader(color=False, compress=True)
    loader.save_pfm(path, image)
    loaded = loader.load_pfm(path)
    assert loaded.shape == image.shape
    np.testing.assert_allclose(loaded, image)


def test_loaded_data_is_float32_compatible(tmp_path):
    path = str(tmp_path / "gray.pfm")
    image = np.random.RandomState(1).rand(2, 3).astype(np.float32)
    loader = make_loader()
    loader.save_pfm(path, image)
    loaded = loader.load_pfm(path)
    assert np.issubdtype(np.asarray(loaded).dtype, np.floating)


def test_saved_grayscale_header_is_Pf(tmp_path):
    path = str(tmp_path / "gray.pfm")
    image = np.zeros((2, 3), dtype=np.float32)
    make_loader(color=False).save_pfm(path, image)
    with open(path, "rb") as f:
        first_line = f.readline().strip()
    assert first_line == b"Pf"


def test_saved_color_header_is_PF(tmp_path):
    path = str(tmp_path / "color.pfm")
    image = np.zeros((2, 3, 3), dtype=np.float32)
    make_loader(color=True).save_pfm(path, image)
    with open(path, "rb") as f:
        first_line = f.readline().strip()
    assert first_line == b"PF"


def test_load_handwritten_little_endian_grayscale(tmp_path):
    path = str(tmp_path / "le.pfm")
    values = [1.0, 2.5, -3.0]
    write_raw_pfm(path, b"Pf\n3 1\n-1.0\n", struct.pack("<3f", *values))
    loaded = make_loader().load_pfm(path)
    np.testing.assert_allclose(np.asarray(loaded).reshape(-1), values)


def test_load_handwritten_big_endian_grayscale(tmp_path):
    path = str(tmp_path / "be.pfm")
    values = [4.0, 0.5, 7.25]
    write_raw_pfm(path, b"Pf\n3 1\n1.0\n", struct.pack(">3f", *values))
    loaded = make_loader().load_pfm(path)
    np.testing.assert_allclose(np.asarray(loaded).reshape(-1), values)


def test_load_handwritten_color_shape(tmp_path):
    path = str(tmp_path / "color_le.pfm")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    write_raw_pfm(path, b"PF\n2 1\n-1.0\n", struct.pack("<6f", *values))
    loaded = make_loader(color=True).load_pfm(path)
    assert np.asarray(loaded).shape == (1, 2, 3)
    np.testing.assert_allclose(np.asarray(loaded).reshape(-1), values)


def test_load_invalid_header_raises(tmp_path):
    path = str(tmp_path / "bad.pfm")
    write_raw_pfm(path, b"P5\n3 1\n-1.0\n", struct.pack("<3f", 1.0, 2.0, 3.0))
    with pytest.raises(Exception):
        make_loader().load_pfm(path)


def test_load_missing_file_raises(tmp_path):
    path = str(tmp_path / "does_not_exist.pfm")
    with pytest.raises(Exception):
        make_loader().load_pfm(path)
