import inspect
import os

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(color=False):
    """Build a PFMLoader, passing only constructor arguments it accepts."""
    params = inspect.signature(PFMLoader.__init__).parameters
    kwargs = {}
    if "color" in params:
        kwargs["color"] = color
    return PFMLoader(**kwargs)


def _first(result):
    """load_pfm may return an array or a tuple (array, scale)."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _save(loader, path, image):
    loader.save_pfm(str(path), image)


def _load(loader, path):
    return np.asarray(_first(loader.load_pfm(str(path))))


def test_package_exports_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_loader_is_class_and_instantiable():
    assert inspect.isclass(PFMLoader)
    loader = make_loader()
    assert isinstance(loader, PFMLoader)


def test_loader_has_public_api():
    loader = make_loader()
    assert callable(getattr(loader, "load_pfm", None))
    assert callable(getattr(loader, "save_pfm", None))


def test_roundtrip_grayscale(tmp_path):
    loader = make_loader(color=False)
    image = np.random.RandomState(0).rand(6, 8).astype(np.float32)
    path = tmp_path / "gray.pfm"
    _save(loader, path, image)
    assert path.exists()
    loaded = _load(loader, path)
    assert loaded.shape == image.shape
    assert np.allclose(loaded, image)


def test_roundtrip_color(tmp_path):
    loader = make_loader(color=True)
    image = np.random.RandomState(1).rand(5, 7, 3).astype(np.float32)
    path = tmp_path / "color.pfm"
    _save(loader, path, image)
    loaded = _load(loader, path)
    assert loaded.shape == image.shape
    assert np.allclose(loaded, image)


def test_loaded_dtype_is_float(tmp_path):
    loader = make_loader()
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "dtype.pfm"
    _save(loader, path, image)
    loaded = _load(loader, path)
    assert np.issubdtype(loaded.dtype, np.floating)


def test_grayscale_header_magic(tmp_path):
    loader = make_loader(color=False)
    image = np.ones((4, 5), dtype=np.float32)
    path = tmp_path / "hdr_gray.pfm"
    _save(loader, path, image)
    with open(str(path), "rb") as f:
        assert f.readline().strip() == b"Pf"


def test_color_header_magic(tmp_path):
    loader = make_loader(color=True)
    image = np.ones((4, 5, 3), dtype=np.float32)
    path = tmp_path / "hdr_color.pfm"
    _save(loader, path, image)
    with open(str(path), "rb") as f:
        assert f.readline().strip() == b"PF"


def test_header_contains_dimensions(tmp_path):
    loader = make_loader(color=False)
    image = np.zeros((4, 9), dtype=np.float32)  # height 4, width 9
    path = tmp_path / "dims.pfm"
    _save(loader, path, image)
    with open(str(path), "rb") as f:
        f.readline()
        dims = f.readline().split()
    assert [int(d) for d in dims] == [9, 4]


def test_roundtrip_preserves_special_values(tmp_path):
    loader = make_loader()
    image = np.array([[0.0, -1.5, 1e10], [3.14159, -2e-5, 42.0]], dtype=np.float32)
    path = tmp_path / "special.pfm"
    _save(loader, path, image)
    loaded = _load(loader, path)
    assert np.allclose(loaded, image)


def test_orientation_is_consistent(tmp_path):
    loader = make_loader()
    image = np.zeros((4, 4), dtype=np.float32)
    image[0, :] = 1.0  # top row marked
    path = tmp_path / "orient.pfm"
    _save(loader, path, image)
    loaded = _load(loader, path)
    assert np.allclose(loaded[0, :], 1.0)
    assert np.allclose(loaded[1:, :], 0.0)


def test_overwrite_existing_file(tmp_path):
    loader = make_loader()
    path = tmp_path / "over.pfm"
    _save(loader, path, np.zeros((3, 3), dtype=np.float32))
    new_image = np.full((2, 5), 7.0, dtype=np.float32)
    _save(loader, path, new_image)
    loaded = _load(loader, path)
    assert loaded.shape == (2, 5)
    assert np.allclose(loaded, 7.0)


def test_load_missing_file_raises(tmp_path):
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_load_invalid_file_raises(tmp_path):
    loader = make_loader()
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"this is definitely not a pfm file\n")
    with pytest.raises(Exception):
        loader.load_pfm(str(path))
