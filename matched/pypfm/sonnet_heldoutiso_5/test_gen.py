import inspect

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(color=False, compress=False):
    """Build a loader, passing only the keyword arguments the constructor supports."""
    params = inspect.signature(PFMLoader.__init__).parameters
    kwargs = {}
    if "color" in params:
        kwargs["color"] = color
    if "compress" in params:
        kwargs["compress"] = compress
    return PFMLoader(**kwargs)


def supports_color_kw():
    return "color" in inspect.signature(PFMLoader.__init__).parameters


def to_array(result):
    """load_pfm may return the image alone or a tuple starting with the image."""
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result)


def read_bytes(path):
    with open(str(path), "rb") as f:
        return f.read()


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_pfmloader_is_class_with_expected_methods():
    assert inspect.isclass(PFMLoader)
    assert callable(getattr(PFMLoader, "load_pfm", None))
    assert callable(getattr(PFMLoader, "save_pfm", None))


def test_instantiate_default():
    loader = make_loader()
    assert isinstance(loader, PFMLoader)


def test_save_creates_file(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "gray.pfm"
    img = np.random.RandomState(0).rand(4, 6).astype(np.float32)
    loader.save_pfm(str(path), img)
    assert path.exists()
    assert path.stat().st_size > 0


def test_grayscale_header_magic(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "gray.pfm"
    img = np.random.RandomState(1).rand(4, 6).astype(np.float32)
    loader.save_pfm(str(path), img)
    assert read_bytes(path).startswith(b"Pf")


@pytest.mark.skipif(not supports_color_kw(), reason="no color option")
def test_color_header_magic(tmp_path):
    loader = make_loader(color=True)
    path = tmp_path / "color.pfm"
    img = np.random.RandomState(2).rand(4, 6, 3).astype(np.float32)
    loader.save_pfm(str(path), img)
    assert read_bytes(path).startswith(b"PF")


def test_header_contains_dimensions(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "dims.pfm"
    img = np.zeros((4, 6), dtype=np.float32)
    loader.save_pfm(str(path), img)
    header = read_bytes(path)[:30]
    assert b"6 4" in header


def test_grayscale_roundtrip_values(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "rt.pfm"
    img = np.random.RandomState(3).rand(5, 7).astype(np.float32)
    loader.save_pfm(str(path), img)
    loaded = to_array(loader.load_pfm(str(path)))
    assert loaded.shape == img.shape
    assert np.allclose(loaded, img)


@pytest.mark.skipif(not supports_color_kw(), reason="no color option")
def test_color_roundtrip_values(tmp_path):
    loader = make_loader(color=True)
    path = tmp_path / "rtc.pfm"
    img = np.random.RandomState(4).rand(5, 7, 3).astype(np.float32)
    loader.save_pfm(str(path), img)
    loaded = to_array(loader.load_pfm(str(path)))
    assert loaded.shape == img.shape
    assert np.allclose(loaded, img)


def test_loaded_data_is_float(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "dtype.pfm"
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    loader.save_pfm(str(path), img)
    loaded = to_array(loader.load_pfm(str(path)))
    assert np.issubdtype(loaded.dtype, np.floating)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (8, 2), (16, 16)])
def test_roundtrip_various_shapes(tmp_path, shape):
    loader = make_loader(color=False)
    path = tmp_path / "shape.pfm"
    img = np.random.RandomState(5).rand(*shape).astype(np.float32)
    loader.save_pfm(str(path), img)
    loaded = to_array(loader.load_pfm(str(path)))
    assert loaded.shape == shape
    assert np.allclose(loaded, img)


def test_roundtrip_preserves_row_order(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "order.pfm"
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    loader.save_pfm(str(path), img)
    loaded = to_array(loader.load_pfm(str(path)))
    assert np.allclose(loaded[0], img[0])
    assert np.allclose(loaded[-1], img[-1])


def test_load_missing_file_raises(tmp_path):
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_load_invalid_file_raises(tmp_path):
    loader = make_loader()
    path = tmp_path / "garbage.pfm"
    path.write_bytes(b"this is definitely not a pfm file")
    with pytest.raises(Exception):
        loader.load_pfm(str(path))
