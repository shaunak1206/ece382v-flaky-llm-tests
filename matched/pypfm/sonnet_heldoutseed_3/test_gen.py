import inspect
import os

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(width=4, height=3, color=False):
    """Build a PFMLoader, adapting to the constructor signature."""
    params = inspect.signature(PFMLoader.__init__).parameters
    kwargs = {}
    if "width" in params:
        kwargs["width"] = width
    if "height" in params:
        kwargs["height"] = height
    if "color" in params:
        kwargs["color"] = color
    if "compress" in params:
        kwargs["compress"] = False
    return PFMLoader(**kwargs)


def do_load(loader, path):
    result = loader.load_pfm(path)
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result)


def matches(expected, actual):
    """PFM stores rows bottom-to-top; accept either orientation."""
    if actual.shape != expected.shape:
        return False
    return (np.allclose(expected, actual) or
            np.allclose(np.flipud(expected), actual))


def gray_image(h=3, w=4):
    return (np.arange(h * w, dtype=np.float32).reshape(h, w) / 3.0).astype(np.float32)


def color_image(h=3, w=4):
    return (np.arange(h * w * 3, dtype=np.float32).reshape(h, w, 3) / 7.0).astype(np.float32)


def test_package_version():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader
    assert inspect.isclass(PFMLoader)


def test_loader_has_public_methods():
    assert callable(getattr(PFMLoader, "load_pfm", None))
    assert callable(getattr(PFMLoader, "save_pfm", None))


def test_loader_instantiation_gray():
    loader = make_loader(color=False)
    assert isinstance(loader, PFMLoader)


def test_loader_instantiation_color():
    loader = make_loader(color=True)
    assert isinstance(loader, PFMLoader)


def test_save_creates_file(tmp_path):
    path = str(tmp_path / "gray.pfm")
    loader = make_loader(color=False)
    loader.save_pfm(path, gray_image())
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_gray_header_is_pf_lowercase(tmp_path):
    path = str(tmp_path / "gray.pfm")
    make_loader(color=False).save_pfm(path, gray_image())
    with open(path, "rb") as f:
        assert f.read(2) == b"Pf"


def test_color_header_is_PF_uppercase(tmp_path):
    path = str(tmp_path / "color.pfm")
    make_loader(color=True).save_pfm(path, color_image())
    with open(path, "rb") as f:
        assert f.read(2) == b"PF"


def test_header_contains_dimensions(tmp_path):
    path = str(tmp_path / "dims.pfm")
    make_loader(width=4, height=3, color=False).save_pfm(path, gray_image(3, 4))
    with open(path, "rb") as f:
        data = f.read()
    header_tokens = data.split()[:3]
    assert header_tokens[1] == b"4"
    assert header_tokens[2] == b"3"


def test_file_size_gray_contains_pixel_data(tmp_path):
    path = str(tmp_path / "size.pfm")
    make_loader(color=False).save_pfm(path, gray_image(3, 4))
    # 12 float32 values = 48 bytes of payload at the end
    assert os.path.getsize(path) >= 48


def test_roundtrip_gray(tmp_path):
    path = str(tmp_path / "rt_gray.pfm")
    image = gray_image()
    loader = make_loader(color=False)
    loader.save_pfm(path, image)
    loaded = do_load(loader, path)
    assert matches(image, loaded)


def test_roundtrip_color(tmp_path):
    path = str(tmp_path / "rt_color.pfm")
    image = color_image()
    loader = make_loader(color=True)
    loader.save_pfm(path, image)
    loaded = do_load(loader, path)
    assert matches(image, loaded)


def test_loaded_data_is_float(tmp_path):
    path = str(tmp_path / "dtype.pfm")
    loader = make_loader(color=False)
    loader.save_pfm(path, gray_image())
    loaded = do_load(loader, path)
    assert np.issubdtype(loaded.dtype, np.floating)


def test_roundtrip_non_square_image(tmp_path):
    path = str(tmp_path / "rect.pfm")
    image = gray_image(h=5, w=2)
    loader = make_loader(width=2, height=5, color=False)
    loader.save_pfm(path, image)
    loaded = do_load(loader, path)
    assert loaded.shape == (5, 2)
    assert matches(image, loaded)


def test_load_missing_file_raises(tmp_path):
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_load_invalid_file_raises(tmp_path):
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"this is definitely not a pfm file")
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(path))
