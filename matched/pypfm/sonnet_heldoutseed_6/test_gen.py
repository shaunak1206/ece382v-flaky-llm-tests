import os

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(color=False):
    try:
        return PFMLoader(color=color)
    except TypeError:
        # Fallback for variants whose constructor requires explicit dimensions
        return PFMLoader(width=4, height=3, color=color)


def _gray_image():
    return np.arange(12, dtype=np.float32).reshape(3, 4) / 3.0


def _color_image():
    return (np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 7.0)


def _load(loader, path):
    return np.asarray(loader.load_pfm(str(path)))


def test_version_attribute():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert pypfm.PFMLoader is PFMLoader
    assert isinstance(PFMLoader, type)


def test_can_instantiate_gray_loader():
    loader = make_loader(color=False)
    assert isinstance(loader, PFMLoader)


def test_can_instantiate_color_loader():
    loader = make_loader(color=True)
    assert isinstance(loader, PFMLoader)


def test_loader_has_public_methods():
    loader = make_loader()
    assert callable(getattr(loader, "load_pfm", None))
    assert callable(getattr(loader, "save_pfm", None))


def test_save_creates_file(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "gray.pfm"
    loader.save_pfm(str(path), _gray_image())
    assert path.exists()
    assert os.path.getsize(str(path)) > 0


def test_gray_header_magic(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "gray.pfm"
    loader.save_pfm(str(path), _gray_image())
    with open(str(path), "rb") as f:
        assert f.read(2) == b"Pf"


def test_color_header_magic(tmp_path):
    loader = make_loader(color=True)
    path = tmp_path / "color.pfm"
    loader.save_pfm(str(path), _color_image())
    with open(str(path), "rb") as f:
        assert f.read(2) == b"PF"


def test_gray_roundtrip_values(tmp_path):
    loader = make_loader(color=False)
    img = _gray_image()
    path = tmp_path / "gray.pfm"
    loader.save_pfm(str(path), img)
    out = _load(loader, path)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img, rtol=1e-6)


def test_color_roundtrip_values(tmp_path):
    loader = make_loader(color=True)
    img = _color_image()
    path = tmp_path / "color.pfm"
    loader.save_pfm(str(path), img)
    out = _load(loader, path)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img, rtol=1e-6)


def test_loaded_data_is_float(tmp_path):
    loader = make_loader(color=False)
    path = tmp_path / "gray.pfm"
    loader.save_pfm(str(path), _gray_image())
    out = _load(loader, path)
    assert np.issubdtype(out.dtype, np.floating)


def test_roundtrip_preserves_negative_and_zero(tmp_path):
    loader = make_loader(color=False)
    img = np.array([[0.0, -1.5, 2.25, 100.0],
                    [-0.125, 3.5, -42.0, 0.5],
                    [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "neg.pfm"
    loader.save_pfm(str(path), img)
    out = _load(loader, path)
    np.testing.assert_allclose(out, img, rtol=1e-6)


def test_load_handwritten_little_endian_file(tmp_path):
    # 3 wide, 2 high, little-endian (scale < 0). PFM stores rows bottom-to-top.
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4")
    path = tmp_path / "manual.pfm"
    with open(str(path), "wb") as f:
        f.write(b"Pf\n3 2\n-1.0\n")
        f.write(data.tobytes())
    loader = make_loader(color=False)
    out = _load(loader, path)
    assert out.shape == (2, 3)
    # Accept either row orientation convention, but values must be intact
    assert (np.allclose(out, data) or np.allclose(out, np.flipud(data)))


def test_load_nonexistent_file_raises(tmp_path):
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_multiple_roundtrips_are_stable(tmp_path):
    loader = make_loader(color=False)
    img = _gray_image()
    p1 = tmp_path / "a.pfm"
    p2 = tmp_path / "b.pfm"
    loader.save_pfm(str(p1), img)
    first = _load(loader, p1)
    loader.save_pfm(str(p2), first)
    second = _load(loader, p2)
    np.testing.assert_allclose(second, img, rtol=1e-6)
