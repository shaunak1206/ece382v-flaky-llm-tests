import inspect

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def _make_loader(width=None, height=None, **kwargs):
    """Build a PFMLoader, adapting to the constructor signature."""
    params = inspect.signature(PFMLoader.__init__).parameters
    call_kwargs = {}
    for name, param in params.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in kwargs:
            call_kwargs[name] = kwargs[name]
        elif name == "width" and width is not None:
            call_kwargs[name] = width
        elif name == "height" and height is not None:
            call_kwargs[name] = height
        elif param.default is inspect.Parameter.empty:
            call_kwargs[name] = 1
    return PFMLoader(**call_kwargs)


def _load(loader, path):
    result = loader.load_pfm(path)
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result)


def _roundtrip(tmp_path, image, color, name="img.pfm"):
    h, w = image.shape[:2]
    path = str(tmp_path / name)
    loader = _make_loader(width=w, height=h, color=color)
    loader.save_pfm(path, image)
    return path, _load(loader, path)


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    import pypfm.pypfm as inner

    assert inspect.isclass(PFMLoader)
    assert pypfm.PFMLoader is inner.PFMLoader


def test_loader_can_be_instantiated():
    loader = _make_loader(width=4, height=3)
    assert isinstance(loader, PFMLoader)
    assert hasattr(loader, "load_pfm")
    assert hasattr(loader, "save_pfm")


def test_roundtrip_grayscale(tmp_path):
    np.random.seed(0)
    image = np.random.rand(6, 8).astype(np.float32)
    _, loaded = _roundtrip(tmp_path, image, color=False)
    assert np.squeeze(loaded).shape == image.shape
    assert np.allclose(np.squeeze(loaded), image, atol=1e-6)


def test_roundtrip_color(tmp_path):
    np.random.seed(1)
    image = np.random.rand(5, 7, 3).astype(np.float32)
    _, loaded = _roundtrip(tmp_path, image, color=True)
    assert loaded.shape == image.shape
    assert np.allclose(loaded, image, atol=1e-6)


def test_loaded_data_is_float32(tmp_path):
    np.random.seed(2)
    image = np.random.rand(4, 4).astype(np.float32)
    _, loaded = _roundtrip(tmp_path, image, color=False)
    assert loaded.dtype == np.float32


def test_grayscale_header_magic(tmp_path):
    np.random.seed(3)
    image = np.random.rand(3, 5).astype(np.float32)
    path, _ = _roundtrip(tmp_path, image, color=False)
    with open(path, "rb") as f:
        data = f.read()
    assert data.startswith(b"Pf")


def test_color_header_magic(tmp_path):
    np.random.seed(4)
    image = np.random.rand(3, 5, 3).astype(np.float32)
    path, _ = _roundtrip(tmp_path, image, color=True)
    with open(path, "rb") as f:
        data = f.read()
    assert data.startswith(b"PF")


def test_header_contains_dimensions(tmp_path):
    np.random.seed(5)
    image = np.random.rand(3, 5).astype(np.float32)  # height 3, width 5
    path, _ = _roundtrip(tmp_path, image, color=False)
    with open(path, "rb") as f:
        head = f.read(32)
    assert b"5 3" in head


def test_file_size_covers_pixel_data(tmp_path):
    np.random.seed(6)
    image = np.random.rand(4, 6).astype(np.float32)
    path, _ = _roundtrip(tmp_path, image, color=False)
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) >= 4 * 6 * 4
    assert len(data) < 4 * 6 * 4 + 100  # header is small


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (16, 12)])
def test_roundtrip_various_shapes(tmp_path, shape):
    np.random.seed(7)
    image = np.random.rand(*shape).astype(np.float32)
    _, loaded = _roundtrip(tmp_path, image, color=False)
    assert np.squeeze(loaded).shape == np.squeeze(image).shape
    assert np.allclose(np.squeeze(loaded), np.squeeze(image), atol=1e-6)


def test_roundtrip_special_values(tmp_path):
    np.random.seed(8)
    image = np.array(
        [[0.0, -1.5, 1e6], [3.25, -1e-3, 42.0]], dtype=np.float32
    )
    _, loaded = _roundtrip(tmp_path, image, color=False)
    assert np.allclose(np.squeeze(loaded), image, rtol=1e-6, atol=1e-6)


def test_roundtrip_asymmetric_values_preserve_orientation(tmp_path):
    np.random.seed(9)
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    _, loaded = _roundtrip(tmp_path, image, color=False)
    loaded = np.squeeze(loaded)
    assert loaded[0, 0] == pytest.approx(image[0, 0])
    assert loaded[2, 3] == pytest.approx(image[2, 3])
    assert np.allclose(loaded, image, atol=1e-6)


def test_two_files_are_independent(tmp_path):
    np.random.seed(10)
    a = np.random.rand(4, 4).astype(np.float32)
    b = (np.random.rand(4, 4) + 5).astype(np.float32)
    loader = _make_loader(width=4, height=4, color=False)
    pa, pb = str(tmp_path / "a.pfm"), str(tmp_path / "b.pfm")
    loader.save_pfm(pa, a)
    loader.save_pfm(pb, b)
    assert np.allclose(np.squeeze(_load(loader, pa)), a, atol=1e-6)
    assert np.allclose(np.squeeze(_load(loader, pb)), b, atol=1e-6)


def test_load_missing_file_raises(tmp_path):
    np.random.seed(11)
    loader = _make_loader(width=2, height=2)
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_load_invalid_file_raises(tmp_path):
    np.random.seed(12)
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"this is definitely not a pfm file\n")
    loader = _make_loader(width=2, height=2)
    with pytest.raises(Exception):
        loader.load_pfm(str(path))
