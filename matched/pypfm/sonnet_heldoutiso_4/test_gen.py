import inspect
import os

import pytest

np = pytest.importorskip("numpy")

import pypfm
from pypfm import PFMLoader


def make_loader(color=False, width=4, height=3):
    """Build a PFMLoader, adapting to whatever constructor parameters exist."""
    sig = inspect.signature(PFMLoader.__init__)
    values = {"color": color, "width": width, "height": height}
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Parameter.empty:
            pytest.skip("Unsupported constructor parameter: %s" % name)
    return PFMLoader(**kwargs)


def require_methods(loader):
    for name in ("load_pfm", "save_pfm"):
        if not callable(getattr(loader, name, None)):
            pytest.skip("loader has no %s" % name)


def as_array(result):
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result)


def matches(result, expected):
    got = np.squeeze(as_array(result))
    exp = np.squeeze(expected)
    if got.shape != exp.shape:
        return False
    return bool(
        np.allclose(got, exp) or np.allclose(got, np.flipud(exp))
    )


def write_pfm(path, data, color=False, scale=-1.0):
    """Write a little-endian PFM file by hand (rows stored bottom to top)."""
    h, w = data.shape[:2]
    header = ("PF\n" if color else "Pf\n") + "%d %d\n%f\n" % (w, h, scale)
    with open(str(path), "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.flipud(data).astype("<f4").tobytes())


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    from pypfm.pypfm import PFMLoader as Inner
    assert PFMLoader is Inner


def test_pfmloader_is_class():
    assert inspect.isclass(PFMLoader)


def test_instantiate_grayscale():
    loader = make_loader(color=False)
    assert isinstance(loader, PFMLoader)


def test_instantiate_color():
    loader = make_loader(color=True)
    assert isinstance(loader, PFMLoader)


def test_has_public_methods():
    loader = make_loader()
    require_methods(loader)
    assert callable(loader.load_pfm)
    assert callable(loader.save_pfm)


def test_save_creates_file(tmp_path):
    loader = make_loader(color=False)
    require_methods(loader)
    path = tmp_path / "gray.pfm"
    data = np.random.RandomState(0).rand(3, 4).astype(np.float32)
    loader.save_pfm(str(path), data)
    assert path.exists()
    assert path.stat().st_size >= 3 * 4 * 4


def test_saved_gray_header(tmp_path):
    loader = make_loader(color=False)
    require_methods(loader)
    path = tmp_path / "gray.pfm"
    data = np.random.RandomState(1).rand(3, 4).astype(np.float32)
    loader.save_pfm(str(path), data)
    with open(str(path), "rb") as f:
        assert f.read(2) == b"Pf"


def test_saved_color_header(tmp_path):
    loader = make_loader(color=True)
    require_methods(loader)
    path = tmp_path / "color.pfm"
    data = np.random.RandomState(2).rand(3, 4, 3).astype(np.float32)
    loader.save_pfm(str(path), data)
    with open(str(path), "rb") as f:
        assert f.read(2) == b"PF"


def test_roundtrip_grayscale(tmp_path):
    loader = make_loader(color=False)
    require_methods(loader)
    path = tmp_path / "rt_gray.pfm"
    data = np.random.RandomState(3).rand(3, 4).astype(np.float32)
    loader.save_pfm(str(path), data)
    result = loader.load_pfm(str(path))
    assert matches(result, data)


def test_roundtrip_color(tmp_path):
    loader = make_loader(color=True)
    require_methods(loader)
    path = tmp_path / "rt_color.pfm"
    data = np.random.RandomState(4).rand(3, 4, 3).astype(np.float32)
    loader.save_pfm(str(path), data)
    result = loader.load_pfm(str(path))
    assert matches(result, data)


def test_roundtrip_different_size(tmp_path):
    loader = make_loader(color=False, width=7, height=5)
    require_methods(loader)
    path = tmp_path / "rt_size.pfm"
    data = np.random.RandomState(5).rand(5, 7).astype(np.float32)
    loader.save_pfm(str(path), data)
    result = loader.load_pfm(str(path))
    assert matches(result, data)


def test_load_returns_float32(tmp_path):
    loader = make_loader(color=False)
    require_methods(loader)
    path = tmp_path / "dtype.pfm"
    data = np.random.RandomState(6).rand(3, 4).astype(np.float32)
    loader.save_pfm(str(path), data)
    arr = as_array(loader.load_pfm(str(path)))
    assert arr.dtype == np.float32


def test_load_handwritten_gray_file(tmp_path):
    loader = make_loader(color=False)
    require_methods(loader)
    path = tmp_path / "hand_gray.pfm"
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_pfm(path, data, color=False)
    result = loader.load_pfm(str(path))
    assert matches(result, data)


def test_load_handwritten_color_file(tmp_path):
    loader = make_loader(color=True)
    require_methods(loader)
    path = tmp_path / "hand_color.pfm"
    data = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    write_pfm(path, data, color=True)
    result = loader.load_pfm(str(path))
    assert matches(result, data)


def test_load_missing_file_raises(tmp_path):
    loader = make_loader()
    require_methods(loader)
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_load_invalid_header_raises(tmp_path):
    loader = make_loader()
    require_methods(loader)
    path = tmp_path / "bad.pfm"
    with open(str(path), "wb") as f:
        f.write(b"XX\n4 3\n-1.0\n" + b"\x00" * 48)
    with pytest.raises(Exception):
        loader.load_pfm(str(path))
