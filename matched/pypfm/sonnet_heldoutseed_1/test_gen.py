import inspect
import os

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(color=False, width=4, height=3):
    """Build a PFMLoader, adapting to the constructor signature."""
    params = inspect.signature(PFMLoader.__init__).parameters
    kwargs = {}
    if "color" in params:
        kwargs["color"] = color
    if "compress" in params:
        kwargs["compress"] = False
    if "width" in params and params["width"].default is inspect.Parameter.empty:
        kwargs["width"] = width
    if "height" in params and params["height"].default is inspect.Parameter.empty:
        kwargs["height"] = height
    return PFMLoader(**kwargs)


def do_load(loader, path):
    result = loader.load_pfm(str(path))
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result)


def do_save(loader, path, image):
    loader.save_pfm(str(path), image)


def matches_either_orientation(actual, expected):
    return (actual.shape == expected.shape and
            (np.allclose(actual, expected) or
             np.allclose(actual, np.flipud(expected))))


def write_raw_pfm(path, header_id, width, height, scale, data, dtype):
    with open(str(path), "wb") as f:
        f.write(header_id + b"\n")
        f.write(("%d %d\n" % (width, height)).encode("ascii"))
        f.write(("%s\n" % scale).encode("ascii"))
        f.write(np.asarray(data, dtype=dtype).tobytes())


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    from pypfm.pypfm import PFMLoader as Inner
    assert pypfm.PFMLoader is Inner


def test_instantiate_grayscale_loader():
    loader = make_loader(color=False)
    assert isinstance(loader, PFMLoader)


def test_instantiate_color_loader():
    loader = make_loader(color=True)
    assert isinstance(loader, PFMLoader)


def test_loader_has_public_methods():
    loader = make_loader()
    assert callable(getattr(loader, "load_pfm", None))
    assert callable(getattr(loader, "save_pfm", None))


def test_roundtrip_grayscale(tmp_path):
    img = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5
    path = tmp_path / "gray.pfm"
    loader = make_loader(color=False)
    do_save(loader, path, img)
    assert os.path.exists(str(path))
    out = do_load(make_loader(color=False), path)
    assert out.shape == img.shape
    assert np.allclose(out, img)


def test_roundtrip_color(tmp_path):
    rng = np.random.RandomState(0)
    img = rng.rand(3, 4, 3).astype(np.float32)
    path = tmp_path / "color.pfm"
    do_save(make_loader(color=True), path, img)
    out = do_load(make_loader(color=True), path)
    assert out.shape == img.shape
    assert np.allclose(out, img)


def test_roundtrip_negative_and_zero_values(tmp_path):
    img = np.array([[-1.5, 0.0, 2.25, -100.0],
                    [0.0, 0.0, 3.5, 1e-3],
                    [7.0, -0.25, 0.0, 42.0]], dtype=np.float32)
    path = tmp_path / "neg.pfm"
    do_save(make_loader(), path, img)
    out = do_load(make_loader(), path)
    assert np.allclose(out, img)


def test_loaded_data_is_float(tmp_path):
    img = np.ones((3, 4), dtype=np.float32)
    path = tmp_path / "f.pfm"
    do_save(make_loader(), path, img)
    out = do_load(make_loader(), path)
    assert np.issubdtype(out.dtype, np.floating)


def test_saved_grayscale_header(tmp_path):
    img = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "hdr_gray.pfm"
    do_save(make_loader(color=False), path, img)
    with open(str(path), "rb") as f:
        content = f.read()
    assert content.startswith(b"Pf")
    first_lines = content.split(b"\n")[:3]
    assert b"4" in first_lines[1] and b"3" in first_lines[1]


def test_saved_color_header(tmp_path):
    img = np.zeros((3, 4, 3), dtype=np.float32)
    path = tmp_path / "hdr_color.pfm"
    do_save(make_loader(color=True), path, img)
    with open(str(path), "rb") as f:
        content = f.read()
    assert content.startswith(b"PF")


def test_saved_file_contains_all_pixel_data(tmp_path):
    img = np.ones((3, 4), dtype=np.float32)
    path = tmp_path / "size.pfm"
    do_save(make_loader(), path, img)
    assert os.path.getsize(str(path)) >= img.size * 4


def test_load_handwritten_little_endian_grayscale(tmp_path):
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "le.pfm"
    # PFM stores rows bottom-to-top
    write_raw_pfm(path, b"Pf", 4, 3, "-1.0", np.flipud(expected), "<f4")
    out = do_load(make_loader(color=False), path)
    assert matches_either_orientation(out, expected)


def test_load_handwritten_big_endian_grayscale(tmp_path):
    expected = np.arange(12, dtype=np.float32).reshape(3, 4) + 0.25
    path = tmp_path / "be.pfm"
    write_raw_pfm(path, b"Pf", 4, 3, "1.0", np.flipud(expected), ">f4")
    out = do_load(make_loader(color=False), path)
    assert matches_either_orientation(out, expected)


def test_load_handwritten_color(tmp_path):
    expected = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    path = tmp_path / "color_raw.pfm"
    write_raw_pfm(path, b"PF", 4, 3, "-1.0", np.flipud(expected), "<f4")
    out = do_load(make_loader(color=True), path)
    assert matches_either_orientation(out, expected)


def test_load_missing_file_raises(tmp_path):
    loader = make_loader()
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))
