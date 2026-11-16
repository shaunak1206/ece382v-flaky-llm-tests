import inspect
import struct

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


def make_loader(width, height, color=False, compress=False):
    """Build a PFMLoader, passing only the constructor arguments it accepts."""
    params = inspect.signature(PFMLoader.__init__).parameters
    values = {
        "width": width,
        "height": height,
        "color": color,
        "compress": compress,
    }
    kwargs = {}
    for name, value in values.items():
        if name in params:
            # compress is only passed when it is the default value
            if name == "compress" and value:
                continue
            kwargs[name] = value
    return PFMLoader(**kwargs)


def first(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def gray_image(h=3, w=4):
    return (np.arange(h * w, dtype=np.float32).reshape(h, w) * 0.5) - 3.0


def color_image(h=3, w=4):
    return (np.arange(h * w * 3, dtype=np.float32).reshape(h, w, 3) * 0.25) - 2.0


def test_version_string():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    import pypfm.pypfm as inner

    assert pypfm.PFMLoader is inner.PFMLoader
    assert inspect.isclass(PFMLoader)


def test_loader_has_load_and_save_methods():
    assert callable(getattr(PFMLoader, "load_pfm", None))
    assert callable(getattr(PFMLoader, "save_pfm", None))


def test_roundtrip_grayscale_values(tmp_path):
    img = gray_image()
    path = str(tmp_path / "gray.pfm")
    loader = make_loader(4, 3)
    loader.save_pfm(path, img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert out.shape == img.shape
    assert np.allclose(out, img, atol=1e-6)


def test_roundtrip_color_values(tmp_path):
    img = color_image()
    path = str(tmp_path / "color.pfm")
    loader = make_loader(4, 3, color=True)
    loader.save_pfm(path, img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert out.shape == img.shape
    assert np.allclose(out, img, atol=1e-6)


def test_loaded_dtype_is_float32(tmp_path):
    img = gray_image()
    path = str(tmp_path / "dtype.pfm")
    loader = make_loader(4, 3)
    loader.save_pfm(path, img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert out.dtype == np.float32


def test_saved_grayscale_header_magic(tmp_path):
    img = gray_image()
    path = tmp_path / "g_header.pfm"
    loader = make_loader(4, 3)
    loader.save_pfm(str(path), img)
    data = path.read_bytes()
    assert data.startswith(b"Pf")


def test_saved_color_header_magic(tmp_path):
    img = color_image()
    path = tmp_path / "c_header.pfm"
    loader = make_loader(4, 3, color=True)
    loader.save_pfm(str(path), img)
    data = path.read_bytes()
    assert data.startswith(b"PF")


def test_saved_header_contains_dimensions(tmp_path):
    img = gray_image(h=3, w=5)
    path = tmp_path / "dims.pfm"
    loader = make_loader(5, 3)
    loader.save_pfm(str(path), img)
    data = path.read_bytes()
    lines = data.split(b"\n", 3)
    assert lines[1].split() == [b"5", b"3"]


def test_saved_file_contains_all_pixel_data(tmp_path):
    img = gray_image(h=3, w=4)
    path = tmp_path / "size.pfm"
    loader = make_loader(4, 3)
    loader.save_pfm(str(path), img)
    assert path.stat().st_size >= 3 * 4 * 4


def test_load_handwritten_little_endian_single_row(tmp_path):
    values = [1.5, -2.25, 8.0]
    path = tmp_path / "hand.pfm"
    path.write_bytes(b"Pf\n3 1\n-1.0\n" + struct.pack("<3f", *values))
    loader = make_loader(3, 1)
    out = np.asarray(first(loader.load_pfm(str(path))))
    assert out.size == 3
    assert np.allclose(out.reshape(-1), values, atol=1e-6)


def test_load_handwritten_color_single_pixel(tmp_path):
    values = [0.5, 1.0, 2.0]
    path = tmp_path / "hand_color.pfm"
    path.write_bytes(b"PF\n1 1\n-1.0\n" + struct.pack("<3f", *values))
    loader = make_loader(1, 1, color=True)
    out = np.asarray(first(loader.load_pfm(str(path))))
    assert out.size == 3
    assert np.allclose(out.reshape(-1), values, atol=1e-6)


def test_roundtrip_preserves_row_order(tmp_path):
    img = np.zeros((4, 2), dtype=np.float32)
    img[0, :] = 1.0
    img[3, :] = 9.0
    path = str(tmp_path / "order.pfm")
    loader = make_loader(2, 4)
    loader.save_pfm(path, img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[3], 9.0)
    assert np.allclose(out[1:3], 0.0)


def test_roundtrip_special_values(tmp_path):
    img = np.array([[0.0, -0.0, 1e-30], [1e30, -1e30, 3.14159]], dtype=np.float32)
    path = str(tmp_path / "special.pfm")
    loader = make_loader(3, 2)
    loader.save_pfm(path, img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert np.allclose(out, img, rtol=1e-6, atol=0.0)


def test_second_save_overwrites_first(tmp_path):
    path = str(tmp_path / "over.pfm")
    loader = make_loader(4, 3)
    loader.save_pfm(path, gray_image())
    new_img = gray_image() + 100.0
    loader.save_pfm(path, new_img)
    out = np.asarray(first(loader.load_pfm(path)))
    assert np.allclose(out, new_img, atol=1e-4)


def test_load_missing_file_raises(tmp_path):
    loader = make_loader(4, 3)
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))
