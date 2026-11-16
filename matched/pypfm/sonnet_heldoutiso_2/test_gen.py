import os

import numpy as np
import pytest

import pypfm
from pypfm import PFMLoader


@pytest.fixture
def loader():
    return PFMLoader()


def _write_raw_pfm(path, header_id, width, height, scale, data, endian="<"):
    with open(str(path), "wb") as f:
        f.write(header_id + b"\n")
        f.write(("%d %d\n" % (width, height)).encode("ascii"))
        f.write(("%f\n" % scale).encode("ascii"))
        f.write(np.asarray(data, dtype=endian + "f4").tobytes())


def test_package_version():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert PFMLoader is pypfm.PFMLoader
    assert isinstance(PFMLoader(), PFMLoader)


def test_roundtrip_grayscale(tmp_path, loader):
    img = np.random.RandomState(0).rand(6, 8).astype(np.float32)
    path = str(tmp_path / "gray.pfm")
    loader.save_pfm(path, img)
    assert os.path.exists(path)
    data, scale = loader.load_pfm(path)
    assert data.shape == (6, 8)
    assert abs(scale) == pytest.approx(1.0)
    np.testing.assert_allclose(data, img)


def test_roundtrip_color(tmp_path, loader):
    img = np.random.RandomState(1).rand(5, 7, 3).astype(np.float32)
    path = str(tmp_path / "color.pfm")
    loader.save_pfm(path, img)
    data, scale = loader.load_pfm(path)
    assert data.shape == (5, 7, 3)
    np.testing.assert_allclose(data, img)


def test_saved_gray_header(tmp_path, loader):
    img = np.zeros((4, 3), dtype=np.float32)
    path = str(tmp_path / "hdr.pfm")
    loader.save_pfm(path, img)
    with open(path, "rb") as f:
        content = f.read()
    lines = content.split(b"\n")
    assert lines[0].strip() == b"Pf"
    assert lines[1].split() == [b"3", b"4"]


def test_saved_color_header(tmp_path, loader):
    img = np.zeros((4, 3, 3), dtype=np.float32)
    path = str(tmp_path / "hdrc.pfm")
    loader.save_pfm(path, img)
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    assert lines[0].strip() == b"PF"
    assert lines[1].split() == [b"3", b"4"]


def test_saved_file_size(tmp_path, loader):
    img = np.ones((4, 3), dtype=np.float32)
    path = str(tmp_path / "size.pfm")
    loader.save_pfm(path, img)
    with open(path, "rb") as f:
        content = f.read()
    # raw float data occupies the last H*W*4 bytes
    assert len(content) > 4 * 3 * 4
    raw = np.frombuffer(content[-48:], dtype="<f4")
    np.testing.assert_allclose(raw, 1.0)


def test_load_flips_rows_bottom_to_top(tmp_path, loader):
    # PFM stores the bottom row first; the first stored row must be last row after loading
    raw = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # stored order
    path = tmp_path / "manual.pfm"
    _write_raw_pfm(path, b"Pf", 2, 3, -1.0, raw)
    data, scale = loader.load_pfm(str(path))
    assert data.shape == (3, 2)
    np.testing.assert_allclose(data, raw[::-1])
    assert abs(scale) == pytest.approx(1.0)


def test_load_little_endian_manual_color(tmp_path, loader):
    raw = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    path = tmp_path / "manual_color.pfm"
    _write_raw_pfm(path, b"PF", 2, 2, -1.0, raw)
    data, _ = loader.load_pfm(str(path))
    assert data.shape == (2, 2, 3)
    np.testing.assert_allclose(data, raw[::-1])


def test_load_big_endian_manual(tmp_path, loader):
    raw = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    path = tmp_path / "be.pfm"
    _write_raw_pfm(path, b"Pf", 2, 2, 1.0, raw, endian=">")
    data, scale = loader.load_pfm(str(path))
    assert scale == pytest.approx(1.0)
    np.testing.assert_allclose(data, raw[::-1])


def test_load_invalid_header_raises(tmp_path, loader):
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"P5\n2 2\n-1.0\n" + b"\x00" * 16)
    with pytest.raises(Exception):
        loader.load_pfm(str(path))


def test_load_missing_file_raises(tmp_path, loader):
    with pytest.raises(Exception):
        loader.load_pfm(str(tmp_path / "does_not_exist.pfm"))


def test_save_non_float32_raises(tmp_path, loader):
    img = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        loader.save_pfm(str(tmp_path / "x.pfm"), img)


def test_save_invalid_shape_raises(tmp_path, loader):
    img = np.zeros((2, 2, 2, 2), dtype=np.float32)
    with pytest.raises(Exception):
        loader.save_pfm(str(tmp_path / "x.pfm"), img)


def test_roundtrip_preserves_special_values(tmp_path, loader):
    img = np.array([[0.0, -1.5, 1e-10], [3.14159, 1e10, -0.0]], dtype=np.float32)
    path = str(tmp_path / "special.pfm")
    loader.save_pfm(path, img)
    data, _ = loader.load_pfm(path)
    np.testing.assert_array_equal(data, img)


def test_multiple_saves_overwrite(tmp_path, loader):
    path = str(tmp_path / "over.pfm")
    loader.save_pfm(path, np.zeros((2, 2), dtype=np.float32))
    new = np.full((3, 4), 7.0, dtype=np.float32)
    loader.save_pfm(path, new)
    data, _ = loader.load_pfm(path)
    assert data.shape == (3, 4)
    np.testing.assert_allclose(data, 7.0)
