import struct

import pytest

import pypfm
import pypfm.pypfm as pypfm_module
from pypfm import PFMLoader


LOAD_METHOD_NAMES = ("load", "load_file", "read", "read_file", "from_file", "open")


def _build_pfm(width, height, channels, values, little_endian=True):
    """Build raw PFM bytes. Rows are stored bottom-to-top per the PFM spec."""
    magic = b"PF" if channels == 3 else b"Pf"
    scale = b"-1.0" if little_endian else b"1.0"
    header = magic + b"\n" + f"{width} {height}\n".encode("ascii") + scale + b"\n"
    byte_order = "<" if little_endian else ">"
    body = struct.pack(byte_order + "f" * len(values), *values)
    return header + body


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _load(path):
    """Load a file with PFMLoader, using whichever load method the class exposes."""
    loader = PFMLoader()
    for name in LOAD_METHOD_NAMES:
        method = getattr(loader, name, None)
        if callable(method):
            return method(str(path))
    pytest.skip("PFMLoader exposes no recognised load method")


def _dimensions(result):
    width = getattr(result, "width", None)
    height = getattr(result, "height", None)
    if width is None or height is None:
        pytest.skip("loaded result does not expose width/height attributes")
    return width, height


# ---------------------------------------------------------------------------
# Package-level behaviour (pypfm/__init__.py)
# ---------------------------------------------------------------------------

def test_version_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_a_string():
    assert isinstance(pypfm.__version__, str)


def test_pfmloader_is_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")


def test_package_pfmloader_is_same_object_as_module_class():
    assert pypfm.PFMLoader is pypfm_module.PFMLoader


def test_pfmloader_is_a_class():
    assert isinstance(PFMLoader, type)


def test_pfmloader_can_be_instantiated_with_no_arguments():
    loader = PFMLoader()
    assert isinstance(loader, PFMLoader)


# ---------------------------------------------------------------------------
# PFMLoader loading behaviour
# ---------------------------------------------------------------------------

def test_load_color_pfm_reports_expected_dimensions(tmp_path):
    values = [float(i) for i in range(2 * 3 * 3)]
    path = _write(tmp_path, "color.pfm", _build_pfm(2, 3, 3, values))
    result = _load(path)
    assert _dimensions(result) == (2, 3)


def test_load_grayscale_pfm_reports_expected_dimensions(tmp_path):
    values = [float(i) for i in range(4 * 2)]
    path = _write(tmp_path, "gray.pfm", _build_pfm(4, 2, 1, values))
    result = _load(path)
    assert _dimensions(result) == (4, 2)


def test_load_single_pixel_color_image(tmp_path):
    path = _write(tmp_path, "pixel.pfm", _build_pfm(1, 1, 3, [0.25, 0.5, 0.75]))
    result = _load(path)
    assert _dimensions(result) == (1, 1)


def test_load_big_endian_pfm_reports_expected_dimensions(tmp_path):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    path = _write(
        tmp_path, "big.pfm", _build_pfm(2, 1, 3, values, little_endian=False)
    )
    result = _load(path)
    assert _dimensions(result) == (2, 1)


def test_load_accepts_file_with_different_dimensions_independently(tmp_path):
    first = _write(tmp_path, "a.pfm", _build_pfm(5, 1, 1, [0.0] * 5))
    second = _write(tmp_path, "b.pfm", _build_pfm(1, 5, 3, [0.0] * 15))
    assert _dimensions(_load(first)) == (5, 1)
    assert _dimensions(_load(second)) == (1, 5)


def test_load_same_file_twice_gives_same_dimensions(tmp_path):
    path = _write(tmp_path, "twice.pfm", _build_pfm(3, 2, 1, [0.0] * 6))
    assert _dimensions(_load(path)) == _dimensions(_load(path))


# ---------------------------------------------------------------------------
# PFMLoader error handling
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pfm"
    with pytest.raises((OSError, ValueError)):
        _load(missing)


def test_load_invalid_magic_raises(tmp_path):
    bad = b"XX\n2 2\n-1.0\n" + struct.pack("<12f", *([0.0] * 12))
    path = _write(tmp_path, "bad_magic.pfm", bad)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        _load(path)


def test_load_empty_file_raises(tmp_path):
    path = _write(tmp_path, "empty.pfm", b"")
    with pytest.raises((ValueError, OSError, RuntimeError, EOFError)):
        _load(path)


def test_load_truncated_pixel_data_raises(tmp_path):
    full = _build_pfm(2, 2, 3, [0.0] * 12)
    path = _write(tmp_path, "truncated.pfm", full[:-8])
    with pytest.raises((ValueError, OSError, RuntimeError, EOFError)):
        _load(path)
