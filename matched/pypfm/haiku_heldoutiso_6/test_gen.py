import inspect
import re
import struct
from pathlib import Path

import pytest

import pypfm
from pypfm import PFMLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pfm(path, width, height, channels, values, scale=-1.0):
    """Write a PFM file. Negative scale means little-endian data."""
    magic = b"PF" if channels == 3 else b"Pf"
    header = magic + b"\n" + f"{width} {height}\n".encode("ascii") + f"{scale}\n".encode("ascii")
    endian = "<" if scale < 0 else ">"
    body = struct.pack(f"{endian}{len(values)}f", *values)
    Path(path).write_bytes(header + body)
    return path


def _load_with_loader(loader, path):
    """Call the loader's load-style method on a path."""
    for name in ("load", "read", "load_file", "open"):
        method = getattr(loader, name, None)
        if callable(method):
            return method(str(path))
    pytest.skip("PFMLoader exposes no known load method (load/read/load_file/open)")


def _flatten(obj):
    """Flatten nested lists / array-like objects into a flat list of numbers."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return [obj]


def _shape_of(obj):
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return tuple(shape)
    return None


def _public_callables(cls):
    return [
        name for name, member in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    ]


# ---------------------------------------------------------------------------
# Package-level tests (pypfm/__init__.py)
# ---------------------------------------------------------------------------

def test_version_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_string_in_semver_form():
    assert isinstance(pypfm.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", pypfm.__version__)


def test_package_exports_pfmloader_class():
    assert hasattr(pypfm, "PFMLoader")
    assert inspect.isclass(pypfm.PFMLoader)


def test_pfmloader_is_same_object_as_submodule_class():
    from pypfm.pypfm import PFMLoader as SubmoduleLoader

    assert pypfm.PFMLoader is SubmoduleLoader
    assert PFMLoader is SubmoduleLoader


# ---------------------------------------------------------------------------
# PFMLoader class-level tests
# ---------------------------------------------------------------------------

def test_pfmloader_can_be_instantiated_without_arguments():
    sig = inspect.signature(PFMLoader)
    for param in sig.parameters.values():
        assert param.default is not inspect.Parameter.empty or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ), f"constructor parameter {param.name!r} has no default"
    loader = PFMLoader()
    assert isinstance(loader, PFMLoader)


def test_pfmloader_exposes_public_api():
    assert _public_callables(PFMLoader), "PFMLoader should expose at least one public method"


def test_pfmloader_instances_are_independent():
    a = PFMLoader()
    b = PFMLoader()
    assert a is not b


# ---------------------------------------------------------------------------
# PFMLoader behaviour tests (file loading)
# ---------------------------------------------------------------------------

def test_load_grayscale_pfm_returns_values(tmp_path):
    path = _write_pfm(tmp_path / "gray.pfm", width=4, height=1, channels=1,
                      values=[0.0, 1.5, -2.25, 3.0])
    result = _load_with_loader(PFMLoader(), path)
    assert result is not None
    flat = _flatten(result)
    assert flat == pytest.approx([0.0, 1.5, -2.25, 3.0])


def test_load_color_pfm_returns_values(tmp_path):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2 pixels, 3 channels, 1 row
    path = _write_pfm(tmp_path / "color.pfm", width=2, height=1, channels=3, values=values)
    result = _load_with_loader(PFMLoader(), path)
    flat = _flatten(result)
    assert flat == pytest.approx(values)


def test_load_preserves_dimensions_when_shape_available(tmp_path):
    width, height = 3, 2
    values = [float(i) for i in range(width * height)]
    path = _write_pfm(tmp_path / "dims.pfm", width=width, height=height,
                      channels=1, values=values)
    result = _load_with_loader(PFMLoader(), path)
    shape = _shape_of(result)
    if shape is None:
        pytest.skip("loader result does not expose a shape attribute")
    assert width in shape and height in shape


def test_load_accepts_pathlib_path_and_str_equivalently(tmp_path):
    path = _write_pfm(tmp_path / "same.pfm", width=3, height=1, channels=1,
                      values=[7.0, 8.0, 9.0])
    loader = PFMLoader()
    from_str = _flatten(_load_with_loader(loader, str(path)))
    from_path = _flatten(_load_with_loader(loader, path))
    assert from_str == from_path == pytest.approx([7.0, 8.0, 9.0])


def test_load_is_deterministic_across_repeated_calls(tmp_path):
    path = _write_pfm(tmp_path / "repeat.pfm", width=2, height=1, channels=1,
                      values=[0.5, 0.25])
    loader = PFMLoader()
    first = _flatten(_load_with_loader(loader, path))
    second = _flatten(_load_with_loader(loader, path))
    assert first == second


def test_load_big_endian_pfm_positive_scale(tmp_path):
    values = [10.0, 20.0, 30.0]
    path = _write_pfm(tmp_path / "be.pfm", width=3, height=1, channels=1,
                      values=values, scale=1.0)
    result = _load_with_loader(PFMLoader(), path)
    assert _flatten(result) == pytest.approx(values)


def test_load_different_files_with_same_loader(tmp_path):
    p1 = _write_pfm(tmp_path / "one.pfm", width=2, height=1, channels=1, values=[1.0, 2.0])
    p2 = _write_pfm(tmp_path / "two.pfm", width=2, height=1, channels=1, values=[3.0, 4.0])
    loader = PFMLoader()
    assert _flatten(_load_with_loader(loader, p1)) == pytest.approx([1.0, 2.0])
    assert _flatten(_load_with_loader(loader, p2)) == pytest.approx([3.0, 4.0])


def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pfm"
    with pytest.raises(Exception):
        _load_with_loader(PFMLoader(), missing)


def test_load_invalid_magic_raises(tmp_path):
    bad = tmp_path / "bad_magic.pfm"
    bad.write_bytes(b"XX\n2 1\n-1.0\n" + struct.pack("<2f", 1.0, 2.0))
    with pytest.raises(Exception):
        _load_with_loader(PFMLoader(), bad)


def test_load_truncated_pixel_data_raises(tmp_path):
    truncated = tmp_path / "truncated.pfm"
    # Header claims 4 floats but only 1 is present.
    truncated.write_bytes(b"Pf\n4 1\n-1.0\n" + struct.pack("<f", 1.0))
    with pytest.raises(Exception):
        _load_with_loader(PFMLoader(), truncated)


def test_load_header_only_file_raises(tmp_path):
    header_only = tmp_path / "header_only.pfm"
    header_only.write_bytes(b"Pf\n2 1\n-1.0\n")
    with pytest.raises(Exception):
        _load_with_loader(PFMLoader(), header_only)
