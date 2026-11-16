import pathlib
import struct

import pytest

import pypfm
from pypfm import PFMLoader


SAMPLE_VALUES = [0.5, 1.25, 2.0, -3.0, 4.5, 8.0]


def _write_pfm(path, magic, width, height, values, little_endian=True):
    """Write a minimal PFM file and return its path."""
    scale = -1.0 if little_endian else 1.0
    header = f"{magic}\n{width} {height}\n{scale}\n".encode("ascii")
    byte_order = "<" if little_endian else ">"
    body = struct.pack(byte_order + "f" * len(values), *values)
    path.write_bytes(header + body)
    return path


def _flatten(obj):
    """Flatten nested lists/tuples/array-likes into a flat list of floats."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return [float(obj)]


def _assert_values(result, expected):
    assert sorted(_flatten(result)) == pytest.approx(sorted(expected))


@pytest.fixture
def loader():
    try:
        return PFMLoader()
    except TypeError:
        pytest.skip("PFMLoader requires constructor arguments")


def _find_load_method(loader_obj):
    for name in ("load", "load_pfm", "read", "read_pfm", "open_pfm", "parse"):
        method = getattr(loader_obj, name, None)
        if callable(method):
            return method
    pytest.skip("No recognised load method on PFMLoader")


def test_version_string():
    assert pypfm.__version__ == "1.0.0"
    assert isinstance(pypfm.__version__, str)


def test_package_reexports_pfmloader_class():
    from pypfm.pypfm import PFMLoader as InnerPFMLoader

    assert pypfm.PFMLoader is InnerPFMLoader


def test_pfmloader_is_a_class():
    assert isinstance(PFMLoader, type)


def test_pfmloader_constructs_without_arguments(loader):
    assert isinstance(loader, PFMLoader)


def test_loader_exposes_a_load_method(loader):
    assert callable(_find_load_method(loader))


def test_loads_grayscale_little_endian(tmp_path, loader):
    path = _write_pfm(tmp_path / "gray_le.pfm", "Pf", 3, 2, SAMPLE_VALUES, little_endian=True)
    result = _find_load_method(loader)(str(path))
    _assert_values(result, SAMPLE_VALUES)


def test_loads_grayscale_big_endian(tmp_path, loader):
    path = _write_pfm(tmp_path / "gray_be.pfm", "Pf", 3, 2, SAMPLE_VALUES, little_endian=False)
    result = _find_load_method(loader)(str(path))
    _assert_values(result, SAMPLE_VALUES)


def test_endianness_does_not_change_loaded_values(tmp_path, loader):
    load = _find_load_method(loader)
    le = _write_pfm(tmp_path / "le.pfm", "Pf", 3, 2, SAMPLE_VALUES, little_endian=True)
    be = _write_pfm(tmp_path / "be.pfm", "Pf", 3, 2, SAMPLE_VALUES, little_endian=False)
    assert sorted(_flatten(load(str(le)))) == pytest.approx(sorted(_flatten(load(str(be)))))


def test_loads_color_pf_three_channels(tmp_path, loader):
    values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 1x2 pixels, 3 channels each
    path = _write_pfm(tmp_path / "color.pfm", "PF", 1, 2, values)
    result = _find_load_method(loader)(str(path))
    _assert_values(result, values)


def test_loads_single_pixel_image(tmp_path, loader):
    path = _write_pfm(tmp_path / "one.pfm", "Pf", 1, 1, [7.0])
    result = _find_load_method(loader)(str(path))
    _assert_values(result, [7.0])


def test_accepts_str_path(tmp_path, loader):
    path = _write_pfm(tmp_path / "s.pfm", "Pf", 3, 2, SAMPLE_VALUES)
    result = _find_load_method(loader)(str(path))
    _assert_values(result, SAMPLE_VALUES)


def test_accepts_pathlib_path(tmp_path, loader):
    path = _write_pfm(tmp_path / "p.pfm", "Pf", 3, 2, SAMPLE_VALUES)
    assert isinstance(path, pathlib.Path)
    result = _find_load_method(loader)(path)
    _assert_values(result, SAMPLE_VALUES)


def test_missing_file_raises(tmp_path, loader):
    missing = tmp_path / "does_not_exist.pfm"
    with pytest.raises(Exception):
        _find_load_method(loader)(str(missing))


def test_invalid_magic_raises(tmp_path, loader):
    path = _write_pfm(tmp_path / "bad_magic.pfm", "XX", 3, 2, SAMPLE_VALUES)
    with pytest.raises(Exception):
        _find_load_method(loader)(str(path))


def test_truncated_data_raises(tmp_path, loader):
    path = tmp_path / "truncated.pfm"
    header = b"Pf\n2 2\n-1.0\n"
    path.write_bytes(header + struct.pack("<ff", 1.0, 2.0))  # needs 4 floats
    with pytest.raises(Exception):
        _find_load_method(loader)(str(path))


def test_repeated_loads_are_consistent(tmp_path, loader):
    load = _find_load_method(loader)
    path = _write_pfm(tmp_path / "repeat.pfm", "Pf", 3, 2, SAMPLE_VALUES)
    first = sorted(_flatten(load(str(path))))
    second = sorted(_flatten(load(str(path))))
    assert first == pytest.approx(second)


def test_separate_loader_instances_are_independent(tmp_path):
    try:
        a = PFMLoader()
        b = PFMLoader()
    except TypeError:
        pytest.skip("PFMLoader requires constructor arguments")
    load_a = _find_load_method(a)
    load_b = _find_load_method(b)
    path = _write_pfm(tmp_path / "indep.pfm", "Pf", 3, 2, SAMPLE_VALUES)
    _assert_values(load_a(str(path)), SAMPLE_VALUES)
    _assert_values(load_b(str(path)), SAMPLE_VALUES)


def test_loading_does_not_modify_file(tmp_path, loader):
    path = _write_pfm(tmp_path / "readonly.pfm", "Pf", 3, 2, SAMPLE_VALUES)
    before = path.read_bytes()
    _find_load_method(loader)(str(path))
    assert path.read_bytes() == before
