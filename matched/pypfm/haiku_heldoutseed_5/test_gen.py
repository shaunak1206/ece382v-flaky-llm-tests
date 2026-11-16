import struct
from pathlib import Path

import pytest

import pypfm
from pypfm import PFMLoader


def _write_pfm(path, width, height, values, channels=1, little_endian=True):
    """Write a PFM file. `values` is a flat sequence of floats in file (bottom-up) order."""
    magic = b"PF" if channels == 3 else b"Pf"
    scale = -1.0 if little_endian else 1.0
    header = (
        magic
        + b"\n"
        + "{} {}\n".format(width, height).encode("ascii")
        + "{}\n".format(scale).encode("ascii")
    )
    endian = "<" if little_endian else ">"
    body = struct.pack(endian + "%df" % len(values), *values)
    Path(path).write_bytes(header + body)


def _flatten(data):
    if hasattr(data, "tolist"):
        data = data.tolist()
    out = []

    def _walk(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _walk(sub)
        else:
            out.append(item)

    _walk(data)
    return out


def _shape(data):
    if hasattr(data, "shape"):
        return tuple(data.shape)
    shape = []
    current = data
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def test_version_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_pfmloader_exported_from_package():
    assert pypfm.PFMLoader is PFMLoader
    assert isinstance(PFMLoader, type)


def test_pfmloader_can_be_instantiated():
    loader = PFMLoader()
    assert loader is not None


def test_load_grayscale_returns_expected_dimensions(tmp_path):
    values = [float(i) for i in range(6)]
    path = tmp_path / "gray.pfm"
    _write_pfm(path, width=3, height=2, values=values, channels=1)

    data = PFMLoader().load(str(path))

    non_unit = tuple(d for d in _shape(data) if d != 1)
    assert non_unit == (2, 3)


def test_load_grayscale_preserves_pixel_values(tmp_path):
    values = [0.5, 1.25, -2.0, 8.0, 0.0, 3.5]
    path = tmp_path / "gray_values.pfm"
    _write_pfm(path, width=3, height=2, values=values, channels=1)

    data = PFMLoader().load(str(path))

    assert sorted(_flatten(data)) == sorted(values)


def test_load_color_has_three_channels_per_pixel(tmp_path):
    width, height = 2, 2
    values = [float(i) for i in range(width * height * 3)]
    path = tmp_path / "color.pfm"
    _write_pfm(path, width=width, height=height, values=values, channels=3)

    data = PFMLoader().load(str(path))

    flat = _flatten(data)
    assert len(flat) == width * height * 3
    assert sorted(flat) == sorted(values)


def test_load_little_endian_values(tmp_path):
    values = [1.0, 2.0, 4.0, 8.0]
    path = tmp_path / "little.pfm"
    _write_pfm(path, width=2, height=2, values=values, little_endian=True)

    data = PFMLoader().load(str(path))

    assert sorted(_flatten(data)) == sorted(values)


def test_load_big_endian_values(tmp_path):
    values = [1.0, 2.0, 4.0, 8.0]
    path = tmp_path / "big.pfm"
    _write_pfm(path, width=2, height=2, values=values, little_endian=False)

    data = PFMLoader().load(str(path))

    assert sorted(_flatten(data)) == sorted(values)


def test_load_single_pixel(tmp_path):
    path = tmp_path / "one.pfm"
    _write_pfm(path, width=1, height=1, values=[42.5], channels=1)

    data = PFMLoader().load(str(path))

    assert _flatten(data) == [42.5]


def test_load_accepts_str_path(tmp_path):
    path = tmp_path / "str_path.pfm"
    _write_pfm(path, width=2, height=1, values=[1.0, 2.0])

    data = PFMLoader().load(str(path))

    assert sorted(_flatten(data)) == [1.0, 2.0]


def test_load_accepts_pathlib_path(tmp_path):
    path = tmp_path / "pathlib_path.pfm"
    _write_pfm(path, width=2, height=1, values=[3.0, 4.0])

    data = PFMLoader().load(path)

    assert sorted(_flatten(data)) == [3.0, 4.0]


def test_load_same_file_twice_gives_same_result(tmp_path):
    values = [0.25, 0.5, 0.75, 1.0]
    path = tmp_path / "twice.pfm"
    _write_pfm(path, width=2, height=2, values=values)

    loader = PFMLoader()
    first = _flatten(loader.load(str(path)))
    second = _flatten(loader.load(str(path)))

    assert first == second


def test_load_does_not_modify_file(tmp_path):
    values = [1.0, 2.0, 3.0, 4.0]
    path = tmp_path / "unchanged.pfm"
    _write_pfm(path, width=2, height=2, values=values)
    before = path.read_bytes()

    PFMLoader().load(str(path))

    assert path.read_bytes() == before


def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pfm"

    with pytest.raises((OSError, IOError)):
        PFMLoader().load(str(missing))


def test_load_invalid_magic_raises(tmp_path):
    path = tmp_path / "bad_magic.pfm"
    path.write_bytes(b"XX\n2 2\n-1.0\n" + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))

    with pytest.raises(Exception):
        PFMLoader().load(str(path))


def test_load_truncated_data_raises(tmp_path):
    path = tmp_path / "truncated.pfm"
    # Header claims 2x2 grayscale (4 floats) but only 2 floats are present.
    path.write_bytes(b"Pf\n2 2\n-1.0\n" + struct.pack("<2f", 1.0, 2.0))

    with pytest.raises(Exception):
        PFMLoader().load(str(path))
