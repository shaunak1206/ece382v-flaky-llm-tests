import numpy as np
import pytest

from pypfm import PFMLoader
import pypfm
from pypfm.pypfm import PFMLoader as InnerPFMLoader


def _pfm_bytes(header, width, height, scale_line, values, big_endian=False):
    """Build raw PFM file bytes for a single-row image (no row-order ambiguity)."""
    dtype = ">f4" if big_endian else "<f4"
    data = np.asarray(values, dtype=dtype).tobytes()
    return (
        header.encode("ascii") + b"\n"
        + f"{width} {height}".encode("ascii") + b"\n"
        + scale_line.encode("ascii") + b"\n"
        + data
    )


def test_version_is_expected_string():
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_package_exports_same_pfmloader_class_as_submodule():
    assert pypfm.PFMLoader is InnerPFMLoader


def test_pfmloader_is_a_class():
    assert isinstance(PFMLoader, type)


def test_pfmloader_can_be_instantiated_with_defaults():
    loader = PFMLoader()
    assert isinstance(loader, PFMLoader)


def test_two_loader_instances_are_independent_objects():
    a = PFMLoader()
    b = PFMLoader()
    assert a is not b


def test_load_grayscale_pf_file_shape(tmp_path):
    np.random.seed(0)
    path = tmp_path / "gray.pfm"
    path.write_bytes(_pfm_bytes("Pf", 3, 1, "-1.0", [0.5, 1.5, 2.5]))

    data = PFMLoader().load_pfm(str(path))

    arr = np.asarray(data[0] if isinstance(data, tuple) else data)
    assert arr.shape[:2] == (1, 3)
    assert arr.size == 3


def test_load_grayscale_pf_file_values(tmp_path):
    np.random.seed(0)
    values = [0.5, 1.5, 2.5]
    path = tmp_path / "gray_values.pfm"
    path.write_bytes(_pfm_bytes("Pf", 3, 1, "-1.0", values))

    data = PFMLoader().load_pfm(str(path))

    arr = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=np.float64)
    np.testing.assert_allclose(arr.ravel(), np.array(values), rtol=1e-6, atol=1e-6)


def test_load_color_pf_file_has_three_channels(tmp_path):
    np.random.seed(0)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    path = tmp_path / "color.pfm"
    path.write_bytes(_pfm_bytes("PF", 3, 1, "-1.0", values))

    data = PFMLoader().load_pfm(str(path))

    arr = np.asarray(data[0] if isinstance(data, tuple) else data)
    assert arr.shape[:2] == (1, 3)
    assert arr.size == 9


def test_load_big_endian_file_values_match(tmp_path):
    np.random.seed(0)
    values = [0.25, -0.75, 4.0]
    path = tmp_path / "big.pfm"
    path.write_bytes(_pfm_bytes("Pf", 3, 1, "1.0", values, big_endian=True))

    data = PFMLoader().load_pfm(str(path))

    arr = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=np.float64)
    np.testing.assert_allclose(arr.ravel(), np.array(values), rtol=1e-6, atol=1e-6)


def test_load_is_repeatable_for_same_file(tmp_path):
    np.random.seed(0)
    values = [0.1, 0.2, 0.3]
    path = tmp_path / "repeat.pfm"
    path.write_bytes(_pfm_bytes("Pf", 3, 1, "-1.0", values))

    loader = PFMLoader()
    first = np.asarray(
        loader.load_pfm(str(path))[0]
        if isinstance(loader.load_pfm(str(path)), tuple)
        else loader.load_pfm(str(path)),
        dtype=np.float64,
    )
    second = np.asarray(
        loader.load_pfm(str(path))[0]
        if isinstance(loader.load_pfm(str(path)), tuple)
        else loader.load_pfm(str(path)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_load_relative_path_from_changed_cwd(tmp_path, monkeypatch):
    np.random.seed(0)
    values = [3.0, 2.0, 1.0]
    (tmp_path / "rel.pfm").write_bytes(_pfm_bytes("Pf", 3, 1, "-1.0", values))
    monkeypatch.chdir(tmp_path)

    data = PFMLoader().load_pfm("rel.pfm")

    arr = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=np.float64)
    np.testing.assert_allclose(arr.ravel(), np.array(values), rtol=1e-6, atol=1e-6)


def test_load_missing_file_raises(tmp_path):
    np.random.seed(0)
    missing = tmp_path / "does_not_exist.pfm"
    with pytest.raises(Exception):
        PFMLoader().load_pfm(str(missing))


def test_load_invalid_header_raises(tmp_path):
    np.random.seed(0)
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"XX\n3 1\n-1.0\n" + b"\x00" * 12)
    with pytest.raises(Exception):
        PFMLoader().load_pfm(str(path))


def test_save_writes_pfm_magic_header(tmp_path):
    np.random.seed(0)
    data = np.random.rand(1, 3).astype(np.float32)
    path = tmp_path / "saved.pfm"

    PFMLoader().save_pfm(data, str(path))

    raw = path.read_bytes()
    assert raw[:2] in (b"Pf", b"PF")


def test_save_then_load_roundtrip_preserves_values(tmp_path):
    np.random.seed(0)
    data = np.random.rand(1, 4).astype(np.float32)
    path = tmp_path / "roundtrip.pfm"

    loader = PFMLoader()
    loader.save_pfm(data, str(path))
    loaded = loader.load_pfm(str(path))

    arr = np.asarray(loaded[0] if isinstance(loaded, tuple) else loaded, dtype=np.float64)
    np.testing.assert_allclose(arr.reshape(data.shape), data.astype(np.float64),
                               rtol=1e-6, atol=1e-6)


def test_roundtrip_is_stable_across_two_cycles(tmp_path):
    np.random.seed(0)
    data = np.random.rand(1, 5).astype(np.float32)
    first_path = tmp_path / "first.pfm"
    second_path = tmp_path / "second.pfm"

    loader = PFMLoader()
    loader.save_pfm(data, str(first_path))
    once = loader.load_pfm(str(first_path))
    once_arr = np.asarray(once[0] if isinstance(once, tuple) else once, dtype=np.float64)

    loader.save_pfm(once_arr.reshape(data.shape).astype(np.float32), str(second_path))
    twice = loader.load_pfm(str(second_path))
    twice_arr = np.asarray(twice[0] if isinstance(twice, tuple) else twice, dtype=np.float64)

    np.testing.assert_allclose(twice_arr, once_arr, rtol=1e-6, atol=1e-6)
