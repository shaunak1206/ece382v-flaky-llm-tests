import json
import os
import warnings

import numpy as np
import pytest

import occamtools.occam_data as od


class FakeFort1:
    def __init__(self, file_name):
        self.file_name = file_name
        self.num_lines = 10
        self.comment_format_known = True
        self.file_contents = "ignored"
        self.n_time_steps_ = 3
        self.a = 1.0
        self.shared = "same"
        self.arr1 = np.array([1, 2, 3])

    def read_file(self, silent=False):
        self.read_called = True
        self.read_silent = silent


class FakeFort7:
    def __init__(self, file_name):
        self.file_name = file_name
        self.num_lines = 20
        self.comment_format_known = True
        self.file_contents = "ignored"
        self.n_time_steps_ = 3
        self.a = 1.0
        self.shared = "same"
        self.arr7 = np.array([4, 5])

    def read_file(self, silent=False):
        self.read_called = True
        self.read_silent = silent


class FakeXyz:
    def __init__(self, file_name):
        self.file_name = file_name
        self.num_lines = 30
        self.comment_format_known = True
        self.file_contents = "ignored"
        self.n_time_steps_ = 3
        self.a = 1.0
        self.shared = "same"
        self.arrx = np.array([[9, 8], [7, 6]])

    def read_file(self, silent=False):
        self.read_called = True
        self.read_silent = silent


@pytest.fixture
def patch_fakes(monkeypatch):
    monkeypatch.setattr(od, "Fort1", FakeFort1)
    monkeypatch.setattr(od, "Fort7", FakeFort7)
    monkeypatch.setattr(od, "Xyz", FakeXyz)


def _make_dataset(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "fort.1").write_text("f1")
    (base / "fort.7").write_text("f7")
    (base / "fort.8").write_text("x")
    return base


def test_check_internal_consistency_returns_true_for_matching_values(monkeypatch):
    monkeypatch.setattr(od, "_are_floats", lambda a, b: isinstance(a, float) and isinstance(b, float))
    a = FakeFort1("a")
    b = FakeFort7("b")
    assert od._check_internal_consistency(a, b) is True


def test_check_internal_consistency_warns_on_mismatch(monkeypatch):
    monkeypatch.setattr(od, "_are_floats", lambda a, b: isinstance(a, float) and isinstance(b, float))
    a = FakeFort1("a")
    b = FakeFort7("b")
    b.a = 2.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = od._check_internal_consistency(a, b)
    assert result is False
    assert any("Data is not internally consistent" in str(w.message) for w in rec)


def test_check_internal_consistency_ignores_file_name_and_num_lines(monkeypatch):
    monkeypatch.setattr(od, "_are_floats", lambda a, b: False)
    a = FakeFort1("a")
    b = FakeFort7("b")
    b.file_name = "different"
    b.num_lines = 999
    assert od._check_internal_consistency(a, b) is True


def test_check_internal_consistency_all_true(monkeypatch):
    monkeypatch.setattr(od, "_check_internal_consistency", lambda a, b: True)
    assert od._check_internal_consistency_all(FakeFort1("a"), FakeFort7("b"), FakeXyz("c")) is True


def test_check_internal_consistency_all_false(monkeypatch):
    calls = []

    def fake_check(a, b):
        calls.append((type(a).__name__, type(b).__name__))
        return not (type(a).__name__ == "FakeFort1" and type(b).__name__ == "FakeFort7")

    monkeypatch.setattr(od, "_check_internal_consistency", fake_check)
    result = od._check_internal_consistency_all(FakeFort1("a"), FakeFort7("b"), FakeXyz("c"))
    assert result is False
    assert calls == [("FakeFort1", "FakeFort7"), ("FakeFort1", "FakeXyz"), ("FakeFort7", "FakeXyz")]


def test_open_fort_files_with_paths_calls_read_file(monkeypatch, tmp_path, patch_fakes):
    base = _make_dataset(tmp_path)
    f1, f7, x = od._open_fort_files(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"), silent=True)
    assert isinstance(f1, FakeFort1)
    assert isinstance(f7, FakeFort7)
    assert isinstance(x, FakeXyz)
    assert f1.read_called and f7.read_called and x.read_called
    assert f1.read_silent is True and f7.read_silent is True and x.read_silent is True


def test_open_fort_files_with_which_zero(monkeypatch, tmp_path, patch_fakes):
    base = _make_dataset(tmp_path)
    f1, f7, x = od._open_fort_files(str(base / "fort.1"), "ignored", "ignored", silent=False, which=0)
    assert f1.file_name == str(base / "fort.1")
    assert f7.file_name == str(base / "fort.7")
    assert x.file_name == str(base / "fort.8")
    assert f1.read_silent is False and f7.read_silent is False and x.read_silent is False


def test_open_fort_files_with_which_one(monkeypatch, tmp_path, patch_fakes):
    base = _make_dataset(tmp_path)
    f1, f7, x = od._open_fort_files("ignored", str(base / "fort.7"), "ignored", silent=True, which=1)
    assert f1.file_name == str(base / "fort.1")
    assert f7.file_name == str(base / "fort.7")
    assert x.file_name == str(base / "fort.8")


def test_check_npy_dump_exists_true_for_directory(tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1]))
    exists, path = od._check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert path == str(class_dir)


def test_check_npy_dump_exists_false_when_no_dump(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    exists, path = od._check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert path == str(tmp_path / "class_data")


def test_check_npy_dump_exists_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / "missing"))


def test_check_constructor_input_single_path_to_fort1(monkeypatch, tmp_path, patch_fakes):
    base = _make_dataset(tmp_path)
    called = {}

    def fake_open(f1, f7, x, silent, which=None):
        called["args"] = (f1, f7, x, silent, which)
        return FakeFort1(f1), FakeFort7(f7), FakeXyz(x)

    monkeypatch.setattr(od, "_open_fort_files", fake_open)
    f1, f7, x = od._check_constructor_input(str(base / "fort.1"))
    assert called["args"][4] is None
    assert f1.file_name.endswith("fort.1")
    assert f7.file_name.endswith("fort.7")
    assert x.file_name.endswith("fort.8")


def test_check_constructor_input_single_non_string_raises():
    with pytest.raises(TypeError):
        od._check_constructor_input(123)


def test_check_constructor_input_three_strings_selects_automatic_open(monkeypatch, tmp_path, patch_fakes):
    base = _make_dataset(tmp_path)
    called = {}

    def fake_open(f1, f7, x, silent, which=None):
        called["args"] = (f1, f7, x, silent, which)
        return FakeFort1(f1), FakeFort7(f7), FakeXyz(x)

    monkeypatch.setattr(od, "_open_fort_files", fake_open)
    f1, f7, x = od._check_constructor_input(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"))
    assert called["args"][4] is None
    assert f1.file_name.endswith("fort.1")
    assert f7.file_name.endswith("fort.7")
    assert x.file_name.endswith("fort.8")


def test_occamdata_init_from_paths_saves_and_sets_attributes(tmp_path, monkeypatch, patch_fakes):
    base = _make_dataset(tmp_path)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda a, b, c: True)
    data = od.OccamData(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"))
    assert data.consistent is True
    assert data.fort1_file_name.endswith("fort.1")
    assert data.fort7_file_name.endswith("fort.7")
    assert data.xyz_file_name.endswith("fort.8")
    assert hasattr(data, "arr1") and hasattr(data, "arr7") and hasattr(data, "arrx")
    assert os.path.isdir(base / "class_data")
    assert os.path.exists(base / "class_data" / "class.json")


def test_occamdata_save_and_reload_roundtrip(tmp_path, monkeypatch, patch_fakes):
    base = _make_dataset(tmp_path)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda a, b, c: True)
    data = od.OccamData(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"), save_to_npy=False)
    assert data.save() is True
    assert not hasattr(data, "arr1")
    assert not hasattr(data, "arr7")
    assert not hasattr(data, "arrx")
    assert np.array_equal(data.arr1, np.array([1, 2, 3]))
    assert np.array_equal(data.arr7, np.array([4, 5]))
    assert np.array_equal(data.arrx, np.array([[9, 8], [7, 6]]))
    with open(base / "class_data" / "class.json", "r") as f:
        payload = json.load(f)
    assert payload["fort1_file_name"].endswith("fort.1")
    assert payload["fort7_file_name"].endswith("fort.7")
    assert payload["xyz_file_name"].endswith("fort.8")
    assert "arr1" not in payload


def test_occamdata_save_returns_false_when_directory_exists_and_no_overwrite(tmp_path, monkeypatch, patch_fakes):
    base = _make_dataset(tmp_path)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda a, b, c: True)
    data = od.OccamData(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"), save_to_npy=False)
    assert data.save() is True
    assert data.save() is False


def test_occamdata_save_overwrite_true(tmp_path, monkeypatch, patch_fakes):
    base = _make_dataset(tmp_path)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda a, b, c: True)
    data = od.OccamData(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"), save_to_npy=False)
    assert data.save() is True
    assert data.save(overwrite=True) is True


def test_occamdata_init_loads_existing_npy_dump(tmp_path, monkeypatch, patch_fakes):
    base = _make_dataset(tmp_path)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda a, b, c: True)
    first = od.OccamData(str(base / "fort.1"), str(base / "fort.7"), str(base / "fort.8"), save_to_npy=False)
    first.save()
    loaded = od.OccamData(str(base), load_from_npy=True, save_to_npy=False)
    assert np.array_equal(loaded.arr1, np.array([1, 2, 3]))
    assert np.array_equal(loaded.arr7, np.array([4, 5]))
    assert np.array_equal(loaded.arrx, np.array([[9, 8], [7, 6]]))
    assert loaded.fort1_file_name.endswith("fort.1")
    assert loaded.fort7_file_name.endswith("fort.7")
    assert loaded.xyz_file_name.endswith("fort.8")


def test_occamdata_init_single_missing_file_raises(tmp_path, patch_fakes):
    with pytest.raises(FileNotFoundError):
        od.OccamData(str(tmp_path / "missing.fort.1"), save_to_npy=False)
