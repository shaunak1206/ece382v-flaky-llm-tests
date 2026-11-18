import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import occamtools.occam_data as occam_data


class DummyFile:
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.num_lines = attrs.pop("num_lines", 0)
        for k, v in attrs.items():
            setattr(self, k, v)

    def read_file(self, silent=False):
        self.read_silent = silent


class DummyFort1(DummyFile):
    pass


class DummyFort7(DummyFile):
    pass


class DummyXyz(DummyFile):
    pass


@pytest.fixture
def patched_file_classes(monkeypatch):
    monkeypatch.setattr(occam_data, "Fort1", DummyFort1)
    monkeypatch.setattr(occam_data, "Fort7", DummyFort7)
    monkeypatch.setattr(occam_data, "Xyz", DummyXyz)


def test_check_internal_consistency_same_numeric_values_no_warning():
    a = SimpleNamespace(file_name="a", x=1.0, y=np.array([1.0, 2.0]), num_lines=3)
    b = SimpleNamespace(file_name="b", x=1.0, y=np.array([1.0, 2.0]), num_lines=99)
    with pytest.warns(None) as record:
        result = occam_data._check_internal_consistency(a, b)
    assert result is True
    assert len(record) == 0


def test_check_internal_consistency_mismatch_numeric_warns_and_returns_false():
    a = SimpleNamespace(file_name="a", x=1.0, num_lines=3)
    b = SimpleNamespace(file_name="b", x=2.0, num_lines=4)
    with pytest.warns(UserWarning, match="Data is not internally consistent"):
        result = occam_data._check_internal_consistency(a, b)
    assert result is False


def test_check_internal_consistency_mismatch_string_warns_and_returns_false():
    a = SimpleNamespace(file_name="a", label="foo")
    b = SimpleNamespace(file_name="b", label="bar")
    with pytest.warns(UserWarning, match="foo"):
        result = occam_data._check_internal_consistency(a, b)
    assert result is False


def test_check_internal_consistency_all_true_when_all_match():
    f1 = SimpleNamespace(file_name="f1", a=1.0, b="same")
    f7 = SimpleNamespace(file_name="f7", a=1.0, b="same")
    xyz = SimpleNamespace(file_name="xyz", a=1.0, b="same")
    assert occam_data._check_internal_consistency_all(f1, f7, xyz) is True


def test_check_internal_consistency_all_false_when_one_pair_differs():
    f1 = SimpleNamespace(file_name="f1", a=1.0, b="same")
    f7 = SimpleNamespace(file_name="f7", a=2.0, b="same")
    xyz = SimpleNamespace(file_name="xyz", a=1.0, b="same")
    with pytest.warns(UserWarning):
        assert occam_data._check_internal_consistency_all(f1, f7, xyz) is False


def test_check_npy_dump_exists_raises_for_missing_path(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        occam_data._check_npy_dump_exists(str(missing))


def test_check_npy_dump_exists_for_directory(tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1]))
    exists, returned = occam_data._check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert returned == str(class_dir)


def test_check_npy_dump_exists_for_file_path(tmp_path):
    f = tmp_path / "fort.1"
    f.write_text("dummy")
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1]))
    exists, returned = occam_data._check_npy_dump_exists(str(f))
    assert exists is True
    assert returned == str(class_dir)


def test_open_fort_files_reads_all_three_when_paths_provided(patched_file_classes, tmp_path):
    f1 = tmp_path / "fort.1"
    f7 = tmp_path / "fort.7"
    x = tmp_path / "fort.8"
    for p in (f1, f7, x):
        p.write_text("dummy")

    f1_obj, f7_obj, x_obj = occam_data._open_fort_files(str(f1), str(f7), str(x), silent=True)
    assert isinstance(f1_obj, DummyFort1)
    assert isinstance(f7_obj, DummyFort7)
    assert isinstance(x_obj, DummyXyz)
    assert f1_obj.read_silent is True
    assert f7_obj.read_silent is True
    assert x_obj.read_silent is True


def test_open_fort_files_with_which_zero_uses_neighbor_paths(patched_file_classes, tmp_path):
    f1 = tmp_path / "fort.1"
    f7 = tmp_path / "fort.7"
    x = tmp_path / "fort.8"
    for p in (f1, f7, x):
        p.write_text("dummy")

    f1_obj, f7_obj, x_obj = occam_data._open_fort_files(str(f1), None, None, silent=False, which=0)
    assert f1_obj.file_name == str(f1)
    assert f7_obj.file_name == str(f7)
    assert x_obj.file_name == str(x)
    assert f1_obj.read_silent is False


def test_check_constructor_input_with_instances_returns_same_objects(patched_file_classes):
    f1 = DummyFort1("a")
    f7 = DummyFort7("b")
    x = DummyXyz("c")
    out = occam_data._check_constructor_input(f1, f7, x, silent=True)
    assert out == (f1, f7, x)


def test_check_constructor_input_single_non_string_raises_type_error():
    with pytest.raises(TypeError):
        occam_data._check_constructor_input(123)


def test_check_constructor_input_single_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "nope.1"
    with pytest.raises(FileNotFoundError):
        occam_data._check_constructor_input(str(missing))


def test_occamdata_init_from_three_paths_sets_attributes_and_consistency(patched_file_classes, tmp_path):
    f1 = tmp_path / "fort.1"
    f7 = tmp_path / "fort.7"
    x = tmp_path / "fort.8"
    for p in (f1, f7, x):
        p.write_text("dummy")

    class FakeFort1(DummyFort1):
        def __init__(self, file_name):
            super().__init__(file_name, shared=3.14, name="same", arr=np.array([1, 2]))
    class FakeFort7(DummyFort7):
        def __init__(self, file_name):
            super().__init__(file_name, shared=3.14, name="same", arr=np.array([1, 2]))
    class FakeXyz(DummyXyz):
        def __init__(self, file_name):
            super().__init__(file_name, shared=3.14, name="same", arr=np.array([1, 2]))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(occam_data, "Fort1", FakeFort1)
    monkeypatch.setattr(occam_data, "Fort7", FakeFort7)
    monkeypatch.setattr(occam_data, "Xyz", FakeXyz)
    try:
        obj = occam_data.OccamData(str(f1), str(f7), str(x), load_from_npy=False, save_to_npy=False)
    finally:
        monkeypatch.undo()

    assert obj.consistent is True
    assert obj.shared == 3.14
    assert obj.name == "same"
    assert np.array_equal(obj.arr, np.array([1, 2]))
    assert obj.fort1_file_name == str(f1)
    assert obj.fort7_file_name == str(f7)
    assert obj.xyz_file_name == str(x)


def test_occamdata_init_single_path_loads_from_npy_if_present(patched_file_classes, tmp_path, monkeypatch):
    base = tmp_path
    class_dir = base / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "value.npy", np.array([10, 20]))

    class_file = class_dir / "class.json"
    class_file.write_text(json.dumps({"fort1_file_name": str(base / "fort.1")}))

    called = {"ctor": False}

    def boom(*args, **kwargs):
        called["ctor"] = True
        raise AssertionError("_check_constructor_input should not be called")

    monkeypatch.setattr(occam_data, "_check_constructor_input", boom)
    obj = occam_data.OccamData(str(base), load_from_npy=True, save_to_npy=False)
    assert called["ctor"] is False
    assert np.array_equal(obj.value, np.array([10, 20]))
    assert obj.fort1_file_name == str(base / "fort.1")


def test_save_creates_class_data_and_roundtrips_numpy_and_json(patched_file_classes, tmp_path):
    f1 = tmp_path / "fort.1"
    f7 = tmp_path / "fort.7"
    x = tmp_path / "fort.8"
    for p in (f1, f7, x):
        p.write_text("dummy")

    obj = occam_data.OccamData(str(f1), str(f7), str(x), load_from_npy=False, save_to_npy=False)
    obj.arr = np.array([1, 2, 3])
    obj.scalar = 5
    assert obj.save() is True

    save_dir = tmp_path / "class_data"
    assert save_dir.is_dir()
    assert (save_dir / "arr.npy").exists()
    assert (save_dir / "class.json").exists()

    with open(save_dir / "class.json", "r") as f:
        data = json.load(f)
    assert "arr" not in data
    assert data["scalar"] == 5

    assert isinstance(obj.arr, np.ndarray)
    assert np.array_equal(obj.arr, np.array([1, 2, 3]))


def test_save_returns_false_when_directory_exists_and_no_overwrite(patched_file_classes, tmp_path):
    f1 = tmp_path / "fort.1"
    f7 = tmp_path / "fort.7"
    x = tmp_path / "fort.8"
    for p in (f1, f7, x):
        p.write_text("dummy")

    obj = occam_data.OccamData(str(f1), str(f7), str(x), load_from_npy=False, save_to_npy=False)
    obj.arr = np.array([1])
    assert obj.save() is True
    obj.arr = np.array([2])
    assert obj.save(overwrite=False) is False


def test_load_restores_arrays_and_class_attributes(patched_file_classes, tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "a.npy", np.array([4, 5]))
    with open(class_dir / "class.json", "w") as f:
        json.dump({"fort1_file_name": "f1", "foo": "bar"}, f)

    obj = occam_data.OccamData.__new__(occam_data.OccamData)
    obj.load(str(class_dir), silent=True)
    assert np.array_equal(obj.a, np.array([4, 5]))
    assert obj.fort1_file_name == "f1"
    assert obj.foo == "bar"
