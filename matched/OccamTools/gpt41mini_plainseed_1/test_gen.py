import os
import shutil
import tempfile
import json
import numpy as np
import pytest
from occamtools.occam_data import (
    _check_internal_consistency,
    _check_internal_consistency_all,
    _open_fort_files,
    _check_npy_dump_exists,
    _check_constructor_input,
    OccamData,
)
from occamtools.read_xyz import Xyz
from occamtools.read_fort1 import Fort1
from occamtools.read_fort7 import Fort7


# Helpers for tests
class DummyFort1(Fort1):
    def __init__(self, file_name):
        self.file_name = file_name
        self.some_attr = 123.0
        self.str_attr = "abc"
        self.num_lines = 10  # ignored attribute

    def read_file(self, silent=False):
        pass


class DummyFort7(Fort7):
    def __init__(self, file_name):
        self.file_name = file_name
        self.some_attr = 123.0
        self.str_attr = "abc"

    def read_file(self, silent=False):
        pass


class DummyXyz(Xyz):
    def __init__(self, file_name):
        self.file_name = file_name
        self.some_attr = 123.0
        self.str_attr = "abc"

    def read_file(self, silent=False):
        pass


def test_check_internal_consistency_identical_attributes_no_warning():
    a = DummyFort1("file1")
    b = DummyFort1("file2")
    # identical some_attr and str_attr, ignored keys skipped
    with pytest.warns(None) as record:
        res = _check_internal_consistency(a, b)
    assert res is True
    assert len(record) == 0


def test_check_internal_consistency_different_float_warns():
    a = DummyFort1("file1")
    b = DummyFort1("file2")
    b.some_attr = 321.0  # different float value
    with pytest.warns(UserWarning) as w:
        res = _check_internal_consistency(a, b)
    assert res is False
    assert any("Data is not internally consistent" in str(warn.message) for warn in w)


def test_check_internal_consistency_different_str_warns():
    a = DummyFort1("file1")
    b = DummyFort1("file2")
    b.str_attr = "different"
    with pytest.warns(UserWarning):
        res = _check_internal_consistency(a, b)
    assert res is False


def test_check_internal_consistency_ignores_file_name_and_num_lines():
    a = DummyFort1("file1")
    b = DummyFort1("file2")
    a.file_name = "name1"
    b.file_name = "name2"
    a.num_lines = 5
    b.num_lines = 15
    # Changes in ignored attributes should not warn or fail
    with pytest.warns(None):
        res = _check_internal_consistency(a, b)
    assert res is True


def test_check_internal_consistency_all_all_consistent(monkeypatch):
    f1 = DummyFort1("f1")
    f7 = DummyFort7("f7")
    xyz = DummyXyz("xyz")

    result = _check_internal_consistency_all(f1, f7, xyz)
    assert result is True

    # monkeypatch to cause one inconsistency
    def fake_check(a, b):  # return False once
        if getattr(fake_check, "called", False):
            return True
        fake_check.called = True
        return False
    fake_check.called = False

    monkeypatch.setattr("occamtools.occam_data._check_internal_consistency",
                        fake_check)

    result2 = _check_internal_consistency_all(f1, f7, xyz)
    assert result2 is False


def test_open_fort_files_with_none_and_which(monkeypatch):
    def dummy_f1_init(file_name):
        obj = DummyFort1(file_name)
        return obj

    def dummy_f7_init(file_name):
        obj = DummyFort7(file_name)
        return obj

    def dummy_xyz_init(file_name):
        obj = DummyXyz(file_name)
        return obj

    monkeypatch.setattr("occamtools.occam_data.Fort1", dummy_f1_init)
    monkeypatch.setattr("occamtools.occam_data.Fort7", dummy_f7_init)
    monkeypatch.setattr("occamtools.occam_data.Xyz", dummy_xyz_init)

    # Use temp directory and files
    tmpdir = tempfile.mkdtemp()
    try:
        f1_path = os.path.join(tmpdir, "fort.1")
        f7_path = os.path.join(tmpdir, "fort.7")
        xyz_path = os.path.join(tmpdir, "fort.8")
        for p in (f1_path, f7_path, xyz_path):
            with open(p, "w") as f:
                f.write("dummy")

        f1, f7, x = _open_fort_files(f1_path, f7_path, xyz_path, silent=True)
        for obj in (f1, f7, x):
            assert hasattr(obj, "file_name")

        # which=0 uses fort1 path, constructs other paths based on dirname
        f1, f7, x = _open_fort_files(f1_path, f7_path, xyz_path, silent=True, which=0)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)
        assert f7.file_name.endswith("fort.7")
        assert x.file_name.endswith("fort.8")

        # which=1 uses fort7 path
        f1, f7, x = _open_fort_files(f1_path, f7_path, xyz_path, silent=True, which=1)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)
        assert f1.file_name.endswith("fort.1")

        # which=2 uses xyz path
        f1, f7, x = _open_fort_files(f1_path, f7_path, xyz_path, silent=True, which=2)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)
        assert x.file_name.endswith("fort.8")
    finally:
        shutil.rmtree(tmpdir)


def test_check_npy_dump_exists_file_and_dir(tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    npy_file = class_dir / "x.npy"
    np.save(npy_file, np.array([1, 2, 3]))

    # When path is a directory, class_data exists with x.npy
    exists, cdir = _check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert cdir == str(class_dir)

    # When path is a file inside tmp_path
    file_path = tmp_path / "somefile.txt"
    file_path.write_text("hello")
    exists2, cdir2 = _check_npy_dump_exists(str(file_path))
    assert exists2 is True
    assert cdir2 == str(class_dir)

    # When no class_data directory, should return False
    new_dir = tmp_path / "newdir"
    new_dir.mkdir()
    exists3, cdir3 = _check_npy_dump_exists(str(new_dir))
    assert exists3 is False
    assert cdir3 == os.path.join(str(new_dir), "class_data")

    # Test FileNotFoundError raised when path does not exist
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / "non_existing_file"))


def test_check_constructor_input_str_paths(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy fort files in tmpdir
        for fname in ("fort.1", "fort.7", "fort.8"):
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write("dummy")

        # Patch Fort1, Fort7, Xyz constructors to dummy classes
        monkeypatch.setattr("occamtools.occam_data.Fort1", lambda p: DummyFort1(p))
        monkeypatch.setattr("occamtools.occam_data.Fort7", lambda p: DummyFort7(p))
        monkeypatch.setattr("occamtools.occam_data.Xyz", lambda p: DummyXyz(p))

        f1_path = os.path.join(tmpdir, "fort.1")
        f7_path = os.path.join(tmpdir, "fort.7")
        xyz_path = os.path.join(tmpdir, "fort.8")

        # Test with 3 string paths, None which: opens all files normally
        f1, f7, x = _check_constructor_input(f1_path, f7_path, xyz_path, silent=True)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Test with 3 inputs, first is string, others not strings: triggers which=0
        f1, f7, x = _check_constructor_input(f1_path, DummyFort7(f7_path), DummyXyz(xyz_path), silent=True)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Test with 3 inputs, second is string: triggers which=1
        f1, f7, x = _check_constructor_input(DummyFort1(f1_path), f7_path, DummyXyz(xyz_path), silent=True)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Test with 3 inputs, third is string: triggers which=2
        f1, f7, x = _check_constructor_input(DummyFort1(f1_path), DummyFort7(f7_path), xyz_path, silent=True)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Test with invalid types triggers ValueError
        with pytest.raises(ValueError):
            _check_constructor_input(DummyFort1(f1_path), DummyFort7(f7_path), DummyFort1(f1_path), silent=True)

    finally:
        shutil.rmtree(tmpdir)


def test_check_constructor_input_one_arg(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy fort files in tmpdir
        for fname in ("fort.1", "fort.7", "fort.8"):
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write("dummy")

        monkeypatch.setattr("occamtools.occam_data.Fort1", lambda p: DummyFort1(p))
        monkeypatch.setattr("occamtools.occam_data.Fort7", lambda p: DummyFort7(p))
        monkeypatch.setattr("occamtools.occam_data.Xyz", lambda p: DummyXyz(p))

        # Provide fort.1 path string
        f1_path = os.path.join(tmpdir, "fort.1")
        f7_path = os.path.join(tmpdir, "fort.7")
        xyz_path = os.path.join(tmpdir, "fort.8")

        f1, f7, x = _check_constructor_input(f1_path)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Provide fort.7 path string
        f1, f7, x = _check_constructor_input(f7_path)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Provide fort.8 path string
        f1, f7, x = _check_constructor_input(xyz_path)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Provide directory name - also works
        f1, f7, x = _check_constructor_input(tmpdir)
        assert isinstance(f1, DummyFort1)
        assert isinstance(f7, DummyFort7)
        assert isinstance(x, DummyXyz)

        # Provide non-string argument raises TypeError
        with pytest.raises(TypeError):
            _check_constructor_input(DummyFort1(f1_path))

        # Provide string to non-existing file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            _check_constructor_input("/non/existent/path/fort.1")
    finally:
        shutil.rmtree(tmpdir)


def test_occamdata_init_loads_and_saves(tmp_path, monkeypatch):
    # Setup dummy Fort objects with some numpy arrays as attributes
    arr = np.array([1, 2, 3])

    class DummyFort(Fort1):
        def __init__(self, file_name):
            self.file_name = file_name
            self.some_array = arr
            self.some_num = 42
            self.num_lines = 1

        def read_file(self, silent=False):
            pass

    # Patch Fort1, Fort7, Xyz to DummyFort with identical attributes for testing consistency
    monkeypatch.setattr("occamtools.occam_data.Fort1", lambda f: DummyFort(f))
    monkeypatch.setattr("occamtools.occam_data.Fort7", lambda f: DummyFort(f))
    monkeypatch.setattr("occamtools.occam_data.Xyz", lambda f: DummyFort(f))

    # Patch _check_internal_consistency_all to always True to avoid warnings here
    monkeypatch.setattr("occamtools.occam_data._check_internal_consistency_all", lambda f1, f7, x: True)

    # Create fake fort files
    (tmp_path / "fort.1").write_text("dummy")
    (tmp_path / "fort.7").write_text("dummy")
    (tmp_path / "fort.8").write_text("dummy")

    # Instantiate OccamData with paths of fort files, should read and save
    od = OccamData(str(tmp_path), load_from_npy=False, save_to_npy=True, silent=True)

    # Attributes from dummy fort should be set, except ignored attrs
    assert hasattr(od, "some_num")
    # numpy attribute should be saved and then deleted
    assert not hasattr(od, "some_array")

    # Save again with overwrite=False should return False because directory exists
    res = od.save(overwrite=False)
    assert res is False

    # Save again with overwrite=True should return True and save arrays
    res2 = od.save(overwrite=True)
    assert res2 is True


def test_occamdata_load_and_internal_methods(tmp_path, monkeypatch):
    # Setup: create class_data directory with .npy files and class.json
    class_data_dir = tmp_path / "class_data"
    class_data_dir.mkdir()
    arr = np.arange(5)
    np.save(class_data_dir / "some_array.npy", arr)
    json_data = {"attr1": "value1", "attr2": 123}
    with open(class_data_dir / "class.json", "w") as f:
        json.dump(json_data, f)

    od = OccamData.__new__(OccamData)  # bypass __init__

    # Test _load_arrays with silent=True: no output but sets attribute
    od._load_arrays(str(class_data_dir), silent=True)
    assert hasattr(od, "some_array")
    np.testing.assert_array_equal(getattr(od, "some_array"), arr)

    # Test _load_class sets class attributes
    od._load_class(str(class_data_dir))
    assert getattr(od, "attr1") == "value1"
    assert getattr(od, "attr2") == 123

    # Test _save_arrays creates files
    od.fort1_file_name = str(tmp_path / "fort.1")
    od.save_dir = "class_data"
    od.some_array = arr
    od.another_array = np.array([10, 20])
    od._save_arrays()
    saved_files = os.listdir(class_data_dir)
    assert "some_array.npy" in saved_files
    assert "another_array.npy" in saved_files

    # Test _delete_array_attributes deletes numpy arrays
    od._delete_array_attributes()
    assert not hasattr(od, "some_array")
    assert not hasattr(od, "another_array")

    # Test _save_class writes JSON file including current attributes
    od.some_attr = "value"
    od._save_class()
    class_json = json.load(open(class_data_dir / "class.json"))
    assert "some_attr" in class_json
    assert class_json["some_attr"] == "value"


def test_occamdata_init_loads_existing_npy(monkeypatch, tmp_path):
    # arrange: create dummy class_data directory to simulate saved data
    class_data_dir = tmp_path / "class_data"
    class_data_dir.mkdir()
    with open(class_data_dir / "class.json", "w") as f:
        json.dump({"attr": 1}, f)
    np.save(class_data_dir / "x.npy", np.array([1, 2, 3]))

    # Patch _check_npy_dump_exists to True
    def fake_check(file_name):
        return True, str(class_data_dir)
    monkeypatch.setattr("occamtools.occam_data._check_npy_dump_exists", fake_check)

    # create OccamData instance with one arg: load_from_npy True triggers load
    od = OccamData.__new__(OccamData)
    called = {}

    # patch load to record call but call real method
    def fake_load(self, class_path, silent=False):
        called['loaded'] = True
        # call original load to ensure attributes set
        OccamData._load_arrays(self, class_path, silent)
        OccamData._load_class(self, class_path)

    monkeypatch.setattr(OccamData, "load", fake_load)

    # Call __init__ with one arg, load_from_npy=True (default)
    od.__init__(str(tmp_path))
    assert called.get('loaded') is True
    assert hasattr(od, "attr") and od.attr == 1
    assert hasattr(od, "x")
    np.testing.assert_array_equal(od.x, np.array([1, 2, 3]))


def test_occamdata_save_creates_class_data_dir(tmp_path):
    # Setup OccamData instance manually
    od = OccamData.__new__(OccamData)
    od.fort1_file_name = str(tmp_path / "fort.1")
    od.save_dir = "class_data"
    # Create a dummy numpy attribute to be saved
    od.dummy_array = np.array([0, 1, 2])
    # Save without class_data directory present
    saved = od.save(overwrite=False)
    assert saved is True
    class_data_dir = tmp_path / "class_data"
    assert class_data_dir.exists()
    # Files should be created for numpy arrays and class.json should exist
    files = os.listdir(class_data_dir)
    assert "dummy_array.npy" in files
    assert "class.json" in files


def test_occamdata_save_returns_false_if_exists(tmp_path):
    od = OccamData.__new__(OccamData)
    od.fort1_file_name = str(tmp_path / "fort.1")
    od.save_dir = "class_data"
    class_data_dir = tmp_path / "class_data"
    class_data_dir.mkdir()

    res = od.save(overwrite=False)
    assert res is False


def test_occamdata_load_missing_class_json(tmp_path):
    od = OccamData.__new__(OccamData)
    os.makedirs(tmp_path, exist_ok=True)

    # Simulate directory without class.json but with npy files
    np.save(tmp_path / "some_array.npy", np.array([1]))
    # Remove class.json if exists
    json_path = tmp_path / "class.json"
    if json_path.exists():
        json_path.unlink()

    # _load_arrays should work, _load_class should raise because no json file
    od._load_arrays(str(tmp_path), silent=True)
    with pytest.raises(FileNotFoundError):
        od._load_class(str(tmp_path))
