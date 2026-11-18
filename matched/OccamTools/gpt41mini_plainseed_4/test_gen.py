import os
import shutil
import tempfile
import json
import numpy as np
import pytest
import warnings

from occamtools.occam_data import (OccamData, _check_internal_consistency,
                                   _check_internal_consistency_all,
                                   _open_fort_files, _check_npy_dump_exists,
                                   _check_constructor_input)
from occamtools.read_fort1 import Fort1
from occamtools.read_fort7 import Fort7
from occamtools.read_xyz import Xyz, _are_floats


# --- Helpers for test objects ---

class DummyFort1:
    def __init__(self, file_name='fort.1'):
        self.file_name = file_name
        self.num_lines = 10
        self.attribute1 = 3.14
        self.attribute2 = "test"
    def read_file(self, silent=False):
        pass

class DummyFort7:
    def __init__(self, file_name='fort.7'):
        self.file_name = file_name
        self.num_lines = 9
        self.attribute1 = 3.14
        self.attribute2 = "test"
    def read_file(self, silent=False):
        pass

class DummyXyz:
    def __init__(self, file_name='fort.8'):
        self.file_name = file_name
        self.num_lines = 8
        self.attribute1 = 3.14
        self.attribute2 = "test"
    def read_file(self, silent=False):
        pass


# Patch Fort1, Fort7, Xyz to Dummy classes where needed
@pytest.fixture(autouse=True)
def patch_occamtools(monkeypatch):
    monkeypatch.setattr("occamtools.occam_data.Fort1", lambda fn: DummyFort1(fn))
    monkeypatch.setattr("occamtools.occam_data.Fort7", lambda fn: DummyFort7(fn))
    monkeypatch.setattr("occamtools.occam_data.Xyz", lambda fn: DummyXyz(fn))


# ----------------------------
# Tests for _check_internal_consistency
def test_check_internal_consistency_identical_no_warning():
    a = DummyFort1()
    b = DummyFort7()
    b.attribute1 = a.attribute1
    b.attribute2 = a.attribute2
    a.file_name = 'fileA'
    b.file_name = 'fileB'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warning should be emitted
        assert _check_internal_consistency(a, b) is True

def test_check_internal_consistency_float_mismatch_warns():
    a = DummyFort1()
    b = DummyFort7()
    b.attribute1 = a.attribute1 + 1
    a.file_name = 'fileA'
    b.file_name = 'fileB'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _check_internal_consistency(a, b)
        assert not result
        assert len(w) == 1
        assert "Data is not internally consistent" in str(w[0].message)
        assert "fileA" in str(w[0].message)
        assert "fileB" in str(w[0].message)

def test_check_internal_consistency_string_mismatch_warns():
    a = DummyFort1()
    b = DummyFort7()
    b.attribute2 = "different"
    a.file_name = 'A'
    b.file_name = 'B'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _check_internal_consistency(a, b)
        assert not result
        assert len(w) == 1

def test_check_internal_consistency_ignores_file_name_and_num_lines():
    a = DummyFort1()
    b = DummyFort7()
    # Change ignored attrs, no warning expected
    a.file_name = 'fileA'
    b.file_name = 'fileB'
    a.num_lines = 1
    b.num_lines = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _check_internal_consistency(a, b) is True


# ------------------------------
# Tests for _check_internal_consistency_all

def test_check_internal_consistency_all_all_consistent():
    a = DummyFort1()
    b = DummyFort7()
    c = DummyXyz()
    # all attribute1 and attribute2 match
    assert _check_internal_consistency_all(a, b, c) is True

def test_check_internal_consistency_all_any_inconsistent(monkeypatch):
    a = DummyFort1()
    b = DummyFort7()
    c = DummyXyz()
    # Make one attribute differ in b to cause inconsistency
    b.attribute1 = a.attribute1 + 5
    assert _check_internal_consistency_all(a, b, c) is False

# ------------------------------
# Tests for _open_fort_files with different "which"

def test_open_fort_files_none(monkeypatch):
    f1, f7, x = _open_fort_files("f1_path", "f7_path", "xyz_path", silent=True)
    assert isinstance(f1, DummyFort1)
    assert isinstance(f7, DummyFort7)
    assert isinstance(x, DummyXyz)
    # files were passed as is
    assert f1.file_name == "f1_path"
    assert f7.file_name == "f7_path"
    assert x.file_name == "xyz_path"

def test_open_fort_files_which_0(monkeypatch, tmp_path):
    # Setup dummy fort1 path, fort.7 and fort.8 will be inferred
    f1_path = tmp_path / "folder" / "fort.1"
    f1_path.parent.mkdir(parents=True, exist_ok=True)
    f1_path.write_text("dummy fort1 data")
    f1, f7, x = _open_fort_files(str(f1_path), None, None, silent=True, which=0)
    assert isinstance(f1, DummyFort1)
    assert isinstance(f7, DummyFort7)
    assert isinstance(x, DummyXyz)
    assert f1.file_name == str(f1_path)
    assert os.path.basename(f7.file_name) == "fort.7"
    assert os.path.basename(x.file_name) == "fort.8"

def test_open_fort_files_which_1(monkeypatch, tmp_path):
    f7_path = tmp_path / "folder" / "fort.7"
    f7_path.parent.mkdir(parents=True, exist_ok=True)
    f7_path.write_text("dummy fort7 data")
    f1, f7, x = _open_fort_files(None, str(f7_path), None,
                                silent=True, which=1)
    assert isinstance(f1, DummyFort1)
    assert isinstance(f7, DummyFort7)
    assert isinstance(x, DummyXyz)
    assert os.path.basename(f1.file_name) == "fort.1"
    assert f7.file_name == str(f7_path)
    assert os.path.basename(x.file_name) == "fort.8"

def test_open_fort_files_which_2(monkeypatch, tmp_path):
    xyz_path = tmp_path / "folder" / "fort.8"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_path.write_text("dummy xyz data")
    f1, f7, x = _open_fort_files(None, None, str(xyz_path),
                                   silent=True, which=2)
    assert isinstance(f1, DummyFort1)
    assert isinstance(f7, DummyFort7)
    assert isinstance(x, DummyXyz)
    assert os.path.basename(f1.file_name) == "fort.1"
    assert os.path.basename(f7.file_name) == "fort.7"
    assert x.file_name == str(xyz_path)

# ------------------------------
# Tests for _check_npy_dump_exists

def test_check_npy_dump_exists_directory_with_class_data(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    class_dir = d / "class_data"
    class_dir.mkdir()
    (class_dir / "x.npy").write_bytes(b"dummy")
    exists, path = _check_npy_dump_exists(str(d))
    assert exists is True
    assert path == str(class_dir)

def test_check_npy_dump_exists_file_with_class_data(tmp_path):
    f = tmp_path / "somefile"
    f.write_text("data")
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    (class_dir / "x.npy").write_bytes(b"dummy")
    exists, path = _check_npy_dump_exists(str(f))
    assert exists is True
    assert path == str(class_dir)

def test_check_npy_dump_exists_no_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / "non_existent_file"))

def test_check_npy_dump_exists_no_class_data(tmp_path):
    f = tmp_path / "somefile"
    f.write_text("data")
    exists, path = _check_npy_dump_exists(str(f))
    assert exists is False
    assert path == str(tmp_path / "class_data")

# ------------------------------
# Tests for _check_constructor_input with different input types and lengths

def test_check_constructor_input_objects(monkeypatch):
    fort1 = DummyFort1()
    fort7 = DummyFort7()
    xyz = DummyXyz()
    out = _check_constructor_input(fort1, fort7, xyz)
    # Returns same objects
    assert out == (fort1, fort7, xyz)

def test_check_constructor_input_strings(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    # Create dummy files to prevent read errors
    (d / "fort.1").write_text("f1")
    (d / "fort.7").write_text("f7")
    (d / "fort.8").write_text("f8")
    out = _check_constructor_input(str(d / "fort.1"),
                                  str(d / "fort.7"),
                                  str(d / "fort.8"))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_partial_strings_which_0(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f1_path = d / "fort.1"
    f1_path.write_text("f1")
    out = _check_constructor_input(str(f1_path), None, None)
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_partial_strings_which_1(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f7_path = d / "fort.7"
    f7_path.write_text("f7")
    out = _check_constructor_input(None, str(f7_path), None)
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_partial_strings_which_2(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    xyz_path = d / "fort.8"
    xyz_path.write_text("f8")
    out = _check_constructor_input(None, None, str(xyz_path))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_invalid_raises():
    with pytest.raises(ValueError):
        _check_constructor_input(1, 2, 3)

def test_check_constructor_input_single_path_str(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    for fname in ("fort.1", "fort.7", "fort.8"):
        (d / fname).write_text(fname)
    # supply directory path, triggers isdir branch
    out = _check_constructor_input(str(d))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_single_path_non_str_raises():
    with pytest.raises(TypeError):
        _check_constructor_input(123)

def test_check_constructor_input_single_path_not_exist(tmp_path):
    not_exist = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        _check_constructor_input(str(not_exist))

def test_check_constructor_input_single_path_extension_1(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f1 = d / "fort.1"
    f7 = d / "fort.7"
    f8 = d / "fort.8"
    f1.write_text("f1")
    f7.write_text("f7")
    f8.write_text("f8")
    out = _check_constructor_input(str(f1))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_single_path_extension_7(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f7 = d / "fort.7"
    f1 = d / "fort.1"
    f8 = d / "fort.8"
    f7.write_text("f7")
    f1.write_text("f1")
    f8.write_text("f8")
    out = _check_constructor_input(str(f7))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_single_path_extension_8(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f8 = d / "fort.8"
    f1 = d / "fort.1"
    f7 = d / "fort.7"
    f8.write_text("f8")
    f1.write_text("f1")
    f7.write_text("f7")
    out = _check_constructor_input(str(f8))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)

def test_check_constructor_input_single_path_extension_xyz(monkeypatch, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    xyz = d / "test.xyz"
    f1 = d / "fort.1"
    f7 = d / "fort.7"
    xyz.write_text("xyz")
    f1.write_text("f1")
    f7.write_text("f7")
    out = _check_constructor_input(str(xyz))
    assert all(isinstance(x, (DummyFort1, DummyFort7, DummyXyz)) for x in out)


# -------------------------
# Tests for OccamData.__init__, save, load

def test_occamdata_init_loads_from_npy(tmp_path, monkeypatch):
    # Arrange a fake npy dump
    d = tmp_path / "dir"
    d.mkdir()
    class_dir = d / "class_data"
    class_dir.mkdir()
    # save dummy npy file
    np.save(str(class_dir / "attribute1.npy"), np.array([1, 2, 3]))
    # save dummy class.json file
    json_path = class_dir / "class.json"
    with open(json_path, "w") as f:
        json.dump({"attribute2": "value", "fort1_file_name": "fort.1",
                   "fort7_file_name": "fort.7", "xyz_file_name": "fort.8",
                   "consistent": True}, f)

    data = OccamData(str(d), load_from_npy=True, save_to_npy=False, silent=True)
    # Loaded numpy array attribute1 and attribute2 from json
    assert hasattr(data, "attribute1")
    np.testing.assert_array_equal(data.attribute1, np.array([1, 2, 3]))
    assert getattr(data, "attribute2") == "value"
    assert data.fort1_file_name == "fort.1"
    # No exception triggered

def test_occamdata_init_without_npy(monkeypatch, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    # create fort files to satisfy _check_constructor_input
    (d / "fort.1").write_text("f1")
    (d / "fort.7").write_text("f7")
    (d / "fort.8").write_text("f8")

    saved_flag = {}

    def fake_save(self, overwrite=False):
        saved_flag['called'] = True
        return True
    monkeypatch.setattr(OccamData, "save", fake_save)

    data = OccamData(str(d), load_from_npy=True, save_to_npy=True, silent=True)
    assert hasattr(data, "fort1_file_name")
    assert saved_flag.get('called', False)

def test_occamdata_save_creates_class_data_and_files(tmp_path, monkeypatch):
    d = tmp_path / "dir"
    d.mkdir()
    f1 = d / "fort.1"
    f7 = d / "fort.7"
    f8 = d / "fort.8"
    f1.write_text("f1")
    f7.write_text("f7")
    f8.write_text("f8")

    data = OccamData(str(d), load_from_npy=False, save_to_npy=False, silent=True)
    # Add numpy arrays to data attributes
    data.some_array = np.array([1, 2, 3])
    # Setup save_path manually since save() needs fort1_file_name attribute
    data.fort1_file_name = str(f1)
    # Call save, should create class_data directory and save files
    result = data.save(overwrite=True)
    assert result is True
    class_data_path = os.path.join(os.path.dirname(data.fort1_file_name), "class_data")
    assert os.path.isdir(class_data_path)
    # numpy array file saved
    npy_file = os.path.join(class_data_path, "some_array.npy")
    assert os.path.isfile(npy_file)
    # class.json file saved
    class_json = os.path.join(class_data_path, "class.json")
    assert os.path.isfile(class_json)
    # numpy array attribute should be deleted after saving
    assert not hasattr(data, "some_array")

def test_occamdata_save_no_overwrite_returns_false(tmp_path, monkeypatch):
    d = tmp_path / "dir"
    d.mkdir()
    f1 = d / "fort.1"
    f1.write_text("f1")
    data = OccamData(str(d), load_from_npy=False, save_to_npy=False, silent=True)
    data.fort1_file_name = str(f1)
    class_data_path = os.path.join(os.path.dirname(data.fort1_file_name), "class_data")
    os.mkdir(class_data_path)
    # Create dummy file to simulate existing data
    with open(os.path.join(class_data_path, "dummy.npy"), "w") as f:
        f.write("")

    # Call save without overwrite - should return False
    result = data.save(overwrite=False)
    assert result is False

def test_occamdata_load_loads_arrays_and_class(monkeypatch, tmp_path):
    d = tmp_path / "dir"
    class_dir = d / "class_data"
    class_dir.mkdir(parents=True)
    np.save(class_dir / "arr.npy", np.array([1, 2]))
    with open(class_dir / "class.json", "w") as f:
        json.dump({"foo": "bar", "fort1_file_name": "fort.1",
                   "fort7_file_name": "fort.7", "xyz_file_name": "fort.8",
                   "consistent": True}, f)
    data = OccamData.__new__(OccamData)  # bypass __init__
    data.load(str(class_dir), silent=True)
    assert hasattr(data, "arr")
    np.testing.assert_array_equal(data.arr, np.array([1, 2]))
    assert getattr(data, "foo") == "bar"

def test_private_methods_save_delete_and_load(monkeypatch, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    f1 = d / "fort.1"
    f1.write_text("f1")

    data = OccamData(str(d), load_from_npy=False, save_to_npy=False, silent=True)
    data.fort1_file_name = str(f1)
    data.save_path = os.path.join(os.path.dirname(data.fort1_file_name), data.save_dir)
    os.mkdir(data.save_path)

    # Add numpy arrays to test _save_arrays and _delete_array_attributes
    data.arr1 = np.array([1])
    data.arr2 = np.array([2])
    data.other = "not array"

    data._save_arrays()
    files_before = os.listdir(data.save_path)
    assert "arr1.npy" in files_before
    assert "arr2.npy" in files_before

    data._delete_array_attributes()
    assert not hasattr(data, "arr1")
    assert not hasattr(data, "arr2")
    assert hasattr(data, "other")

    # test _save_class writes json file
    data._save_class()
    json_path = os.path.join(data.save_path, "class.json")
    assert os.path.isfile(json_path)

    # test _load_class loads json correctly
    # Create new dummy attribute to be overwritten
    data.dummy = None
    data._load_class(data.save_path)
    with open(json_path, "r") as f:
        attributes = json.load(f)
    for k, v in attributes.items():
        assert getattr(data, k) == v

def test__load_arrays_prints_with_tqdm(capsys, monkeypatch, tmp_path):
    # Setup data directory with two npy files and one non-npy file
    d = tmp_path / "dir"
    d.mkdir()
    np.save(d / "a.npy", np.array([1]))
    np.save(d / "b.npy", np.array([2]))
    (d / "not_npy.txt").write_text("nope")

    data = OccamData.__new__(OccamData)
    data._load_arrays(str(d), silent=False)
    captured = capsys.readouterr()
    assert "Loading data from .npy files" in captured.out
    assert hasattr(data, "a")
    assert hasattr(data, "b")
    assert not hasattr(data, "not_npy")

def test__load_arrays_silent(monkeypatch, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    np.save(d / "a.npy", np.array([1]))
    data = OccamData.__new__(OccamData)
    # Should not print anything in silent mode
    data._load_arrays(str(d), silent=True)
    assert hasattr(data, "a")

def test__open_fort_files_read_file_called(monkeypatch):
    called = {}

    class DummyF1(DummyFort1):
        def read_file(self, silent=False):
            called['f1'] = True
    class DummyF7(DummyFort7):
        def read_file(self, silent=False):
            called['f7'] = True
    class DummyX(DummyXyz):
        def read_file(self, silent=False):
            called['x'] = True
    monkeypatch.setattr("occamtools.occam_data.Fort1", lambda fn: DummyF1(fn))
    monkeypatch.setattr("occamtools.occam_data.Fort7", lambda fn: DummyF7(fn))
    monkeypatch.setattr("occamtools.occam_data.Xyz", lambda fn: DummyX(fn))

    f1, f7, x = _open_fort_files("f1", "f7", "x", silent=True)
    assert called['f1'] and called['f7'] and called['x']


# Test _are_floats helper sanity (not required but good coverage)
def test_are_floats_basic():
    assert _are_floats(1.0, 1.0)
    assert _are_floats(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    # Different types should return False
    assert not _are_floats(1, "1")
    assert not _are_floats("a", "b")
