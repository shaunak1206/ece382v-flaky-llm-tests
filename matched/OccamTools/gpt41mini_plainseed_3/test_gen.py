import os
import io
import json
import warnings
import builtins
import numpy as np
import pytest
from occamtools import occam_data
from occamtools.occam_data import OccamData, _check_constructor_input, _open_fort_files, _check_internal_consistency, _check_internal_consistency_all, _check_npy_dump_exists

# Fixtures to create simple dummy Fort1, Fort7, Xyz objects
class DummyBase:
    def __init__(self, file_name='dummy_file'):
        self.file_name = file_name
        self.some_value = 123
        self.array = np.array([1, 2, 3])
        self.str_attr = 'dummy'
        self.num_lines = 10

    def read_file(self, silent=False):
        pass


class DummyFort1(DummyBase):
    pass


class DummyFort7(DummyBase):
    pass


class DummyXyz(DummyBase):
    pass


@pytest.fixture
def dummy_fort1():
    return DummyFort1(file_name='fort.1')


@pytest.fixture
def dummy_fort7():
    return DummyFort7(file_name='fort.7')


@pytest.fixture
def dummy_xyz():
    return DummyXyz(file_name='fort.8')


def test_check_internal_consistency_pass(dummy_fort1, dummy_fort7):
    dummy_fort7.some_value = dummy_fort1.some_value
    dummy_fort7.str_attr = dummy_fort1.str_attr
    assert _check_internal_consistency(dummy_fort1, dummy_fort7) is True


def test_check_internal_consistency_warns_and_false(dummy_fort1, dummy_fort7):
    dummy_fort7.some_value = dummy_fort1.some_value + 1  # numeric mismatch
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _check_internal_consistency(dummy_fort1, dummy_fort7)
        assert result is False
        assert any("Data is not internally consistent" in str(warning.message) for warning in w)


def test_check_internal_consistency_ignore_keys(dummy_fort1, dummy_fort7):
    dummy_fort7.file_name = 'different_name'
    # file_name is ignored key so consistency should remain True
    assert _check_internal_consistency(dummy_fort1, dummy_fort7)


def test_check_internal_consistency_all_all_consistent(dummy_fort1, dummy_fort7, dummy_xyz):
    # all identical values
    dummy_fort7.some_value = dummy_fort1.some_value
    dummy_xyz.some_value = dummy_fort1.some_value
    assert _check_internal_consistency_all(dummy_fort1, dummy_fort7, dummy_xyz) is True


def test_check_internal_consistency_all_some_inconsistent(dummy_fort1, dummy_fort7, dummy_xyz):
    dummy_fort7.some_value = dummy_fort1.some_value + 1
    assert _check_internal_consistency_all(dummy_fort1, dummy_fort7, dummy_xyz) is False


def test_check_npy_dump_exists_dir_and_file(tmp_path):
    # simulate dir with class_data/x.npy
    dir_path = tmp_path / "testdir"
    dir_path.mkdir()
    class_data_dir = dir_path / "class_data"
    class_data_dir.mkdir()
    (class_data_dir / "x.npy").write_text("dummy")  # we only need the file to exist
    exists, class_dir = _check_npy_dump_exists(str(dir_path))
    assert exists is True
    assert class_dir == str(class_data_dir)

    # simulate file path: file exists, class_data/x.npy exists
    file_path = tmp_path / "file.txt"
    file_path.write_text("text")
    class_dir2 = tmp_path / "class_data"
    class_dir2.mkdir()
    (class_dir2 / "x.npy").write_text("dummy")
    exists2, class_dir_found = _check_npy_dump_exists(str(file_path))
    assert exists2 is True
    assert class_dir_found == str(class_dir2)


def test_check_npy_dump_exists_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / "nonexistent"))


def test_open_fort_files_with_no_which(monkeypatch, tmp_path):
    # Patch Fort1, Fort7, Xyz to dummy classes that have read_file
    calls = {}

    class FakeFort1:
        def __init__(self, filename):
            calls['f1_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls['f1_read'] = True

    class FakeFort7:
        def __init__(self, filename):
            calls['f7_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls['f7_read'] = True

    class FakeXyz:
        def __init__(self, filename):
            calls['x_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls['x_read'] = True

    monkeypatch.setattr(occam_data, 'Fort1', FakeFort1)
    monkeypatch.setattr(occam_data, 'Fort7', FakeFort7)
    monkeypatch.setattr(occam_data, 'Xyz', FakeXyz)

    f1_path = str(tmp_path / "fort.1")
    f7_path = str(tmp_path / "fort.7")
    x_path = str(tmp_path / "fort.8")

    f1, f7, x = _open_fort_files(f1_path, f7_path, x_path, silent=True)
    assert calls['f1_init'] == f1_path
    assert calls['f7_init'] == f7_path
    assert calls['x_init'] == x_path
    assert calls['f1_read'] and calls['f7_read'] and calls['x_read']


@pytest.mark.parametrize("which,expected_suffixes", [
    (0, ('fort.1', 'fort.7', 'fort.8')),
    (1, ('fort.1', 'fort.7', 'fort.8')),
    (2, ('fort.1', 'fort.7', 'fort.8')),
])
def test_open_fort_files_with_which(monkeypatch, tmp_path, which, expected_suffixes):
    calls = {}

    class FakeFort1:
        def __init__(self, filename):
            calls['f1_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls.setdefault('read_calls', []).append('f1')

    class FakeFort7:
        def __init__(self, filename):
            calls['f7_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls.setdefault('read_calls', []).append('f7')

    class FakeXyz:
        def __init__(self, filename):
            calls['x_init'] = filename
            self.file_name = filename

        def read_file(self, silent=False):
            calls.setdefault('read_calls', []).append('x')

    monkeypatch.setattr(occam_data, 'Fort1', FakeFort1)
    monkeypatch.setattr(occam_data, 'Fort7', FakeFort7)
    monkeypatch.setattr(occam_data, 'Xyz', FakeXyz)

    # Setup dummy file paths in the same directory
    base_dir = tmp_path / f"dir_{which}"
    base_dir.mkdir()
    base_file = base_dir / "basefile"
    base_file.write_text("test")

    # Provide arguments based on which index
    fort1_file = str(base_dir / "fort.1")
    fort7_file = str(base_dir / "fort.7")
    xyz_file = str(base_dir / "fort.8")

    # Create dummy files so no errors on os.path checks
    for file in (fort1_file, fort7_file, xyz_file):
        with open(file, 'w') as f:
            f.write("test")

    # Provide arguments partially as required depending on which param
    args = [fort1_file, fort7_file, xyz_file]
    if which == 0:
        args = [fort1_file, "ignored", "ignored"]
    elif which == 1:
        args = ["ignored", fort7_file, "ignored"]
    elif which == 2:
        args = ["ignored", "ignored", xyz_file]

    f1, f7, x = _open_fort_files(*args, silent=True, which=which)

    assert calls['f1_init'].endswith("fort.1")
    assert calls['f7_init'].endswith("fort.7")
    assert calls['x_init'].endswith("fort.8")
    assert set(calls['read_calls']) == {'f1', 'f7', 'x'}


def test_check_constructor_input_with_objects(dummy_fort1, dummy_fort7, dummy_xyz):
    # Passing Fort1, Fort7, Xyz instances returns them untouched
    f1, f7, x = _check_constructor_input(dummy_fort1, dummy_fort7, dummy_xyz, silent=True)
    assert f1 is dummy_fort1
    assert f7 is dummy_fort7
    assert x is dummy_xyz


def test_check_constructor_input_with_strings_and_which(monkeypatch, tmp_path):
    # We patch _open_fort_files to simplify testing
    def fake_open_fort_files(*args, **kwargs):
        return ("fort1_obj", "fort7_obj", "xyz_obj")

    monkeypatch.setattr(occam_data, "_open_fort_files", fake_open_fort_files)

    # Case: three strings
    a = "a"
    b = "b"
    c = "c"
    out = _check_constructor_input(a, b, c, silent=True)
    assert out == ("fort1_obj", "fort7_obj", "xyz_obj")

    # Case: first string only
    out = _check_constructor_input("fort1", dummy_fort7, dummy_xyz, silent=True)
    assert out == ("fort1_obj", "fort7_obj", "xyz_obj")

    # Case: second string only
    out = _check_constructor_input(dummy_fort1, "fort7", dummy_xyz, silent=True)
    assert out == ("fort1_obj", "fort7_obj", "xyz_obj")

    # Case: third string only
    out = _check_constructor_input(dummy_fort1, dummy_fort7, "xyz", silent=True)
    assert out == ("fort1_obj", "fort7_obj", "xyz_obj")


def test_check_constructor_input_raises_on_bad_input(dummy_fort1, dummy_fort7, dummy_xyz):
    with pytest.raises(ValueError):
        _check_constructor_input(1, 2, 3, silent=True)


def test_check_constructor_input_single_bad_type():
    with pytest.raises(TypeError):
        _check_constructor_input(123)


def test_check_constructor_input_single_nonexistent_file(tmp_path):
    file_path = tmp_path / "not_exist.xyz"
    with pytest.raises(FileNotFoundError):
        _check_constructor_input(str(file_path))


@pytest.mark.parametrize("ext, expected_first_file", [
    ('1', 'fort.1'),
    ('7', 'fort.1'),
    ('8', 'fort.1'),
    ('xyz', 'fort.1'),
])
def test_check_constructor_input_single_with_existing_files(monkeypatch, tmp_path, ext, expected_first_file):
    # Create dummy directory and files accordingly
    dir_a = tmp_path / "dir_ext"
    dir_a.mkdir()
    file_path = dir_a / f"fort.{ext}"
    file_path.write_text("dummy")

    # Create companion files fort.1, fort.7, fort.8
    (dir_a / "fort.1").write_text("fort1")
    (dir_a / "fort.7").write_text("fort7")
    (dir_a / "fort.8").write_text("fort8")

    # Patch _open_fort_files to record inputs and return dummy objects
    called = {}
    def fake_open_fort_files(fort1, fort7, xyz, silent):
        called['fort1'] = fort1
        called['fort7'] = fort7
        called['xyz'] = xyz
        return 'f1', 'f7', 'xyz'

    monkeypatch.setattr(occam_data, "_open_fort_files", fake_open_fort_files)

    out = _check_constructor_input(str(file_path), silent=True)
    assert out == ('f1', 'f7', 'xyz')
    # fort1 path should end with fort.1
    assert called['fort1'].endswith('fort.1')
    assert called['fort7'].endswith('fort.7')
    assert called['xyz'].endswith('fort.8')


def test_occamdata_init_loads_and_saves(tmp_path, monkeypatch, dummy_fort1, dummy_fort7, dummy_xyz):
    # Patch _check_npy_dump_exists to always return False so it loads from files
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda file_name: (False, None))

    # Patch save to track if called
    saved = {}

    def fake_save(self, overwrite=False):
        saved['called'] = True
        return True

    monkeypatch.setattr(OccamData, "save", fake_save)

    # Patch _check_constructor_input to return dummy objs and patch consistency check
    monkeypatch.setattr(occam_data, "_check_constructor_input", lambda *args, **kwargs: (dummy_fort1, dummy_fort7, dummy_xyz))
    monkeypatch.setattr(occam_data, "_check_internal_consistency_all", lambda f1, f7, x: True)

    od = OccamData(dummy_fort1, dummy_fort7, dummy_xyz, silent=True)
    # Check attributes copied (some_value and str_attr should be copied)
    assert od.some_value == dummy_fort1.some_value
    assert od.str_attr == dummy_fort1.str_attr
    assert od.fort1_file_name == dummy_fort1.file_name
    assert od.fort7_file_name == dummy_fort7.file_name
    assert od.xyz_file_name == dummy_xyz.file_name
    # Save was called
    assert saved.get('called', False)


def test_occamdata_init_loads_from_npy(monkeypatch):
    # Setup dummy paths
    path = "some_dir"

    # Patch _check_npy_dump_exists to return True to simulate npy load
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda fn: (True, fn))

    # Patch load method to track call and check silent argument
    called = {}
    def fake_load(self, class_path, silent=False):
        called['class_path'] = class_path
        called['silent'] = silent
    monkeypatch.setattr(OccamData, "load", fake_load)

    od = OccamData(path, silent=True)
    # load should be called with correct class_path
    assert called.get('class_path') == path
    assert called.get('silent') is True


def test_occamdata_save_creates_dir_and_returns_true(tmp_path, monkeypatch, dummy_fort1, dummy_fort7, dummy_xyz):
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda fn: (False, None))
    monkeypatch.setattr(occam_data, "_check_constructor_input", lambda *args, **kwargs: (dummy_fort1, dummy_fort7, dummy_xyz))
    monkeypatch.setattr(occam_data, "_check_internal_consistency_all", lambda f1, f7, x: True)
    od = OccamData(dummy_fort1, dummy_fort7, dummy_xyz, silent=True, save_to_npy=False)
    # Setup dummy fort1_file_name directory
    dir_path = tmp_path / "save_dir"
    dir_path.mkdir()
    setattr(od, 'fort1_file_name', str(tmp_path / "fort.1"))
    save_dir = os.path.join(os.path.dirname(od.fort1_file_name), od.save_dir)

    # Patch os.mkdir to track if called
    mkdir_called = {}
    original_mkdir = os.mkdir

    def fake_mkdir(path):
        mkdir_called['called'] = path
        original_mkdir(path)

    monkeypatch.setattr(os, 'mkdir', fake_mkdir)

    # Patch helper methods to ensure they run without error but don't save real files
    monkeypatch.setattr(od, '_save_arrays', lambda: None)
    monkeypatch.setattr(od, '_delete_array_attributes', lambda: None)
    monkeypatch.setattr(od, '_save_class', lambda: None)
    monkeypatch.setattr(od, '_load_arrays', lambda path, silent=True: None)

    # Remove save_dir if exists
    if os.path.exists(save_dir):
        try:
            os.rmdir(save_dir)
        except Exception:
            pass

    result = od.save(overwrite=False)
    assert result is True
    # mkdir should have been called for save_dir
    assert mkdir_called.get('called') == save_dir


def test_occamdata_save_existing_dir_no_overwrite(tmp_path, dummy_fort1, dummy_fort7, dummy_xyz, monkeypatch):
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda fn: (False, None))
    monkeypatch.setattr(occam_data, "_check_constructor_input", lambda *args, **kwargs: (dummy_fort1, dummy_fort7, dummy_xyz))
    monkeypatch.setattr(occam_data, "_check_internal_consistency_all", lambda f1, f7, x: True)
    od = OccamData(dummy_fort1, dummy_fort7, dummy_xyz, silent=True, save_to_npy=False)
    setattr(od, 'fort1_file_name', str(tmp_path / "fort.1"))
    save_dir = os.path.join(os.path.dirname(od.fort1_file_name), od.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Patch helper methods to track calls if called
    called = {}
    monkeypatch.setattr(od, '_save_arrays', lambda: called.setdefault('save_arrays', True))
    monkeypatch.setattr(od, '_delete_array_attributes', lambda: called.setdefault('delete_arrays', True))
    monkeypatch.setattr(od, '_save_class', lambda: called.setdefault('save_class', True))
    monkeypatch.setattr(od, '_load_arrays', lambda path, silent=True: called.setdefault('load_arrays', True))

    result = od.save(overwrite=False)
    # Should return False without saving because dir exists and overwrite is False
    assert result is False
    # None of the helper methods should be called
    assert not called


def test_load_arrays_creates_attributes(tmp_path, monkeypatch):
    # Setup a temp dir with .npy files and some other files
    test_dir = tmp_path / "load_test"
    test_dir.mkdir()
    np.save(str(test_dir / "attr1.npy"), np.array([1, 2, 3]))
    np.save(str(test_dir / "attr2.npy"), np.array([4, 5, 6]))
    (test_dir / "ignore.txt").write_text("ignore")

    od = OccamData.__new__(OccamData)

    # Patch tqdm to dummy context manager to suppress progress bar
    class DummyTqdm:
        def __init__(self, total):
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(self, n):
            pass

    monkeypatch.setattr(occam_data, "tqdm", DummyTqdm)

    # Run with silent=False and check that attributes are set correctly
    od._load_arrays(str(test_dir), silent=False)
    assert hasattr(od, 'attr1')
    assert hasattr(od, 'attr2')
    np.testing.assert_array_equal(getattr(od, 'attr1'), np.array([1, 2, 3]))
    np.testing.assert_array_equal(getattr(od, 'attr2'), np.array([4, 5, 6]))


def test_delete_array_attributes_removes_arrays():
    od = OccamData.__new__(OccamData)
    od.attr1 = np.array([1])
    od.attr2 = "string"
    od.attr3 = np.array([2])
    od._delete_array_attributes()
    assert not hasattr(od, 'attr1')
    assert hasattr(od, 'attr2')
    assert not hasattr(od, 'attr3')


def test_save_and_load_class(tmp_path):
    od = OccamData.__new__(OccamData)
    od.attr1 = 123
    od.attr2 = "abc"
    od.fort1_file_name = str(tmp_path / "fort.1")
    od.save_dir = "class_data"
    save_path = tmp_path / od.save_dir
    save_path.mkdir()

    od._save_class()
    json_file = save_path / "class.json"
    assert json_file.exists()

    # create new instance to load from file
    od2 = OccamData.__new__(OccamData)
    od2._load_class(str(save_path))
    assert od2.attr1 == 123
    assert od2.attr2 == "abc"


def test_occamdata_save_and_load_cycle(tmp_path, monkeypatch, dummy_fort1, dummy_fort7, dummy_xyz):
    # Construct OccamData normally with save_to_npy
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda fn: (False, None))
    monkeypatch.setattr(occam_data, "_check_constructor_input", lambda *args, **kwargs: (dummy_fort1, dummy_fort7, dummy_xyz))
    monkeypatch.setattr(occam_data, "_check_internal_consistency_all", lambda f1,f7,x: True)

    od = OccamData(dummy_fort1, dummy_fort7, dummy_xyz, silent=True, save_to_npy=False)
    od.attr_array = np.array([1, 2, 3])

    # Use tmp_path for files
    fort1_path = tmp_path / "fort.1"
    fort1_path.write_text("dummy")
    od.fort1_file_name = str(fort1_path)
    od.save_dir = "class_data"

    # Save
    result = od.save(overwrite=True)
    assert result is True

    # Delete array attribute to simulate state after save
    del od.attr_array

    # Load
    od.load(os.path.join(os.path.dirname(od.fort1_file_name), od.save_dir), silent=True)
    assert hasattr(od, "attr_array")
    np.testing.assert_array_equal(od.attr_array, np.array([1,2,3]))


def test_save_does_not_overwrite(tmp_path, dummy_fort1, dummy_fort7, dummy_xyz, monkeypatch):
    monkeypatch.setattr(occam_data, "_check_npy_dump_exists", lambda fn: (False, None))
    monkeypatch.setattr(occam_data, "_check_constructor_input", lambda *args, **kwargs: (dummy_fort1, dummy_fort7, dummy_xyz))
    monkeypatch.setattr(occam_data, "_check_internal_consistency_all", lambda f1, f7, x: True)
    od = OccamData(dummy_fort1, dummy_fort7, dummy_xyz, silent=True, save_to_npy=False)

    setattr(od, 'fort1_file_name', str(tmp_path / "fort.1"))
    save_dir = os.path.join(os.path.dirname(od.fort1_file_name), od.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    calls = {}
    monkeypatch.setattr(od, '_save_arrays', lambda: calls.setdefault('save_arrays', True))
    monkeypatch.setattr(od, '_delete_array_attributes', lambda: calls.setdefault('delete_arrays', True))
    monkeypatch.setattr(od, '_save_class', lambda: calls.setdefault('save_class', True))
    monkeypatch.setattr(od, '_load_arrays', lambda path, silent=True: calls.setdefault('load_arrays', True))

    # Should return False and not call save functions
    result = od.save(overwrite=False)
    assert result is False
    assert calls == {}

    # With overwrite=True save functions are called
    result = od.save(overwrite=True)
    assert result is True
    assert calls == {'save_arrays': True, 'delete_arrays': True, 'save_class': True, 'load_arrays': True}
