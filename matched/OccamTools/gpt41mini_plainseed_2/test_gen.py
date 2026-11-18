import os
import shutil
import json
import tempfile
import warnings
import numpy as np
import pytest
from occamtools import occam_data as od
from occamtools.read_xyz import Xyz
from occamtools.read_fort1 import Fort1
from occamtools.read_fort7 import Fort7


# Helpers to create dummy Fort1, Fort7, Xyz objects with minimal attributes
def create_dummy_fort1(file_name):
    f = Fort1.__new__(Fort1)
    f.file_name = file_name
    f.num_lines = 10
    f.some_float = 1.0
    f.some_str = "a"
    f.read_file = lambda silent=False: None
    f.__dict__['file_contents'] = "dummy"
    return f


def create_dummy_fort7(file_name):
    f = Fort7.__new__(Fort7)
    f.file_name = file_name
    f.num_lines = 10
    f.some_float = 1.0
    f.some_str = "a"
    f.read_file = lambda silent=False: None
    f.__dict__['file_contents'] = "dummy"
    return f


def create_dummy_xyz(file_name):
    x = Xyz.__new__(Xyz)
    x.file_name = file_name
    x.num_lines = 10
    x.some_float = 1.0
    x.some_str = "a"
    x.read_file = lambda silent=False: None
    x.__dict__['file_contents'] = "dummy"
    return x


# Patch Fort1, Fort7, and Xyz constructors to return dummy objects when called by _open_fort_files
@pytest.fixture(autouse=True)
def patch_file_classes(monkeypatch):
    monkeypatch.setattr(od, "Fort1", lambda f: create_dummy_fort1(f))
    monkeypatch.setattr(od, "Fort7", lambda f: create_dummy_fort7(f))
    monkeypatch.setattr(od, "Xyz", lambda f: create_dummy_xyz(f))


def test_check_internal_consistency_returns_true_when_all_same():
    f1 = create_dummy_fort1("fort.1")
    f7 = create_dummy_fort7("fort.7")
    xyz = create_dummy_xyz("fort.8")
    # all attributes match (some_float=1.0, some_str="a")
    assert od._check_internal_consistency(f1, f7)
    assert od._check_internal_consistency(f7, xyz)
    assert od._check_internal_consistency(f1, xyz)


def test_check_internal_consistency_warns_on_different_float_and_str():
    f1 = create_dummy_fort1("fort.1")
    f7 = create_dummy_fort7("fort.7")
    f7.some_float = 2.0  # different float
    f7.some_str = "b"    # different string

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = od._check_internal_consistency(f1, f7)
        # Should warn twice, once for float difference, once for str difference
        assert len(w) == 2
        assert not res


def test_check_internal_consistency_ignores_attributes_in_ignore_list():
    f1 = create_dummy_fort1("fort.1")
    f7 = create_dummy_fort7("fort.7")
    # Change file_name and num_lines, which are in ignore list
    f7.file_name = "different"
    f7.num_lines = 1000
    # Should be consistent because ignore list skips these
    assert od._check_internal_consistency(f1, f7)


def test_check_internal_consistency_all_returns_true_if_all_consistent():
    f1 = create_dummy_fort1("fort.1")
    f7 = create_dummy_fort7("fort.7")
    xyz = create_dummy_xyz("fort.8")
    assert od._check_internal_consistency_all(f1, f7, xyz)


def test_check_internal_consistency_all_returns_false_if_any_inconsistent():
    f1 = create_dummy_fort1("fort.1")
    f7 = create_dummy_fort7("fort.7")
    xyz = create_dummy_xyz("fort.8")
    xyz.some_float = 9.9
    assert not od._check_internal_consistency_all(f1, f7, xyz)


def test_open_fort_files_reads_files_correctly():
    # Provide file paths; will call patched constructors
    fort1_path = "path/fort.1"
    fort7_path = "path/fort.7"
    xyz_path = "path/fort.8"

    f1, f7, x = od._open_fort_files(fort1_path, fort7_path, xyz_path, silent=True)
    # read_file method called -> dummy does nothing
    # Confirm returned objects have .file_name set correctly
    assert f1.file_name == fort1_path
    assert f7.file_name == fort7_path
    assert x.file_name == xyz_path


@pytest.mark.parametrize("which, expected_filenames", [
    (0, ("path/fort.1", "path/fort.7", "path/fort.8")),
    (1, ("path/fort.1", "path/fort.7", "path/fort.8")),
    (2, ("path/fort.1", "path/fort.7", "path/fort.8")),
])
def test_open_fort_files_with_which(monkeypatch, which, expected_filenames):
    base_dir = "path"
    fort1_path = os.path.join(base_dir, "fort.1")
    fort7_path = os.path.join(base_dir, "fort.7")
    xyz_path = os.path.join(base_dir, "fort.8")

    # The arguments can be mixed depending on which argument
    args_map = {
        0: (fort1_path, None, None),
        1: (None, fort7_path, None),
        2: (None, None, xyz_path),
    }

    args = args_map[which]
    # For args None values, fill with strings to suit signature (they won't be used anyway)
    args = tuple(a if a is not None else "dummy" for a in args)

    f1, f7, x = od._open_fort_files(*args, silent=True, which=which)

    # All underlying file names should be as expected
    assert (f1.file_name.endswith("fort.1") or f1.file_name == expected_filenames[0])
    assert (f7.file_name.endswith("fort.7") or f7.file_name == expected_filenames[1])
    assert (x.file_name.endswith("fort.8") or x.file_name == expected_filenames[2])


def test_check_npy_dump_exists_detection(tmp_path):
    # Create directory and simulate class_data/x.npy inside
    dir_path = tmp_path / "data_dir"
    dir_path.mkdir()
    class_data_dir = dir_path / "class_data"
    class_data_dir.mkdir()
    npy_file = class_data_dir / "x.npy"
    np.save(str(npy_file), np.array([1, 2, 3]))

    # Check passing directory path
    exists, class_dir = od._check_npy_dump_exists(str(dir_path))
    assert exists
    assert class_dir == str(class_data_dir)

    # Check passing file path inside directory
    file_path = dir_path / "file.dat"
    file_path.write_text("dummy")
    exists2, class_dir2 = od._check_npy_dump_exists(str(file_path))
    assert exists2
    assert class_dir2 == str(class_data_dir)

    # Check raises FileNotFoundError if file/dir doesn't exist
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists("/nonexistent/path/to/file")


@pytest.mark.parametrize(
    "inputs, expect_type",
    [
        # All Fort1, Fort7, Xyz objects
        (
            (create_dummy_fort1("f1"), create_dummy_fort7("f7"), create_dummy_xyz("x")),
            'obj',
        ),
        # All strings (file paths)
        (
            ("f1path", "f7path", "xpath"),
            'str',
        ),
        # First arg string triggers which=0
        (
            ("f1path", create_dummy_fort7("f7"), create_dummy_xyz("x")),
            'which0',
        ),
        # Second arg string triggers which=1
        (
            (create_dummy_fort1("f1"), "f7path", create_dummy_xyz("x")),
            'which1',
        ),
        # Third arg string triggers which=2
        (
            (create_dummy_fort1("f1"), create_dummy_fort7("f7"), "xpath"),
            'which2',
        ),
    ]
)
def test_check_constructor_input_various(monkeypatch, inputs, expect_type):
    # Patch _open_fort_files to identify which was called
    called = {}

    def fake_open_fort_files(*args, silent=False, which=None):
        called['which'] = which
        return (create_dummy_fort1("f1dummy"),
                create_dummy_fort7("f7dummy"),
                create_dummy_xyz("xdummy"))

    monkeypatch.setattr(od, "_open_fort_files", fake_open_fort_files)

    if expect_type == 'obj':
        f1, f7, x = od._check_constructor_input(*inputs)
        # Should return the input objects unchanged
        assert f1 is inputs[0]
        assert f7 is inputs[1]
        assert x is inputs[2]

    elif expect_type == 'str':
        f1, f7, x = od._check_constructor_input(*inputs)
        assert called['which'] is None

    else:
        f1, f7, x = od._check_constructor_input(*inputs)
        mapping = {'which0': 0, 'which1': 1, 'which2': 2}
        assert called['which'] == mapping[expect_type]

    # Check that invalid input raises ValueError
    if expect_type == 'obj':
        with pytest.raises(ValueError):
            od._check_constructor_input(object(), object(), object())


def test_check_constructor_input_single_arg_errors(tmp_path):
    invalid_input = object()
    with pytest.raises(TypeError):
        od._check_constructor_input(invalid_input)

    # String path not exists -> FileNotFoundError
    fake_path = str(tmp_path / "nonexistent_file.xyz")
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(fake_path)

    # Valid file extension handling
    base_dir = tmp_path / "dir_ext"
    base_dir.mkdir()
    for ext in ['1', '7', '8', 'xyz']:
        file_path = base_dir / f"file.{ext}"
        file_path.write_text("content")
        # Create dummy fort.1, fort.7, fort.8 for directory to exist
        for fname in ['fort.1', 'fort.7', 'fort.8']:
            (base_dir / fname).write_text("dummy")

        # Patch _open_fort_files to verify args
        called_args = {}
        def fake_open(f1, f7, x, silent):
            called_args['f1'], called_args['f7'], called_args['x'] = f1, f7, x
            return create_dummy_fort1(f1), create_dummy_fort7(f7), create_dummy_xyz(x)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(od, "_open_fort_files", fake_open)

        od._check_constructor_input(str(file_path))
        # Called paths must be consistent with extension rules
        if ext == '1':
            assert called_args['f1'].endswith('.1')
            assert called_args['f7'].endswith('fort.7')
            assert called_args['x'].endswith('fort.8')
        elif ext == '7':
            assert called_args['f7'].endswith('.7')
            assert called_args['f1'].endswith('fort.1')
            assert called_args['x'].endswith('fort.8')
        else:  # 8 and xyz
            assert called_args['x'].endswith(f".{ext}")
            assert called_args['f1'].endswith('fort.1')
            assert called_args['f7'].endswith('fort.7')

        monkeypatch.undo()


def test_occamdata_init_loads_and_saves(tmp_path, monkeypatch):
    # Setup dummy class_data directory with npy and class.json file for load_from_npy test
    class_data_dir = tmp_path / "class_data"
    class_data_dir.mkdir()
    arr = np.arange(3)
    np.save(str(class_data_dir / "some_array.npy"), arr)
    class_attrs = {"fort1_file_name": "dummy1", "fort7_file_name": "dummy7",
                   "xyz_file_name": "dummy8", "some_array": arr.tolist()}
    with open(class_data_dir / "class.json", "w") as f:
        json.dump(class_attrs, f)

    # Patch _check_npy_dump_exists to True for this test
    monkeypatch.setattr(od, "_check_npy_dump_exists", lambda f: (True, str(tmp_path / "class_data")))

    # Patch load to confirm called
    load_called = {}
    def fake_load(self, class_path, silent=False):
        load_called['called'] = True
        self.some_array = arr
    monkeypatch.setattr(od.OccamData, "load", fake_load)

    # Instantiate OccamData with single arg triggers load from npy
    od_instance = od.OccamData(str(tmp_path))
    assert load_called.get('called', False)

    # Now test with load_from_npy=False forces reading data
    monkeypatch.setattr(od, "_check_npy_dump_exists", lambda f: (False, None))

    called_open = {}
    def fake_check_constructor_input(*args, silent=False):
        called_open['called'] = True
        return create_dummy_fort1("f1"), create_dummy_fort7("f7"), create_dummy_xyz("x")

    monkeypatch.setattr(od, "_check_constructor_input", fake_check_constructor_input)
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda f1, f7, x: True)
    monkeypatch.setattr(od.OccamData, "save", lambda self, overwrite=False: True)

    od_instance2 = od.OccamData("f1path", "f7path", "xpath", load_from_npy=False)
    assert called_open['called']


def test_occamdata_save_creates_directory_and_files(tmp_path):
    # Create dummy Fort1, Fort7, Xyz objects with file_names pointing inside tmp_path
    f1 = create_dummy_fort1(str(tmp_path / "fort.1"))
    f7 = create_dummy_fort7(str(tmp_path / "fort.7"))
    x = create_dummy_xyz(str(tmp_path / "fort.8"))

    odinst = od.OccamData.__new__(od.OccamData)  # create instance w/o __init__
    odinst.fort1_file_name = f1.file_name
    odinst.fort7_file_name = f7.file_name
    odinst.xyz_file_name = x.file_name
    odinst.save_dir = "class_data"
    odinst.some_array = np.arange(5)
    odinst.some_other = "not an array"

    save_path = os.path.join(tmp_path, "class_data")
    # Save should make the directory and save arrays + class.json
    result = odinst.save(overwrite=True)
    assert result is True
    assert os.path.isdir(save_path)
    # There must be npy file for some_array
    assert os.path.isfile(os.path.join(save_path, "some_array.npy"))
    # And class.json file must exist
    assert os.path.isfile(os.path.join(save_path, "class.json"))
    # After save, array attributes should be deleted
    assert not hasattr(odinst, "some_array")


def test_occamdata_save_returns_false_if_dir_exists_and_not_overwrite(tmp_path):
    # Setup dummy instance
    odinst = od.OccamData.__new__(od.OccamData)
    odinst.fort1_file_name = str(tmp_path / "fort.1")
    odinst.save_dir = "class_data"
    # Create class_data directory
    save_dir = tmp_path / "class_data"
    save_dir.mkdir()

    odinst.some_array = np.array([1, 2])

    # Should return False since directory exists and overwrite=False default
    res = odinst.save()
    assert res is False


def test_load_arrays_sets_attributes(tmp_path):
    d = tmp_path / "class_data"
    d.mkdir()
    # Save 2 npy files
    np.save(str(d / "attr1.npy"), np.array([1, 2]))
    np.save(str(d / "attr2.npy"), np.array([3, 4]))

    # Also add some non-npy file that should be ignored
    (d / "ignore.txt").write_text("ignore me")

    odinst = od.OccamData.__new__(od.OccamData)
    # Silent load disables tqdm print, test both silent True and False
    odinst._load_arrays(str(d), silent=True)
    assert np.array_equal(odinst.attr1, np.array([1,2]))
    assert np.array_equal(odinst.attr2, np.array([3,4]))

    odinst2 = od.OccamData.__new__(od.OccamData)
    odinst2._load_arrays(str(d), silent=False)
    assert np.array_equal(odinst2.attr1, np.array([1,2]))
    assert np.array_equal(odinst2.attr2, np.array([3,4]))


def test_save_and_load_class_writes_and_reads_json(tmp_path):
    odinst = od.OccamData.__new__(od.OccamData)
    odinst.fort1_file_name = str(tmp_path / "fort.1")
    odinst.save_dir = "class_data"
    odinst.some_attr = "data"
    odinst.num = 42

    # Save class writes class.json correctly
    odinst.save_path = str(tmp_path / "class_data")
    os.makedirs(odinst.save_path, exist_ok=True)
    odinst._save_class()

    json_file = os.path.join(odinst.save_path, "class.json")
    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["some_attr"] == "data"
    assert data["num"] == 42

    # Now load class sets attributes correctly
    new_inst = od.OccamData.__new__(od.OccamData)
    new_inst._load_class(odinst.save_path)
    assert new_inst.some_attr == "data"
    assert new_inst.num == 42


def test_delete_array_attributes_removes_only_ndarrays():
    odinst = od.OccamData.__new__(od.OccamData)
    odinst.a = np.array([1, 2, 3])
    odinst.b = "not an array"
    odinst.c = np.array([4, 5])
    odinst._delete_array_attributes()
    assert not hasattr(odinst, "a")
    assert hasattr(odinst, "b")
    assert not hasattr(odinst, "c")
