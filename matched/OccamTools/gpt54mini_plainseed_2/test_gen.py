import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import occamtools.occam_data as od


class FakeFile:
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.read_calls = []
        for k, v in attrs.items():
            setattr(self, k, v)

    def read_file(self, silent=False):
        self.read_calls.append(silent)
        return self


def make_fake_triplet(root, *, consistent=True):
    common = {
        "shared_str": "same",
        "shared_num": 1.0,
        "shared_arr": np.array([1, 2, 3]),
        "file_name": "ignored",
        "num_lines": 10,
    }
    fort1 = FakeFile(os.path.join(root, "fort.1"), **common)
    fort7 = FakeFile(os.path.join(root, "fort.7"), **common)
    xyz = FakeFile(os.path.join(root, "fort.8"), **common)
    if not consistent:
        fort7.shared_str = "different"
    return fort1, fort7, xyz


def test_check_internal_consistency_true_for_matching_numbers_and_strings():
    a = FakeFile("a", x=1.0, y="hello", file_name="a")
    b = FakeFile("b", x=1.0, y="hello", file_name="b")
    assert od._check_internal_consistency(a, b) is True


def test_check_internal_consistency_warns_and_returns_false_for_mismatch():
    a = FakeFile("a", x=1.0, y="hello", file_name="a")
    b = FakeFile("b", x=2.0, y="hello", file_name="b")
    with pytest.warns(UserWarning, match="Data is not internally consistent"):
        assert od._check_internal_consistency(a, b) is False


def test_check_internal_consistency_ignores_file_name_and_num_lines():
    a = FakeFile("a", file_name="one", num_lines=1)
    b = FakeFile("b", file_name="two", num_lines=999)
    assert od._check_internal_consistency(a, b) is True


def test_check_internal_consistency_all_true_for_consistent_triplet():
    f1, f7, x = make_fake_triplet("/tmp")
    assert od._check_internal_consistency_all(f1, f7, x) is True


def test_check_internal_consistency_all_false_when_one_pair_differs():
    f1, f7, x = make_fake_triplet("/tmp", consistent=False)
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency_all(f1, f7, x) is False


def test_open_fort_files_with_paths_calls_constructors_and_read_file(monkeypatch):
    created = {"f1": [], "f7": [], "x": []}

    class DummyFort1(FakeFile):
        def __init__(self, path):
            created["f1"].append(path)
            super().__init__(path)

    class DummyFort7(FakeFile):
        def __init__(self, path):
            created["f7"].append(path)
            super().__init__(path)

    class DummyXyz(FakeFile):
        def __init__(self, path):
            created["x"].append(path)
            super().__init__(path)

    monkeypatch.setattr(od, "Fort1", DummyFort1)
    monkeypatch.setattr(od, "Fort7", DummyFort7)
    monkeypatch.setattr(od, "Xyz", DummyXyz)

    f1, f7, x = od._open_fort_files("a/fort.1", "a/fort.7", "a/fort.8", silent=True)
    assert created["f1"] == ["a/fort.1"]
    assert created["f7"] == ["a/fort.7"]
    assert created["x"] == ["a/fort.8"]
    assert f1.read_calls == [True]
    assert f7.read_calls == [True]
    assert x.read_calls == [True]


@pytest.mark.parametrize(
    "which, input_path, expected",
    [
        (0, "/tmp/run/fort.1", ("/tmp/run/fort.1", "/tmp/run/fort.7", "/tmp/run/fort.8")),
        (1, "/tmp/run/fort.7", ("/tmp/run/fort.1", "/tmp/run/fort.7", "/tmp/run/fort.8")),
        (2, "/tmp/run/fort.8", ("/tmp/run/fort.1", "/tmp/run/fort.7", "/tmp/run/fort.8")),
    ],
)
def test_open_fort_files_with_which_builds_neighbor_paths(monkeypatch, which, input_path, expected):
    seen = []

    class DummyFort1(FakeFile):
        def __init__(self, path):
            seen.append(("f1", path))
            super().__init__(path)

    class DummyFort7(FakeFile):
        def __init__(self, path):
            seen.append(("f7", path))
            super().__init__(path)

    class DummyXyz(FakeFile):
        def __init__(self, path):
            seen.append(("x", path))
            super().__init__(path)

    monkeypatch.setattr(od, "Fort1", DummyFort1)
    monkeypatch.setattr(od, "Fort7", DummyFort7)
    monkeypatch.setattr(od, "Xyz", DummyXyz)

    od._open_fort_files(input_path, input_path, input_path, silent=False, which=which)
    assert seen == [("f1", expected[0]), ("f7", expected[1]), ("x", expected[2])]


def test_check_npy_dump_exists_on_missing_file_raises(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(missing))


def test_check_npy_dump_exists_detects_dump_in_directory(tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1, 2]))
    exists, path = od._check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert path == str(class_dir)


def test_check_constructor_input_with_three_paths(monkeypatch):
    called = {}

    def fake_open(fort1, fort7, xyz, silent, which=None):
        called["args"] = (fort1, fort7, xyz, silent, which)
        return ("f1", "f7", "x")

    monkeypatch.setattr(od, "_open_fort_files", fake_open)
    assert od._check_constructor_input("a", "b", "c", silent=True) == ("f1", "f7", "x")
    assert called["args"] == ("a", "b", "c", True, None)


def test_check_constructor_input_with_mixed_three_inputs_chooses_correct_branch(monkeypatch):
    calls = []

    def fake_open(fort1, fort7, xyz, silent, which=None):
        calls.append(which)
        return ("f1", "f7", "x")

    monkeypatch.setattr(od, "_open_fort_files", fake_open)
    monkeypatch.setattr(od, "Fort1", FakeFile)
    monkeypatch.setattr(od, "Fort7", FakeFile)
    monkeypatch.setattr(od, "Xyz", FakeFile)

    od._check_constructor_input(FakeFile("f1"), "fort.7", FakeFile("x"), silent=False)
    od._check_constructor_input(FakeFile("f1"), FakeFile("f7"), "x", silent=False)
    assert calls == [1, 2]


def test_check_constructor_input_with_three_nonpaths_raises():
    with pytest.raises(ValueError, match="constructor input not recognized"):
        od._check_constructor_input(1, 2, 3)


def test_check_constructor_input_single_nonstring_raises_typeerror():
    with pytest.raises(TypeError, match="must be a file path"):
        od._check_constructor_input(123)


def test_check_constructor_input_single_missing_path_raises_filenotfound(tmp_path):
    missing = tmp_path / "nope.1"
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(str(missing))


def test_occamdata_init_from_triplet_saves_and_loads_npy(tmp_path, monkeypatch):
    f1, f7, x = make_fake_triplet(str(tmp_path))
    monkeypatch.setattr(od, "_check_constructor_input", lambda *args, **kwargs: (f1, f7, x))
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda *args, **kwargs: True)

    data = od.OccamData("a", "b", "c", save_to_npy=True, load_from_npy=False)
    assert data.consistent is True
    assert isinstance(data.shared_arr, np.ndarray)
    assert os.path.isdir(tmp_path / "class_data")
    assert (tmp_path / "class_data" / "x.npy").exists()
    assert (tmp_path / "class_data" / "class.json").exists()


def test_occamdata_save_returns_false_when_directory_exists_and_no_overwrite(tmp_path, monkeypatch):
    f1, f7, x = make_fake_triplet(str(tmp_path))
    monkeypatch.setattr(od, "_check_constructor_input", lambda *args, **kwargs: (f1, f7, x))
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda *args, **kwargs: True)
    data = od.OccamData("a", "b", "c", save_to_npy=False)

    save_dir = tmp_path / "class_data"
    save_dir.mkdir()
    assert data.save(overwrite=False) is False


def test_occamdata_save_overwrite_writes_and_reload_arrays(tmp_path, monkeypatch):
    f1, f7, x = make_fake_triplet(str(tmp_path))
    monkeypatch.setattr(od, "_check_constructor_input", lambda *args, **kwargs: (f1, f7, x))
    monkeypatch.setattr(od, "_check_internal_consistency_all", lambda *args, **kwargs: True)

    data = od.OccamData("a", "b", "c", save_to_npy=False)
    assert data.save(overwrite=True) is True
    assert isinstance(data.shared_arr, np.ndarray)
    assert data.shared_arr.tolist() == [1, 2, 3]


def test_occamdata_load_from_existing_npy_dump(tmp_path, monkeypatch):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "shared_arr.npy", np.array([9, 8, 7]))
    with open(class_dir / "class.json", "w") as f:
        json.dump({"some_flag": True, "fort1_file_name": str(tmp_path / "fort.1")}, f)

    called = {"constructor": 0}

    def fake_constructor(*args, **kwargs):
        called["constructor"] += 1
        raise AssertionError("constructor should not be called when loading from npy")

    monkeypatch.setattr(od, "_check_npy_dump_exists", lambda path: (True, str(class_dir)))
    monkeypatch.setattr(od, "_check_constructor_input", fake_constructor)

    data = od.OccamData(str(tmp_path), load_from_npy=True, save_to_npy=False)
    assert called["constructor"] == 0
    assert data.some_flag is True
    assert data.shared_arr.tolist() == [9, 8, 7]
