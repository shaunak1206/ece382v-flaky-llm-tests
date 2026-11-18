import json
import os

import numpy as np
import pytest

import occamtools.occam_data as od


class FakeFortBase:
    def __init__(self, file_name, *, n_time_steps_=3, value=1.0, label="same",
                 arr=None):
        self.file_name = file_name
        self.num_lines = 10
        self.n_time_steps_ = n_time_steps_
        self.file_contents = "contents"
        self.comment_format_known = True
        self.value = value
        self.label = label
        self.arr = np.array([1, 2, 3]) if arr is None else np.array(arr)
        self.text = "hello"

    def read_file(self, silent=False):
        self.read_called = True
        self.read_silent = silent


class FakeFort1(FakeFortBase):
    pass


class FakeFort7(FakeFortBase):
    pass


class FakeXyz(FakeFortBase):
    pass


@pytest.fixture
def patch_fake_classes(monkeypatch):
    monkeypatch.setattr(od, "Fort1", FakeFort1)
    monkeypatch.setattr(od, "Fort7", FakeFort7)
    monkeypatch.setattr(od, "Xyz", FakeXyz)
    return monkeypatch


def make_triplet(tmp_path, *, value=1.0, label="same", arr=None):
    f1 = FakeFort1(str(tmp_path / "fort.1"), value=value, label=label, arr=arr)
    f7 = FakeFort7(str(tmp_path / "fort.7"), value=value, label=label, arr=arr)
    x = FakeXyz(str(tmp_path / "fort.8"), value=value, label=label, arr=arr)
    return f1, f7, x


def test_check_internal_consistency_true():
    a = FakeFort1("a", value=1.0, label="x")
    b = FakeFort7("b", value=1.0, label="x")
    assert od._check_internal_consistency(a, b) is True


def test_check_internal_consistency_warns_on_numeric_difference():
    a = FakeFort1("a", value=1.0)
    b = FakeFort7("b", value=2.0)
    with pytest.warns(UserWarning, match="internally consistent"):
        assert od._check_internal_consistency(a, b) is False


def test_check_internal_consistency_warns_on_string_difference():
    a = FakeFort1("a", label="one")
    b = FakeFort7("b", label="two")
    with pytest.warns(UserWarning, match="internally consistent"):
        assert od._check_internal_consistency(a, b) is False


def test_check_internal_consistency_all_true():
    f1 = FakeFort1("a")
    f7 = FakeFort7("b")
    x = FakeXyz("c")
    assert od._check_internal_consistency_all(f1, f7, x) is True


def test_check_internal_consistency_all_false(monkeypatch):
    f1 = FakeFort1("a", value=1.0)
    f7 = FakeFort7("b", value=2.0)
    x = FakeXyz("c", value=1.0)
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency_all(f1, f7, x) is False


def test_check_npy_dump_exists_on_directory(tmp_path):
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1, 2, 3]))
    exists, found_dir = od._check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert found_dir == str(class_dir)


def test_check_npy_dump_exists_on_file_path(tmp_path):
    data_file = tmp_path / "fort.1"
    data_file.write_text("dummy")
    class_dir = tmp_path / "class_data"
    class_dir.mkdir()
    np.save(class_dir / "x.npy", np.array([1, 2, 3]))
    exists, found_dir = od._check_npy_dump_exists(str(data_file))
    assert exists is True
    assert found_dir == str(class_dir)


def test_check_npy_dump_exists_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / "missing.1"))


def test_check_constructor_input_with_existing_objects():
    f1 = FakeFort1("a")
    f7 = FakeFort7("b")
    x = FakeXyz("c")
    out = od._check_constructor_input(f1, f7, x)
    assert out == (f1, f7, x)


def test_check_constructor_input_single_bad_type_raises_type_error():
    with pytest.raises(TypeError):
        od._check_constructor_input(123)


def test_check_constructor_input_single_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input("does_not_exist.1")


def test_check_constructor_input_open_from_extension_1(tmp_path, monkeypatch, patch_fake_classes):
    f1_path = tmp_path / "fort.1"
    f1_path.write_text("dummy")
    calls = []

    def fake_open_fort_files(fort1, fort7, xyz, silent, which=None):
        calls.append((fort1, fort7, xyz, silent, which))
        return FakeFort1(fort1), FakeFort7(fort7), FakeXyz(xyz)

    monkeypatch.setattr(od, "_open_fort_files", fake_open_fort_files)
    out = od._check_constructor_input(str(f1_path), silent=True)
    assert calls and calls[0][4] is None
    assert out[0].file_name == str(f1_path)
    assert out[1].file_name.endswith("fort.7")
    assert out[2].file_name.endswith("fort.8")


def test_occamdata_init_from_objects_saves_and_sets_attributes(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path, value=1.0, label="same")
    data = od.OccamData(f1, f7, x, save_to_npy=False)
    assert data.consistent is True
    assert data.fort1_file_name == f1.file_name
    assert data.fort7_file_name == f7.file_name
    assert data.xyz_file_name == x.file_name
    assert np.array_equal(data.arr, np.array([1, 2, 3]))
    assert data.value == 1.0
    assert data.label == "same"


def test_occamdata_save_creates_class_data_and_roundtrips_arrays(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path)
    data = od.OccamData(f1, f7, x, save_to_npy=False)

    saved = data.save()
    assert saved is True

    class_dir = tmp_path / "class_data"
    assert class_dir.is_dir()
    assert (class_dir / "x.npy").exists()
    assert (class_dir / "arr.npy").exists()
    assert (class_dir / "class.json").exists()

    with open(class_dir / "class.json", "r") as fh:
        class_state = json.load(fh)
    assert "arr" not in class_state

    assert isinstance(data.arr, np.ndarray)
    assert np.array_equal(data.arr, np.array([1, 2, 3]))


def test_occamdata_save_returns_false_when_not_overwriting(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path)
    data = od.OccamData(f1, f7, x, save_to_npy=False)
    assert data.save() is True
    assert data.save() is False


def test_occamdata_save_overwrite_true_succeeds(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path)
    data = od.OccamData(f1, f7, x, save_to_npy=False)
    assert data.save() is True
    assert data.save(overwrite=True) is True


def test_occamdata_init_from_directory_loads_cached_npy(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path)
    data = od.OccamData(f1, f7, x, save_to_npy=False)
    assert data.save() is True

    loaded = od.OccamData(str(tmp_path), load_from_npy=True, save_to_npy=False)
    assert loaded.fort1_file_name == f1.file_name
    assert loaded.fort7_file_name == f7.file_name
    assert loaded.xyz_file_name == x.file_name
    assert np.array_equal(loaded.arr, np.array([1, 2, 3]))
    assert loaded.consistent is True


def test_occamdata_init_from_directory_without_npy_rebuilds_from_files(tmp_path, patch_fake_classes):
    f1, f7, x = make_triplet(tmp_path)
    data = od.OccamData(f1, f7, x, save_to_npy=False)
    assert data.save() is True
    class_dir = tmp_path / "class_data"
    for item in class_dir.iterdir():
        item.unlink()
    class_dir.rmdir()

    rebuilt = od.OccamData(str(tmp_path), load_from_npy=True, save_to_npy=False)
    assert rebuilt.fort1_file_name == f1.file_name
    assert rebuilt.fort7_file_name == f7.file_name
    assert rebuilt.xyz_file_name == x.file_name
    assert np.array_equal(rebuilt.arr, np.array([1, 2, 3]))
