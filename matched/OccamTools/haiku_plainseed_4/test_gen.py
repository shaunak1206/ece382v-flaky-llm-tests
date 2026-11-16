import json
import os
import warnings

import numpy as np
import pytest

from occamtools import occam_data
from occamtools.occam_data import (
    OccamData,
    _check_constructor_input,
    _check_internal_consistency,
    _check_internal_consistency_all,
    _check_npy_dump_exists,
    _open_fort_files,
)

READ_LOG = []


class _FakeFile:
    """Stand-in for Fort1/Fort7/Xyz that avoids real file parsing."""

    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.num_lines = 0
        for key, value in attrs.items():
            setattr(self, key, value)

    def read_file(self, silent=False):
        READ_LOG.append((type(self).__name__, silent))


class FakeFort1(_FakeFile):
    pass


class FakeFort7(_FakeFile):
    pass


class FakeXyz(_FakeFile):
    pass


@pytest.fixture(autouse=True)
def fake_readers(monkeypatch):
    READ_LOG.clear()
    monkeypatch.setattr(occam_data, 'Fort1', FakeFort1)
    monkeypatch.setattr(occam_data, 'Fort7', FakeFort7)
    monkeypatch.setattr(occam_data, 'Xyz', FakeXyz)
    yield


def _paths(tmp_path):
    return (str(tmp_path / 'fort.1'),
            str(tmp_path / 'fort.7'),
            str(tmp_path / 'fort.8'))


def _objects(tmp_path, title_7='run'):
    f1n, f7n, xn = _paths(tmp_path)
    fort1 = FakeFort1(f1n, title='run', temperature=300.0)
    fort7 = FakeFort7(f7n, title=title_7, temperature=300.0)
    xyz = FakeXyz(xn, x=np.arange(6.0).reshape(2, 3))
    return fort1, fort7, xyz


# ---------- _check_internal_consistency ----------

def test_consistency_matching_attributes_no_warning(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), title='same')
    b = FakeFort7(str(tmp_path / 'b'), title='same')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert _check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_consistency_mismatched_string_warns(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), title='alpha')
    b = FakeFort7(str(tmp_path / 'b'), title='beta')
    with pytest.warns(UserWarning, match='not internally consistent'):
        result = _check_internal_consistency(a, b)
    assert result is False


def test_consistency_ignores_file_name_difference(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), title='same')
    b = FakeFort7(str(tmp_path / 'other' / 'b'), title='same')
    assert _check_internal_consistency(a, b) is True


def test_consistency_close_floats_are_consistent(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), value=1.0)
    b = FakeFort7(str(tmp_path / 'b'), value=1.0 + 1e-12)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert _check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_consistency_all_detects_single_mismatch(tmp_path):
    f1 = FakeFort1(str(tmp_path / 'fort.1'), title='a')
    f7 = FakeFort7(str(tmp_path / 'fort.7'), title='a')
    x = FakeXyz(str(tmp_path / 'fort.8'), title='b')
    with pytest.warns(UserWarning):
        assert _check_internal_consistency_all(f1, f7, x) is False


# ---------- _check_npy_dump_exists ----------

def test_check_npy_dump_dir_without_class_data(tmp_path):
    exists, class_dir = _check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert class_dir == str(tmp_path / 'class_data')


def test_check_npy_dump_dir_with_x_npy(tmp_path):
    (tmp_path / 'class_data').mkdir()
    np.save(str(tmp_path / 'class_data' / 'x.npy'), np.zeros(3))
    exists, class_dir = _check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert class_dir == str(tmp_path / 'class_data')


def test_check_npy_dump_from_file_path_uses_sibling(tmp_path):
    f = tmp_path / 'fort.1'
    f.write_text('')
    exists, class_dir = _check_npy_dump_exists(str(f))
    assert exists is False
    assert class_dir == str(tmp_path / 'class_data')


def test_check_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / 'does_not_exist'))


# ---------- _check_constructor_input ----------

def test_constructor_input_returns_objects_unchanged(tmp_path):
    f1, f7, x = _objects(tmp_path)
    result = _check_constructor_input(f1, f7, x)
    assert result[0] is f1
    assert result[1] is f7
    assert result[2] is x


def test_constructor_input_wrong_types_raises_value_error():
    with pytest.raises(ValueError):
        _check_constructor_input(1, 2, 3)


def test_constructor_single_non_string_raises_type_error():
    with pytest.raises(TypeError):
        _check_constructor_input(42)


def test_constructor_single_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_constructor_input(str(tmp_path / 'missing' / 'fort.1'))


def test_constructor_single_fort1_path_resolves_siblings(tmp_path):
    p = tmp_path / 'fort.1'
    p.write_text('')
    f1, f7, x = _check_constructor_input(str(p), silent=True)
    assert f1.file_name == str(p)
    assert f7.file_name == str(tmp_path / 'fort.7')
    assert x.file_name == str(tmp_path / 'fort.8')
    assert READ_LOG == [('FakeFort1', True), ('FakeFort7', True),
                        ('FakeXyz', True)]


def test_open_fort_files_which0_builds_sibling_paths(tmp_path):
    f1, f7, x = _open_fort_files(str(tmp_path / 'fort.1'), 'unused7',
                                 'unused8', False, which=0)
    assert f1.file_name == str(tmp_path / 'fort.1')
    assert f7.file_name == str(tmp_path / 'fort.7')
    assert x.file_name == str(tmp_path / 'fort.8')
    assert all(silent is False for _, silent in READ_LOG)


# ---------- OccamData constructor ----------

def test_init_copies_attributes_without_saving(tmp_path):
    f1, f7, x = _objects(tmp_path)
    obj = OccamData(f1, f7, x, save_to_npy=False)
    assert obj.title == 'run'
    assert obj.temperature == 300.0
    assert np.array_equal(obj.x, x.x)
    assert obj.consistent is True
    assert obj.fort1_file_name == f1.file_name
    assert obj.fort7_file_name == f7.file_name
    assert obj.xyz_file_name == x.file_name
    assert not hasattr(obj, 'file_name')
    assert not hasattr(obj, 'num_lines')
    assert not (tmp_path / 'class_data').exists()


def test_init_mismatch_sets_consistent_false(tmp_path):
    f1, f7, x = _objects(tmp_path, title_7='different')
    with pytest.warns(UserWarning):
        obj = OccamData(f1, f7, x, save_to_npy=False)
    assert obj.consistent is False


# ---------- save / load ----------

def test_save_writes_arrays_and_json(tmp_path):
    f1, f7, x = _objects(tmp_path)
    obj = OccamData(f1, f7, x)  # save_to_npy=True by default
    class_dir = tmp_path / 'class_data'
    assert (class_dir / 'x.npy').exists()
    assert (class_dir / 'class.json').exists()
    with open(class_dir / 'class.json') as fh:
        data = json.load(fh)
    assert data['title'] == 'run'
    assert 'x' not in data
    assert obj.save_path == str(class_dir)
    # Arrays are reloaded into the instance after saving.
    assert np.array_equal(obj.x, x.x)


def test_save_returns_false_when_dump_exists(tmp_path):
    f1, f7, x = _objects(tmp_path)
    obj = OccamData(f1, f7, x)
    assert obj.save() is False


def test_save_with_overwrite_returns_true(tmp_path):
    f1, f7, x = _objects(tmp_path)
    obj = OccamData(f1, f7, x)
    assert obj.save(overwrite=True) is True


def test_roundtrip_load_from_directory(tmp_path, capsys):
    f1, f7, x = _objects(tmp_path)
    original = OccamData(f1, f7, x)
    loaded = OccamData(str(tmp_path))
    out = capsys.readouterr().out
    assert 'Loading data' in out
    assert np.array_equal(loaded.x, original.x)
    assert loaded.title == 'run'
    assert loaded.consistent is True
    assert loaded.fort1_file_name == f1.file_name


def test_load_arrays_ignores_non_npy_files(tmp_path):
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    arr = np.array([1.5, 2.5])
    np.save(str(class_dir / 'x.npy'), arr)
    (class_dir / 'notes.txt').write_text('ignore me')
    bare = OccamData.__new__(OccamData)
    bare._load_arrays(str(class_dir), silent=True)
    assert np.array_equal(bare.x, arr)
    assert not hasattr(bare, 'notes')


def test_save_arrays_writes_npy_and_sets_save_path(tmp_path):
    arr = np.arange(4)
    (tmp_path / 'class_data').mkdir()
    bare = OccamData.__new__(OccamData)
    bare.fort1_file_name = str(tmp_path / 'fort.1')
    bare.x = arr
    bare.title = 't'
    bare._save_arrays()
    assert bare.save_path == str(tmp_path / 'class_data')
    assert np.array_equal(
        np.load(str(tmp_path / 'class_data' / 'x.npy')), arr)
    assert hasattr(bare, 'x')


def test_delete_array_attributes_keeps_other_attributes(tmp_path):
    bare = OccamData.__new__(OccamData)
    bare.x = np.zeros(2)
    bare.title = 'keep'
    bare._delete_array_attributes()
    assert not hasattr(bare, 'x')
    assert bare.title == 'keep'


def test_class_json_roundtrip(tmp_path):
    (tmp_path / 'class_data').mkdir()
    bare = OccamData.__new__(OccamData)
    bare.fort1_file_name = str(tmp_path / 'fort.1')
    bare.title = 'json_title'
    bare.consistent = True
    bare._save_class()

    restored = OccamData.__new__(OccamData)
    restored._load_class(str(tmp_path / 'class_data'))
    assert restored.title == 'json_title'
    assert restored.consistent is True
    assert restored.fort1_file_name == str(tmp_path / 'fort.1')
