import json
import os
import warnings

import numpy as np
import pytest

from occamtools import occam_data as od
from occamtools.occam_data import OccamData


class _Stub:
    """Minimal object with a __dict__ and a file_name, used in place of parsed file objects."""
    pass


def _make_stub(file_name, **attrs):
    stub = _Stub()
    stub.file_name = file_name
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub


def _bare_instance(tmp_path):
    """Create an OccamData without running __init__, pointing at tmp_path."""
    obj = OccamData.__new__(OccamData)
    obj.fort1_file_name = str(tmp_path / 'fort.1')
    obj.fort7_file_name = str(tmp_path / 'fort.7')
    obj.xyz_file_name = str(tmp_path / 'fort.8')
    return obj


# ---------------------------------------------------------------------------
# _check_internal_consistency
# ---------------------------------------------------------------------------

def test_consistency_identical_floats_no_warning():
    a = _make_stub('a.1', temp=300.0)
    b = _make_stub('b.7', temp=300.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_consistency_numeric_mismatch_warns():
    a = _make_stub('a.1', temp=300.0)
    b = _make_stub('b.7', temp=350.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is False
    assert any(issubclass(w.category, UserWarning) for w in record)


def test_consistency_float_within_allclose_tolerance_is_consistent():
    a = _make_stub('a.1', value=1.0)
    b = _make_stub('b.7', value=1.0 + 1e-12)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_ignores_file_name_and_num_lines():
    a = _make_stub('a.1', file_name_extra=1.0, num_lines=10)
    b = _make_stub('b.7', file_name_extra=1.0, num_lines=99)
    # file_name differs on the stubs themselves; num_lines differs in dict
    a.__dict__['file_name'] = 'a.1'
    b.__dict__['file_name'] = 'b.7'
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_string_mismatch_warns():
    a = _make_stub('a.1', label='alpha')
    b = _make_stub('b.7', label='beta')
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is False


def test_consistency_string_match_is_consistent():
    a = _make_stub('a.1', label='alpha')
    b = _make_stub('b.7', label='alpha')
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_ignores_keys_missing_from_other_object():
    a = _make_stub('a.1', only_in_a=5.0)
    b = _make_stub('b.7', only_in_b=7.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_warning_message_mentions_both_files():
    a = _make_stub('first_file.1', temp=1.0)
    b = _make_stub('second_file.7', temp=2.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        od._check_internal_consistency(a, b)
    messages = [str(w.message) for w in record]
    assert any('first_file.1' in m and 'second_file.7' in m for m in messages)


# ---------------------------------------------------------------------------
# _check_internal_consistency_all
# ---------------------------------------------------------------------------

def test_consistency_all_true_when_all_pairs_consistent(monkeypatch):
    monkeypatch.setattr(od, '_check_internal_consistency', lambda a, b: True)
    assert od._check_internal_consistency_all(None, None, None) is True


def test_consistency_all_false_when_any_pair_inconsistent(monkeypatch):
    results = iter([True, False, True])
    monkeypatch.setattr(od, '_check_internal_consistency',
                        lambda a, b: next(results))
    assert od._check_internal_consistency_all(None, None, None) is False


# ---------------------------------------------------------------------------
# _check_npy_dump_exists
# ---------------------------------------------------------------------------

def test_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / 'does_not_exist'))


def test_npy_dump_directory_without_class_data(tmp_path):
    exists, class_dir = od._check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert class_dir == os.path.join(str(tmp_path), 'class_data')


def test_npy_dump_directory_with_x_npy(tmp_path):
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'x.npy'), np.arange(3.0))
    exists, returned_dir = od._check_npy_dump_exists(str(tmp_path))
    assert exists is True
    assert returned_dir == os.path.join(str(tmp_path), 'class_data')


def test_npy_dump_class_data_without_x_npy_is_not_dump(tmp_path):
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'other.npy'), np.arange(3.0))
    exists, _ = od._check_npy_dump_exists(str(tmp_path))
    assert exists is False


def test_npy_dump_uses_parent_dir_for_file_input(tmp_path):
    fort1 = tmp_path / 'fort.1'
    fort1.write_text('')
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'x.npy'), np.arange(2.0))
    exists, returned_dir = od._check_npy_dump_exists(str(fort1))
    assert exists is True
    assert returned_dir == os.path.join(str(tmp_path), 'class_data')


# ---------------------------------------------------------------------------
# _check_constructor_input
# ---------------------------------------------------------------------------

def test_constructor_input_single_non_string_raises_type_error():
    with pytest.raises(TypeError):
        od._check_constructor_input(123)


def test_constructor_input_single_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(str(tmp_path / 'missing.1'))


def test_constructor_input_three_non_objects_raises_value_error():
    with pytest.raises(ValueError):
        od._check_constructor_input(1, 2, 3)


# ---------------------------------------------------------------------------
# save / load round trip and helpers
# ---------------------------------------------------------------------------

def test_save_creates_class_data_directory(tmp_path):
    obj = _bare_instance(tmp_path)
    obj.x = np.arange(4.0)
    assert obj.save() is True
    assert (tmp_path / 'class_data').is_dir()


def test_save_round_trip_restores_arrays_and_scalars(tmp_path):
    obj = _bare_instance(tmp_path)
    arr = np.linspace(0.0, 1.0, 5)
    obj.x = arr
    obj.name = 'test'
    obj.count = 3
    obj.values = [1.0, 2.0]
    assert obj.save() is True

    loaded = OccamData.__new__(OccamData)
    loaded.load(str(tmp_path / 'class_data'), silent=True)
    assert np.array_equal(loaded.x, arr)
    assert loaded.name == 'test'
    assert loaded.count == 3
    assert loaded.values == [1.0, 2.0]


def test_save_removes_arrays_from_class_json(tmp_path):
    obj = _bare_instance(tmp_path)
    obj.x = np.arange(3.0)
    obj.name = 'kept'
    obj.save()
    with open(tmp_path / 'class_data' / 'class.json') as f:
        dumped = json.load(f)
    assert 'x' not in dumped
    assert dumped['name'] == 'kept'


def test_save_returns_false_when_dir_exists_without_overwrite(tmp_path):
    (tmp_path / 'class_data').mkdir()
    obj = _bare_instance(tmp_path)
    obj.x = np.arange(3.0)
    assert obj.save() is False
    assert not (tmp_path / 'class_data' / 'class.json').exists()


def test_save_overwrite_true_writes_when_dir_exists(tmp_path):
    (tmp_path / 'class_data').mkdir()
    obj = _bare_instance(tmp_path)
    obj.x = np.arange(3.0)
    assert obj.save(overwrite=True) is True
    assert (tmp_path / 'class_data' / 'class.json').exists()
    assert (tmp_path / 'class_data' / 'x.npy').exists()


def test_save_arrays_writes_npy_files(tmp_path):
    obj = _bare_instance(tmp_path)
    (tmp_path / 'class_data').mkdir()
    obj.save_path = str(tmp_path / 'class_data')
    obj.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    obj._save_arrays()
    saved = np.load(str(tmp_path / 'class_data' / 'data.npy'))
    assert np.array_equal(saved, obj.data)


def test_delete_array_attributes_removes_only_arrays(tmp_path):
    obj = _bare_instance(tmp_path)
    obj.arr = np.zeros(3)
    obj.label = 'keep me'
    obj._delete_array_attributes()
    assert 'arr' not in obj.__dict__
    assert obj.label == 'keep me'


def test_save_class_writes_json_with_scalars(tmp_path):
    (tmp_path / 'class_data').mkdir()
    obj = _bare_instance(tmp_path)
    obj.temperature = 300.0
    obj._save_class()
    with open(tmp_path / 'class_data' / 'class.json') as f:
        dumped = json.load(f)
    assert dumped['temperature'] == 300.0
    assert dumped['fort1_file_name'] == str(tmp_path / 'fort.1')


def test_load_class_sets_attributes_from_json(tmp_path):
    with open(tmp_path / 'class.json', 'w') as f:
        json.dump({'alpha': 1.5, 'beta': 'b'}, f)
    obj = OccamData.__new__(OccamData)
    obj._load_class(str(tmp_path))
    assert obj.alpha == 1.5
    assert obj.beta == 'b'


def test_load_arrays_ignores_non_npy_files(tmp_path):
    np.save(str(tmp_path / 'a.npy'), np.arange(2.0))
    (tmp_path / 'notes.txt').write_text('not an array')
    obj = OccamData.__new__(OccamData)
    obj._load_arrays(str(tmp_path), silent=True)
    assert np.array_equal(obj.a, np.arange(2.0))
    assert not hasattr(obj, 'notes')


def test_load_not_silent_prints_message(tmp_path, capsys):
    np.save(str(tmp_path / 'a.npy'), np.arange(2.0))
    obj = OccamData.__new__(OccamData)
    obj._load_arrays(str(tmp_path), silent=False)
    captured = capsys.readouterr()
    assert 'Loading data from .npy files' in captured.out
    assert np.array_equal(obj.a, np.arange(2.0))
