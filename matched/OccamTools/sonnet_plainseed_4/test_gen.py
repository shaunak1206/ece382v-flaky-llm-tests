import json
import os
import types
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
)

READS = []


class _FakeBase:
    def __init__(self, file_name):
        self.file_name = file_name
        self.num_lines = 0

    def read_file(self, silent=False):
        READS.append((type(self).__name__, self.file_name))
        self._fill()


class FakeFort1(_FakeBase):
    def _fill(self):
        self.n_atoms = 3.0
        self.name = 'water'
        self.temperature = 300.0


class FakeFort7(_FakeBase):
    def _fill(self):
        self.n_atoms = 3.0
        self.name = 'water'
        self.energies = np.array([1.0, 2.0, 3.0])


class FakeXyz(_FakeBase):
    def _fill(self):
        self.n_atoms = 3.0
        self.x = np.arange(6.0).reshape(2, 3)


@pytest.fixture(autouse=True)
def patch_classes(monkeypatch):
    READS.clear()
    monkeypatch.setattr(occam_data, 'Fort1', FakeFort1)
    monkeypatch.setattr(occam_data, 'Fort7', FakeFort7)
    monkeypatch.setattr(occam_data, 'Xyz', FakeXyz)


@pytest.fixture
def run_dir(tmp_path):
    for name in ('fort.1', 'fort.7', 'fort.8'):
        (tmp_path / name).write_text('')
    return tmp_path


def _ns(file_name, **kwargs):
    return types.SimpleNamespace(file_name=file_name, **kwargs)


def test_consistency_equal_values_true():
    a = _ns('a', val=1.0, name='abc')
    b = _ns('b', val=1.0, name='abc')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _check_internal_consistency(a, b) is True


def test_consistency_float_mismatch_warns():
    a = _ns('a', val=1.0)
    b = _ns('b', val=2.0)
    with pytest.warns(UserWarning, match='not internally consistent'):
        assert _check_internal_consistency(a, b) is False


def test_consistency_string_mismatch_warns():
    a = _ns('a', name='abc')
    b = _ns('b', name='xyz')
    with pytest.warns(UserWarning):
        assert _check_internal_consistency(a, b) is False


def test_consistency_ignores_file_name_and_num_lines():
    a = _ns('a_file', num_lines=10, val=1.0)
    b = _ns('b_file', num_lines=20, val=1.0)
    assert _check_internal_consistency(a, b) is True


def test_consistency_ignores_attributes_missing_in_other():
    a = _ns('a', only_a=1.0, shared=2.0)
    b = _ns('b', only_b=5.0, shared=2.0)
    assert _check_internal_consistency(a, b) is True


def test_consistency_all_true_and_false():
    a = _ns('a', val=1.0)
    b = _ns('b', val=1.0)
    c = _ns('c', val=1.0)
    assert _check_internal_consistency_all(a, b, c) is True
    c2 = _ns('c', val=9.0)
    with pytest.warns(UserWarning):
        assert _check_internal_consistency_all(a, b, c2) is False


def test_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / 'nope'))


def test_npy_dump_directory_without_class_data(tmp_path):
    exists, class_dir = _check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert class_dir == os.path.join(str(tmp_path), 'class_data')


def test_npy_dump_detects_x_npy_from_file_path(run_dir):
    class_dir = run_dir / 'class_data'
    class_dir.mkdir()
    exists, path = _check_npy_dump_exists(str(run_dir / 'fort.1'))
    assert exists is False
    np.save(str(class_dir / 'x.npy'), np.zeros(2))
    exists, path = _check_npy_dump_exists(str(run_dir / 'fort.1'))
    assert exists is True
    assert path == str(class_dir)


def test_constructor_input_single_non_string_raises():
    with pytest.raises(TypeError):
        _check_constructor_input(42)


def test_constructor_input_single_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_constructor_input(str(tmp_path / 'fort.1'))


def test_constructor_input_unrecognized_three_args_raises():
    with pytest.raises(ValueError):
        _check_constructor_input(1, 2, 3)


def test_constructor_input_objects_returned_unchanged():
    f1, f7, x = FakeFort1('a'), FakeFort7('b'), FakeXyz('c')
    r1, r7, rx = _check_constructor_input(f1, f7, x)
    assert r1 is f1 and r7 is f7 and rx is x
    assert READS == []


def test_constructor_input_single_path_infers_siblings(run_dir):
    f1, f7, x = _check_constructor_input(str(run_dir / 'fort.7'), silent=True)
    assert f1.file_name == os.path.join(str(run_dir), 'fort.1')
    assert f7.file_name == str(run_dir / 'fort.7')
    assert x.file_name == os.path.join(str(run_dir), 'fort.8')
    assert len(READS) == 3


def test_constructor_input_single_directory(run_dir):
    f1, f7, x = _check_constructor_input(str(run_dir), silent=True)
    assert f1.file_name == os.path.join(str(run_dir), 'fort.1')
    assert f7.file_name == os.path.join(str(run_dir), 'fort.7')
    assert x.file_name == os.path.join(str(run_dir), 'fort.8')


def test_constructor_input_three_strings(run_dir):
    f1, f7, x = _check_constructor_input(
        str(run_dir / 'fort.1'), str(run_dir / 'fort.7'),
        str(run_dir / 'fort.8'), silent=True)
    assert isinstance(f1, FakeFort1)
    assert isinstance(f7, FakeFort7)
    assert isinstance(x, FakeXyz)
    assert f1.n_atoms == 3.0


def test_constructor_input_mixed_string_and_objects(run_dir):
    f7_obj = FakeFort7('whatever7')
    x_obj = FakeXyz('whateverx')
    f1, f7, x = _check_constructor_input(str(run_dir / 'fort.1'), f7_obj,
                                         x_obj, silent=True)
    assert f1.file_name == str(run_dir / 'fort.1')
    assert f7.file_name == os.path.join(str(run_dir), 'fort.7')
    assert x.file_name == os.path.join(str(run_dir), 'fort.8')


def test_occamdata_copies_attributes_and_saves(run_dir):
    data = OccamData(str(run_dir / 'fort.1'), silent=True)
    assert data.consistent is True
    assert data.temperature == 300.0
    assert data.name == 'water'
    assert data.fort1_file_name == str(run_dir / 'fort.1')
    assert data.fort7_file_name == str(run_dir / 'fort.7')
    assert data.xyz_file_name == str(run_dir / 'fort.8')
    assert not hasattr(data, 'num_lines')
    class_dir = run_dir / 'class_data'
    assert (class_dir / 'x.npy').exists()
    assert (class_dir / 'energies.npy').exists()
    assert (class_dir / 'class.json').exists()
    np.testing.assert_array_equal(data.x, np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(data.energies, [1.0, 2.0, 3.0])
    with open(str(class_dir / 'class.json')) as f:
        saved = json.load(f)
    assert saved['temperature'] == 300.0
    assert 'x' not in saved


def test_occamdata_save_to_npy_false_creates_no_dir(run_dir):
    data = OccamData(str(run_dir / 'fort.1'), save_to_npy=False, silent=True)
    assert not (run_dir / 'class_data').exists()
    assert isinstance(data.x, np.ndarray)


def test_occamdata_loads_from_npy_without_reading(run_dir):
    OccamData(str(run_dir / 'fort.1'), silent=True)
    READS.clear()
    data = OccamData(str(run_dir / 'fort.7'), silent=True)
    assert READS == []
    assert data.temperature == 300.0
    assert data.consistent is True
    np.testing.assert_array_equal(data.energies, [1.0, 2.0, 3.0])


def test_occamdata_load_from_npy_false_rereads(run_dir):
    OccamData(str(run_dir / 'fort.1'), silent=True)
    READS.clear()
    OccamData(str(run_dir / 'fort.1'), load_from_npy=False, save_to_npy=False,
              silent=True)
    assert len(READS) == 3


def test_occamdata_inconsistent_objects_flagged(tmp_path):
    f1 = FakeFort1(str(tmp_path / 'fort.1'))
    f7 = FakeFort7(str(tmp_path / 'fort.7'))
    x = FakeXyz(str(tmp_path / 'fort.8'))
    for f in (f1, f7, x):
        f.read_file()
    f7.n_atoms = 4.0
    with pytest.warns(UserWarning):
        data = OccamData(f1, f7, x, save_to_npy=False)
    assert data.consistent is False


def test_save_returns_false_if_exists_unless_overwrite(run_dir):
    data = OccamData(str(run_dir / 'fort.1'), silent=True)
    assert data.save() is False
    assert data.save(overwrite=True) is True
    assert isinstance(data.x, np.ndarray)
