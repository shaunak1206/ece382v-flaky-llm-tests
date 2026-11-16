import json
import os
import types
import warnings

import numpy as np
import pytest

import occamtools.occam_data as od
from occamtools.occam_data import OccamData

READS = []


class _FakeFile:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_file(self, silent=False):
        READS.append(type(self).__name__)
        self.num_lines = 10
        self.file_contents = 'contents'


class FakeFort1(_FakeFile):
    def read_file(self, silent=False):
        super().read_file(silent=silent)
        self.n_atoms = 3
        self.title = 'water'
        self.temp = 300.0


class FakeFort7(_FakeFile):
    def read_file(self, silent=False):
        super().read_file(silent=silent)
        self.n_atoms = 3
        self.energy = np.array([1.0, 2.0])


class FakeXyz(_FakeFile):
    def read_file(self, silent=False):
        super().read_file(silent=silent)
        self.n_atoms = 3
        self.x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        self.comment_format_known = True


def _fake_are_floats(*args):
    return all(isinstance(a, (int, float)) and not isinstance(a, bool)
               for a in args)


@pytest.fixture(autouse=True)
def patch_classes(monkeypatch):
    READS.clear()
    monkeypatch.setattr(od, 'Fort1', FakeFort1)
    monkeypatch.setattr(od, 'Fort7', FakeFort7)
    monkeypatch.setattr(od, 'Xyz', FakeXyz)
    monkeypatch.setattr(od, '_are_floats', _fake_are_floats)


@pytest.fixture
def run_dir(tmp_path):
    for name in ('fort.1', 'fort.7', 'fort.8'):
        (tmp_path / name).write_text('')
    return tmp_path


def _objects(run_dir):
    f1 = FakeFort1(str(run_dir / 'fort.1'))
    f7 = FakeFort7(str(run_dir / 'fort.7'))
    x = FakeXyz(str(run_dir / 'fort.8'))
    for f in (f1, f7, x):
        f.read_file()
    return f1, f7, x


def _ns(file_name, **kw):
    return types.SimpleNamespace(file_name=file_name, **kw)


def test_consistency_identical_values():
    a = _ns('a', val=1.0, name='x')
    b = _ns('b', val=1.0, name='x')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_numeric_mismatch_warns():
    a = _ns('a', val=1.0)
    b = _ns('b', val=2.0)
    with pytest.warns(UserWarning, match='not internally consistent'):
        assert od._check_internal_consistency(a, b) is False


def test_consistency_string_mismatch_warns():
    a = _ns('a', name='foo')
    b = _ns('b', name='bar')
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency(a, b) is False


def test_consistency_ignores_file_name_and_num_lines():
    a = _ns('a', num_lines=1)
    b = _ns('b', num_lines=99)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_all():
    a = _ns('a', v=1.0)
    b = _ns('b', v=1.0)
    c = _ns('c', v=1.0)
    assert od._check_internal_consistency_all(a, b, c) is True
    c.v = 5.0
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency_all(a, b, c) is False


def test_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / 'nope'))


def test_npy_dump_exists_detection(run_dir):
    exists, class_dir = od._check_npy_dump_exists(str(run_dir))
    assert exists is False
    assert class_dir == os.path.join(str(run_dir), 'class_data')
    os.mkdir(class_dir)
    np.save(os.path.join(class_dir, 'x.npy'), np.zeros(2))
    exists, class_dir2 = od._check_npy_dump_exists(str(run_dir / 'fort.1'))
    assert exists is True
    assert class_dir2 == class_dir


def test_constructor_input_objects_returned_unchanged(run_dir):
    f1, f7, x = _objects(run_dir)
    READS.clear()
    out = od._check_constructor_input(f1, f7, x)
    assert out == (f1, f7, x)
    assert READS == []


def test_constructor_input_three_paths(run_dir):
    out = od._check_constructor_input(str(run_dir / 'fort.1'),
                                      str(run_dir / 'fort.7'),
                                      str(run_dir / 'fort.8'), silent=True)
    assert [type(o) for o in out] == [FakeFort1, FakeFort7, FakeXyz]
    assert out[0].file_name == str(run_dir / 'fort.1')
    assert sorted(READS) == ['FakeFort1', 'FakeFort7', 'FakeXyz']


def test_constructor_input_invalid_three_args():
    with pytest.raises(ValueError):
        od._check_constructor_input(1, 2, 3)


def test_constructor_input_single_arg_errors(tmp_path):
    with pytest.raises(TypeError):
        od._check_constructor_input(5)
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(str(tmp_path / 'missing.1'))


@pytest.mark.parametrize('name', ['fort.1', 'fort.7', 'fort.8'])
def test_constructor_input_single_file_finds_siblings(run_dir, name):
    f1, f7, x = od._check_constructor_input(str(run_dir / name), silent=True)
    assert f1.file_name == str(run_dir / 'fort.1')
    assert f7.file_name == str(run_dir / 'fort.7')
    assert x.file_name == str(run_dir / 'fort.8')


def test_constructor_input_directory(run_dir):
    f1, f7, x = od._check_constructor_input(str(run_dir), silent=True)
    assert f1.file_name == str(run_dir / 'fort.1')
    assert x.file_name == str(run_dir / 'fort.8')


def test_constructor_mixed_path_and_objects(run_dir):
    f1, f7, x = _objects(run_dir)
    out = od._check_constructor_input(str(run_dir / 'fort.1'), f7, x,
                                      silent=True)
    assert out[1].file_name == str(run_dir / 'fort.7')
    assert out[2].file_name == str(run_dir / 'fort.8')


def test_occamdata_from_objects_copies_attributes(run_dir):
    f1, f7, x = _objects(run_dir)
    data = OccamData(f1, f7, x, save_to_npy=False, silent=True)
    assert data.consistent is True
    assert data.title == 'water'
    assert data.temp == 300.0
    assert data.n_atoms == 3
    assert np.array_equal(data.energy, np.array([1.0, 2.0]))
    assert data.fort1_file_name == f1.file_name
    assert data.fort7_file_name == f7.file_name
    assert data.xyz_file_name == x.file_name
    for ignored in ('file_name', 'num_lines', 'file_contents',
                    'comment_format_known'):
        assert not hasattr(data, ignored)
    assert not os.path.exists(os.path.join(str(run_dir), 'class_data'))


def test_occamdata_inconsistent_flag(run_dir):
    f1, f7, x = _objects(run_dir)
    f7.n_atoms = 4
    with pytest.warns(UserWarning):
        data = OccamData(f1, f7, x, save_to_npy=False, silent=True)
    assert data.consistent is False


def test_occamdata_saves_npy_and_json(run_dir):
    data = OccamData(str(run_dir / 'fort.1'), silent=True)
    class_dir = run_dir / 'class_data'
    assert (class_dir / 'x.npy').exists()
    assert (class_dir / 'energy.npy').exists()
    assert (class_dir / 'class.json').exists()
    with open(str(class_dir / 'class.json')) as fh:
        saved = json.load(fh)
    assert saved['title'] == 'water'
    assert 'x' not in saved
    # arrays are reloaded onto the instance after saving
    assert isinstance(data.x, np.ndarray)
    assert data.x.shape == (2, 3)


def test_save_returns_false_unless_overwrite(run_dir):
    data = OccamData(str(run_dir / 'fort.1'), silent=True)
    assert data.save() is False
    assert data.save(overwrite=True) is True
    assert isinstance(data.energy, np.ndarray)


def test_load_from_npy_skips_reading(run_dir):
    OccamData(str(run_dir / 'fort.1'), silent=True)
    READS.clear()
    data = OccamData(str(run_dir / 'fort.1'), silent=True)
    assert READS == []
    assert data.title == 'water'
    assert np.array_equal(data.energy, np.array([1.0, 2.0]))
    assert np.array_equal(data.x, np.array([[0.0, 0.0, 0.0],
                                            [1.0, 1.0, 1.0]]))


def test_load_from_npy_false_rereads_files(run_dir):
    OccamData(str(run_dir / 'fort.1'), silent=True)
    READS.clear()
    data = OccamData(str(run_dir / 'fort.1'), load_from_npy=False,
                     silent=True)
    assert sorted(READS) == ['FakeFort1', 'FakeFort7', 'FakeXyz']
    assert data.consistent is True


def test_save_to_npy_false_with_path_does_not_create_dir(run_dir):
    OccamData(str(run_dir), save_to_npy=False, silent=True)
    assert not (run_dir / 'class_data').exists()
