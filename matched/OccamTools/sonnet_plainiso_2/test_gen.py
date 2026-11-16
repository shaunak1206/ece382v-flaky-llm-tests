import os
import types
import warnings

import numpy as np
import pytest

import occamtools.occam_data as od
from occamtools.occam_data import OccamData


class _FakeFile:
    DATA = {}
    reads = []

    def __init__(self, file_name):
        self.file_name = file_name
        self.num_lines = 0

    def read_file(self, silent=False):
        _FakeFile.reads.append(self.file_name)
        for k, v in self.DATA.items():
            self.__dict__[k] = v.copy() if isinstance(v, np.ndarray) else v


class FakeFort1(_FakeFile):
    DATA = {'title': 'water', 'n_atoms': 3}


class FakeFort7(_FakeFile):
    DATA = {'energy': 1.5, 'n_atoms': 3}


class FakeXyz(_FakeFile):
    DATA = {'x': np.arange(6, dtype=float).reshape(2, 3), 'n_atoms': 3,
            'comment': 'c'}


def _simple_are_floats(a, b):
    def ok(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return ok(a) and ok(b)


@pytest.fixture(autouse=True)
def patch_classes(monkeypatch):
    monkeypatch.setattr(od, 'Fort1', FakeFort1)
    monkeypatch.setattr(od, 'Fort7', FakeFort7)
    monkeypatch.setattr(od, 'Xyz', FakeXyz)
    monkeypatch.setattr(od, '_are_floats', _simple_are_floats)
    _FakeFile.reads = []


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / 'run'
    d.mkdir()
    for name in ('fort.1', 'fort.7', 'fort.8'):
        (d / name).write_text('')
    return d


def _ns(file_name, **kw):
    return types.SimpleNamespace(file_name=file_name, **kw)


def test_consistency_true_when_equal():
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
    a = _ns('a', name='x')
    b = _ns('b', name='y')
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency(a, b) is False


def test_consistency_ignores_file_name_and_num_lines():
    a = _ns('a', num_lines=1, only_a=5.0)
    b = _ns('b', num_lines=99)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert od._check_internal_consistency(a, b) is True


def test_consistency_all():
    f1 = _ns('1', v=1.0)
    f7 = _ns('7', v=1.0)
    x = _ns('x', v=1.0)
    assert od._check_internal_consistency_all(f1, f7, x) is True
    x.v = 3.0
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency_all(f1, f7, x) is False


def test_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / 'nope'))


def test_npy_dump_exists_logic(tmp_path):
    exists, class_dir = od._check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert class_dir == os.path.join(str(tmp_path), 'class_data')
    os.mkdir(class_dir)
    assert od._check_npy_dump_exists(str(tmp_path))[0] is False
    np.save(os.path.join(class_dir, 'x.npy'), np.zeros(2))
    assert od._check_npy_dump_exists(str(tmp_path))[0] is True
    # file path inside the directory also resolves to the same class dir
    f = tmp_path / 'fort.1'
    f.write_text('')
    exists, cdir = od._check_npy_dump_exists(str(f))
    assert exists is True
    assert cdir == class_dir


def test_constructor_input_single_non_string_raises():
    with pytest.raises(TypeError):
        od._check_constructor_input(42)


def test_constructor_input_single_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(str(tmp_path / 'missing.1'))


def test_constructor_input_objects_returned_unchanged():
    f1, f7, x = FakeFort1('a'), FakeFort7('b'), FakeXyz('c')
    out = od._check_constructor_input(f1, f7, x)
    assert out[0] is f1 and out[1] is f7 and out[2] is x
    assert _FakeFile.reads == []


def test_constructor_input_three_paths_reads_all(tmp_path):
    paths = [str(tmp_path / n) for n in ('a.1', 'b.7', 'c.8')]
    f1, f7, x = od._check_constructor_input(*paths)
    assert [f1.file_name, f7.file_name, x.file_name] == paths
    assert sorted(_FakeFile.reads) == sorted(paths)
    assert f1.title == 'water' and f7.energy == 1.5


def test_constructor_input_single_file_infers_siblings(run_dir):
    for name in ('fort.1', 'fort.7', 'fort.8'):
        _FakeFile.reads = []
        f1, f7, x = od._check_constructor_input(str(run_dir / name))
        assert f1.file_name == str(run_dir / 'fort.1')
        assert f7.file_name == str(run_dir / 'fort.7')
        assert x.file_name == str(run_dir / 'fort.8')


def test_constructor_input_single_directory(run_dir):
    f1, f7, x = od._check_constructor_input(str(run_dir))
    assert f1.file_name == os.path.join(str(run_dir), 'fort.1')
    assert f7.file_name == os.path.join(str(run_dir), 'fort.7')
    assert x.file_name == os.path.join(str(run_dir), 'fort.8')


def test_constructor_input_mixed_string_first(run_dir):
    f1_path = str(run_dir / 'fort.1')
    f1, f7, x = od._check_constructor_input(f1_path, FakeFort7('q'),
                                            FakeXyz('r'))
    assert f1.file_name == f1_path
    assert f7.file_name == str(run_dir / 'fort.7')
    assert x.file_name == str(run_dir / 'fort.8')


def test_occamdata_merges_attributes(run_dir):
    data = OccamData(str(run_dir), save_to_npy=False, silent=True)
    assert data.consistent is True
    assert data.title == 'water'
    assert data.energy == 1.5
    assert data.comment == 'c'
    assert np.array_equal(data.x, FakeXyz.DATA['x'])
    assert data.fort1_file_name == str(run_dir / 'fort.1')
    assert data.fort7_file_name == str(run_dir / 'fort.7')
    assert data.xyz_file_name == str(run_dir / 'fort.8')
    assert 'file_name' not in data.__dict__
    assert 'num_lines' not in data.__dict__
    assert not (run_dir / 'class_data').exists()


def test_occamdata_saves_npy_and_json(run_dir):
    data = OccamData(str(run_dir), silent=True)
    class_dir = run_dir / 'class_data'
    assert (class_dir / 'x.npy').exists()
    assert (class_dir / 'class.json').exists()
    # arrays are reloaded into the instance after saving
    assert isinstance(data.x, np.ndarray)
    assert np.array_equal(data.x, FakeXyz.DATA['x'])
    assert data.title == 'water'


def test_save_does_not_overwrite_unless_asked(run_dir):
    data = OccamData(str(run_dir), silent=True)
    assert data.save() is False
    assert data.save(overwrite=True) is True


def test_occamdata_loads_from_npy_without_reading(run_dir):
    OccamData(str(run_dir), silent=True)
    _FakeFile.reads = []
    loaded = OccamData(str(run_dir), silent=True)
    assert _FakeFile.reads == []
    assert loaded.title == 'water'
    assert loaded.energy == 1.5
    assert np.array_equal(loaded.x, FakeXyz.DATA['x'])


def test_occamdata_load_from_npy_false_rereads(run_dir):
    OccamData(str(run_dir), silent=True)
    _FakeFile.reads = []
    data = OccamData(str(run_dir), load_from_npy=False, save_to_npy=False,
                     silent=True)
    assert len(_FakeFile.reads) == 3
    assert data.consistent is True


def test_occamdata_inconsistent_data_warns(run_dir, monkeypatch):
    monkeypatch.setattr(FakeFort7, 'DATA', {'energy': 1.5, 'n_atoms': 4})
    with pytest.warns(UserWarning):
        data = OccamData(str(run_dir), save_to_npy=False, silent=True)
    assert data.consistent is False
