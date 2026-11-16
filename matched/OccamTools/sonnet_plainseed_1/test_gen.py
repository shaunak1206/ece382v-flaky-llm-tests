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
from occamtools.read_fort1 import Fort1
from occamtools.read_fort7 import Fort7
from occamtools.read_xyz import Xyz


class FakeFort1(Fort1):
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.__dict__.update(attrs)

    def read_file(self, silent=False):
        self.read_called = True


class FakeFort7(Fort7):
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.__dict__.update(attrs)

    def read_file(self, silent=False):
        self.read_called = True


class FakeXyz(Xyz):
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.__dict__.update(attrs)

    def read_file(self, silent=False):
        self.read_called = True


class Simple:
    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.__dict__.update(attrs)


def make_fakes(tmp_path):
    f1 = FakeFort1(str(tmp_path / 'fort.1'), temp=300.0, label='abc',
                   shared=1.5)
    f7 = FakeFort7(str(tmp_path / 'fort.7'), x=np.arange(5.0), shared=1.5,
                   n_atoms=3)
    xyz = FakeXyz(str(tmp_path / 'fort.8'), y=np.ones(3), num_lines=10,
                  n_time_steps_=4, file_contents='junk')
    return f1, f7, xyz


@pytest.fixture
def patched_classes(monkeypatch):
    monkeypatch.setattr(occam_data, 'Fort1', FakeFort1)
    monkeypatch.setattr(occam_data, 'Fort7', FakeFort7)
    monkeypatch.setattr(occam_data, 'Xyz', FakeXyz)


def test_consistency_equal_values_no_warning():
    a = Simple('a.1', v=1.0, name='x')
    b = Simple('b.7', v=1.0, name='x')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _check_internal_consistency(a, b) is True


def test_consistency_numeric_mismatch_warns():
    a = Simple('a.1', v=1.0)
    b = Simple('b.7', v=2.0)
    with pytest.warns(UserWarning, match='not internally consistent'):
        assert _check_internal_consistency(a, b) is False


def test_consistency_string_mismatch_warns():
    a = Simple('a.1', name='foo')
    b = Simple('b.7', name='bar')
    with pytest.warns(UserWarning):
        assert _check_internal_consistency(a, b) is False


def test_consistency_ignores_file_name_and_num_lines():
    a = Simple('a.1', num_lines=5, v=1.0)
    b = Simple('b.7', num_lines=99, v=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _check_internal_consistency(a, b) is True


def test_consistency_ignores_attributes_missing_in_other():
    a = Simple('a.1', only_a=1.0)
    b = Simple('b.7', only_b=2.0)
    assert _check_internal_consistency(a, b) is True


def test_consistency_all_true_and_false():
    a = Simple('a', v=1.0)
    b = Simple('b', v=1.0)
    c = Simple('c', v=1.0)
    assert _check_internal_consistency_all(a, b, c) is True
    d = Simple('d', v=5.0)
    with pytest.warns(UserWarning):
        assert _check_internal_consistency_all(a, b, d) is False


def test_check_npy_dump_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_npy_dump_exists(str(tmp_path / 'nope'))


def test_check_npy_dump_directory_cases(tmp_path):
    exists, class_dir = _check_npy_dump_exists(str(tmp_path))
    assert exists is False
    assert class_dir == os.path.join(str(tmp_path), 'class_data')

    os.mkdir(class_dir)
    exists, _ = _check_npy_dump_exists(str(tmp_path))
    assert exists is False

    np.save(os.path.join(class_dir, 'x.npy'), np.arange(3))
    exists, _ = _check_npy_dump_exists(str(tmp_path))
    assert exists is True


def test_check_npy_dump_with_file_path(tmp_path):
    f = tmp_path / 'fort.1'
    f.write_text('data')
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'x.npy'), np.arange(3))
    exists, path = _check_npy_dump_exists(str(f))
    assert exists is True
    assert path == os.path.join(str(tmp_path), 'class_data')


def test_constructor_input_single_non_string_raises():
    with pytest.raises(TypeError):
        _check_constructor_input(123)


def test_constructor_input_single_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _check_constructor_input(str(tmp_path / 'missing.1'))


def test_constructor_input_unrecognized_three_args_raises():
    with pytest.raises(ValueError):
        _check_constructor_input(1, 2, 3)


def test_constructor_input_returns_objects_unchanged(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    r1, r7, rx = _check_constructor_input(f1, f7, xyz)
    assert r1 is f1 and r7 is f7 and rx is xyz


def test_open_fort_files_which_variants(patched_classes, tmp_path):
    d = str(tmp_path)
    paths = {
        0: os.path.join(d, 'fort.1'),
        1: os.path.join(d, 'fort.7'),
        2: os.path.join(d, 'fort.8'),
    }
    for which, given in paths.items():
        args = [None, None, None]
        args[which] = given
        f1, f7, x = _open_fort_files(args[0], args[1], args[2], True,
                                     which=which)
        assert f1.file_name == os.path.join(d, 'fort.1')
        assert f7.file_name == os.path.join(d, 'fort.7')
        assert x.file_name == os.path.join(d, 'fort.8')
        assert f1.read_called and f7.read_called and x.read_called


def test_constructor_input_single_path_by_extension(patched_classes,
                                                    tmp_path):
    d = str(tmp_path)
    for name in ('fort.1', 'fort.7', 'fort.8'):
        (tmp_path / name).write_text('x')
        f1, f7, x = _check_constructor_input(os.path.join(d, name),
                                             silent=True)
        assert f1.file_name == os.path.join(d, 'fort.1')
        assert f7.file_name == os.path.join(d, 'fort.7')
        assert x.file_name == os.path.join(d, 'fort.8')


def test_occamdata_combines_attributes(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    od = OccamData(f1, f7, xyz, save_to_npy=False, silent=True)
    assert od.temp == 300.0
    assert od.label == 'abc'
    assert od.n_atoms == 3
    assert np.array_equal(od.x, np.arange(5.0))
    assert np.array_equal(od.y, np.ones(3))
    assert od.consistent is True
    assert od.fort1_file_name == f1.file_name
    assert od.fort7_file_name == f7.file_name
    assert od.xyz_file_name == xyz.file_name
    for ignored in ('n_time_steps_', 'file_contents', 'num_lines',
                    'file_name'):
        assert not hasattr(od, ignored)
    assert not os.path.exists(str(tmp_path / 'class_data'))


def test_occamdata_inconsistent_inputs_flagged(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    f7.shared = 99.0
    with pytest.warns(UserWarning):
        od = OccamData(f1, f7, xyz, save_to_npy=False, silent=True)
    assert od.consistent is False


def test_occamdata_save_creates_files_and_reloads_arrays(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    od = OccamData(f1, f7, xyz, silent=True)
    class_dir = tmp_path / 'class_data'
    assert class_dir.is_dir()
    assert (class_dir / 'x.npy').exists()
    assert (class_dir / 'y.npy').exists()
    assert (class_dir / 'class.json').exists()
    assert np.array_equal(od.x, np.arange(5.0))
    assert np.array_equal(od.y, np.ones(3))


def test_occamdata_save_no_overwrite(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    od = OccamData(f1, f7, xyz, silent=True)
    assert od.save() is False
    assert od.save(overwrite=True) is True


def test_occamdata_load_from_saved_directory(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    OccamData(f1, f7, xyz, silent=True)
    loaded = OccamData(str(tmp_path), silent=True)
    assert loaded.temp == 300.0
    assert loaded.label == 'abc'
    assert loaded.n_atoms == 3
    assert loaded.fort1_file_name == f1.file_name
    assert np.array_equal(loaded.x, np.arange(5.0))
    assert np.array_equal(loaded.y, np.ones(3))


def test_occamdata_load_with_non_npy_files_ignored(tmp_path):
    f1, f7, xyz = make_fakes(tmp_path)
    OccamData(f1, f7, xyz, silent=True)
    (tmp_path / 'class_data' / 'notes.txt').write_text('ignore me')
    od = OccamData(str(tmp_path), silent=True)
    assert not hasattr(od, 'notes')
    assert np.array_equal(od.x, np.arange(5.0))
