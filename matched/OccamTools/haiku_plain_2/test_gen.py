import json
import os
import warnings

import numpy as np
import pytest

from occamtools import occam_data as od


class FakeFort(object):
    """Lightweight stand-in for Fort1/Fort7/Xyz readers."""

    def __init__(self, file_name, **attrs):
        self.file_name = file_name
        self.__dict__.update(attrs)
        self.read_silent = None

    def read_file(self, silent=False):
        self.read_silent = silent


class FakeFort1(FakeFort):
    pass


class FakeFort7(FakeFort):
    pass


class FakeXyz(FakeFort):
    pass


@pytest.fixture
def fake_forts(monkeypatch):
    monkeypatch.setattr(od, 'Fort1', FakeFort1)
    monkeypatch.setattr(od, 'Fort7', FakeFort7)
    monkeypatch.setattr(od, 'Xyz', FakeXyz)


def make_blank(**attrs):
    """Create an OccamData instance without running the constructor."""
    obj = od.OccamData.__new__(od.OccamData)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def write_dump(path, arrays, attrs):
    os.makedirs(path, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(path, name + '.npy'), arr)
    with open(os.path.join(path, 'class.json'), 'w') as f:
        json.dump(attrs, f)


def test_check_npy_dump_exists_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_npy_dump_exists(str(tmp_path / 'missing.1'))


def test_check_npy_dump_exists_detects_x_npy(tmp_path):
    fort1 = tmp_path / 'fort.1'
    fort1.write_text('')
    class_dir = str(tmp_path / 'class_data')

    # No class_data directory yet
    assert od._check_npy_dump_exists(str(fort1)) == (False, class_dir)

    # class_data exists but has no x.npy
    (tmp_path / 'class_data').mkdir()
    assert od._check_npy_dump_exists(str(fort1)) == (False, class_dir)

    # x.npy present
    np.save(str(tmp_path / 'class_data' / 'x.npy'), np.zeros(2))
    assert od._check_npy_dump_exists(str(fort1)) == (True, class_dir)


def test_check_constructor_input_rejects_non_string_single_arg():
    with pytest.raises(TypeError):
        od._check_constructor_input(42)


def test_check_constructor_input_missing_single_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        od._check_constructor_input(str(tmp_path / 'nope.1'))


def test_check_constructor_input_unrecognized_three_args_raises():
    with pytest.raises(ValueError):
        od._check_constructor_input(1, 2, 3)


@pytest.mark.parametrize('name, expected', [
    ('run.1', ('run.1', 'fort.7', 'fort.8')),
    ('run.7', ('fort.1', 'run.7', 'fort.8')),
    ('run.8', ('fort.1', 'fort.7', 'run.8')),
    ('run.xyz', ('fort.1', 'fort.7', 'run.xyz')),
])
def test_check_constructor_input_single_file_derives_sibling_paths(
        tmp_path, monkeypatch, name, expected):
    target = tmp_path / name
    target.write_text('')

    def fake_open(fort1, fort7, xyz, silent, which=None):
        return fort1, fort7, xyz

    monkeypatch.setattr(od, '_open_fort_files', fake_open)
    result = od._check_constructor_input(str(target))
    assert list(result) == [str(tmp_path / n) for n in expected]


def test_check_constructor_input_directory_uses_standard_names(
        tmp_path, monkeypatch):
    run_dir = tmp_path / 'run_dir'
    run_dir.mkdir()

    def fake_open(fort1, fort7, xyz, silent, which=None):
        return fort1, fort7, xyz

    monkeypatch.setattr(od, '_open_fort_files', fake_open)
    f1, f7, x = od._check_constructor_input(str(run_dir))
    assert f1 == os.path.join(str(run_dir), 'fort.1')
    assert f7 == os.path.join(str(run_dir), 'fort.7')
    assert x == os.path.join(str(run_dir), 'fort.8')


def test_check_constructor_input_three_objects_returned_unchanged(
        fake_forts, tmp_path):
    objs = (FakeFort1(str(tmp_path / 'a.1')),
            FakeFort7(str(tmp_path / 'b.7')),
            FakeXyz(str(tmp_path / 'c.xyz')))
    assert od._check_constructor_input(*objs) == objs


@pytest.mark.parametrize('position, expected_which', [(0, 0), (1, 1), (2, 2)])
def test_check_constructor_input_mixed_inputs_selects_open_mode(
        monkeypatch, position, expected_which):
    args = [object(), object(), object()]
    args[position] = 'some/path'
    seen = []

    def fake_open(fort1, fort7, xyz, silent, which=None):
        seen.append((which, silent))
        return 'opened'

    monkeypatch.setattr(od, '_open_fort_files', fake_open)
    assert od._check_constructor_input(*args, silent=True) == 'opened'
    assert seen == [(expected_which, True)]


def test_open_fort_files_without_which_uses_given_paths(tmp_path, fake_forts):
    p1 = str(tmp_path / 'a.1')
    p7 = str(tmp_path / 'b.7')
    px = str(tmp_path / 'c.xyz')
    f1, f7, x = od._open_fort_files(p1, p7, px, silent=True)
    assert isinstance(f1, FakeFort1) and f1.file_name == p1
    assert isinstance(f7, FakeFort7) and f7.file_name == p7
    assert isinstance(x, FakeXyz) and x.file_name == px
    assert f1.read_silent is True
    assert f7.read_silent is True
    assert x.read_silent is True


@pytest.mark.parametrize('which, given, expected', [
    (0, 'run.1', ('run.1', 'fort.7', 'fort.8')),
    (1, 'run.7', ('fort.1', 'run.7', 'fort.8')),
    (2, 'run.xyz', ('fort.1', 'fort.7', 'run.xyz')),
])
def test_open_fort_files_with_which_derives_missing_paths(
        tmp_path, fake_forts, which, given, expected):
    paths = [None, None, None]
    paths[which] = str(tmp_path / given)
    f1, f7, x = od._open_fort_files(*paths, silent=False, which=which)
    assert f1.file_name == str(tmp_path / expected[0])
    assert f7.file_name == str(tmp_path / expected[1])
    assert x.file_name == str(tmp_path / expected[2])


def test_check_internal_consistency_matching_values_no_warning(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), temperature=300.0, label='run')
    b = FakeFort7(str(tmp_path / 'b'), temperature=300.0, label='run')
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_check_internal_consistency_numeric_mismatch_warns(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), temperature=300.0)
    b = FakeFort7(str(tmp_path / 'b'), temperature=350.0)
    with pytest.warns(UserWarning, match='not internally consistent'):
        assert od._check_internal_consistency(a, b) is False


def test_check_internal_consistency_string_mismatch_warns(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), label='alpha')
    b = FakeFort7(str(tmp_path / 'b'), label='beta')
    with pytest.warns(UserWarning, match='not internally consistent'):
        assert od._check_internal_consistency(a, b) is False


def test_check_internal_consistency_ignores_file_name_and_num_lines(tmp_path):
    a = FakeFort1(str(tmp_path / 'a'), energy=1.0, num_lines=4)
    b = FakeFort7(str(tmp_path / 'b'), energy=1.0, num_lines=99)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert od._check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_check_internal_consistency_all_false_if_any_pair_differs(tmp_path):
    fort1 = FakeFort1(str(tmp_path / 'a'), energy=-1.5)
    fort7 = FakeFort7(str(tmp_path / 'b'), energy=-1.5)
    xyz_ok = FakeXyz(str(tmp_path / 'c'), energy=-1.5)
    xyz_bad = FakeXyz(str(tmp_path / 'c'), energy=-9.0)

    assert od._check_internal_consistency_all(fort1, fort7, xyz_ok) is True
    with pytest.warns(UserWarning):
        assert od._check_internal_consistency_all(
            fort1, fort7, xyz_bad) is False


def test_save_writes_npy_and_json_and_reloads_arrays(tmp_path):
    fort1 = str(tmp_path / 'fort.1')
    obj = make_blank(fort1_file_name=fort1, name='water',
                     x=np.arange(5.0))
    assert obj.save() is True

    save_dir = tmp_path / 'class_data'
    assert (save_dir / 'x.npy').is_file()
    assert (save_dir / 'class.json').is_file()
    with open(str(save_dir / 'class.json')) as f:
        saved = json.load(f)
    assert saved['name'] == 'water'
    assert 'x' not in saved
    # Arrays are reloaded into the object at the end of save()
    np.testing.assert_array_equal(obj.x, np.arange(5.0))


def test_save_returns_false_when_dump_exists_unless_overwrite(tmp_path):
    fort1 = str(tmp_path / 'fort.1')
    (tmp_path / 'class_data').mkdir()
    obj = make_blank(fort1_file_name=fort1, x=np.zeros(3))

    assert obj.save() is False
    assert not (tmp_path / 'class_data' / 'x.npy').exists()

    assert obj.save(overwrite=True) is True
    assert (tmp_path / 'class_data' / 'x.npy').is_file()


def test_load_restores_arrays_and_attributes_ignoring_non_npy(tmp_path):
    class_dir = tmp_path / 'class_data'
    write_dump(str(class_dir), {'x': np.ones(4)},
               {'temperature': 300.0, 'label': 'abc'})
    (class_dir / 'notes.txt').write_text('not data')

    obj = make_blank()
    obj.load(str(class_dir), silent=True)

    np.testing.assert_array_equal(obj.x, np.ones(4))
    assert obj.temperature == 300.0
    assert obj.label == 'abc'
    assert not hasattr(obj, 'notes')


def test_load_reports_progress_when_not_silent(tmp_path, capsys):
    class_dir = tmp_path / 'class_data'
    write_dump(str(class_dir), {'x': np.arange(3.0)}, {'label': 'loud'})

    obj = make_blank()
    obj.load(str(class_dir), silent=False)

    captured = capsys.readouterr()
    assert 'Loading data' in captured.out
    np.testing.assert_array_equal(obj.x, np.arange(3.0))


@pytest.mark.parametrize('mode', ['file', 'directory'])
def test_constructor_loads_existing_dump_without_reading_fort_files(
        tmp_path, monkeypatch, mode):
    fort1 = tmp_path / 'fort.1'
    fort1.write_text('')
    write_dump(str(tmp_path / 'class_data'), {'x': np.ones(4)},
               {'temperature': 300.0, 'fort1_file_name': str(fort1)})

    def must_not_be_called(*args, **kwargs):
        raise AssertionError('raw files should not be parsed')

    monkeypatch.setattr(od, '_check_constructor_input', must_not_be_called)
    target = str(fort1) if mode == 'file' else str(tmp_path)

    obj = od.OccamData(target, silent=True)
    assert obj.temperature == 300.0
    np.testing.assert_array_equal(obj.x, np.ones(4))


def test_constructor_merges_attributes_and_skips_ignored_keys(
        tmp_path, fake_forts):
    f1 = str(tmp_path / 'fort.1')
    f7 = str(tmp_path / 'fort.7')
    fx = str(tmp_path / 'fort.8')
    fort1 = FakeFort1(f1, temperature=300.0, n_time_steps_=10, num_lines=4,
                      comment_format_known=True, file_contents='raw')
    fort7 = FakeFort7(f7, temperature=300.0, energy=-1.5)
    xyz = FakeXyz(fx, energy=-1.5, num_lines=3)

    obj = od.OccamData(fort1, fort7, xyz, save_to_npy=False)

    assert obj.consistent is True
    assert obj.temperature == 300.0
    assert obj.energy == -1.5
    for key in ('n_time_steps_', 'file_contents', 'comment_format_known',
                'num_lines', 'file_name'):
        assert not hasattr(obj, key)
    assert obj.fort1_file_name == f1
    assert obj.fort7_file_name == f7
    assert obj.xyz_file_name == fx


def test_constructor_with_save_to_npy_writes_dump_next_to_fort1(
        tmp_path, fake_forts):
    f1 = str(tmp_path / 'fort.1')
    fort1 = FakeFort1(f1, temperature=300.0, grid=np.arange(3.0))
    fort7 = FakeFort7(str(tmp_path / 'fort.7'), temperature=300.0)
    xyz = FakeXyz(str(tmp_path / 'fort.8'), temperature=300.0)

    od.OccamData(fort1, fort7, xyz, save_to_npy=True)

    save_dir = tmp_path / 'class_data'
    assert (save_dir / 'class.json').is_file()
    assert (save_dir / 'grid.npy').is_file()
