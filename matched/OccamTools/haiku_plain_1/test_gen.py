import json
import os
import warnings

import numpy as np
import pytest

import occamtools.occam_data as occam_data


class FakeReader:
    """Lightweight stand-in for Fort1, Fort7 and Xyz readers."""

    def __init__(self, file_name=None, **attrs):
        self.file_name = file_name
        self.num_lines = 0
        for key, value in attrs.items():
            setattr(self, key, value)

    def read_file(self, silent=False):
        # No-op: fake readers hold their data directly.
        pass


@pytest.fixture
def fakes(monkeypatch):
    class Fort1Fake(FakeReader):
        pass

    class Fort7Fake(FakeReader):
        pass

    class XyzFake(FakeReader):
        pass

    monkeypatch.setattr(occam_data, 'Fort1', Fort1Fake)
    monkeypatch.setattr(occam_data, 'Fort7', Fort7Fake)
    monkeypatch.setattr(occam_data, 'Xyz', XyzFake)
    return Fort1Fake, Fort7Fake, XyzFake


def _make_trio(tmp_path, fakes):
    F1, F7, X = fakes
    f1 = F1(str(tmp_path / 'fort.1'), n_atoms=3, temperature=300.0,
            label='run')
    f7 = F7(str(tmp_path / 'fort.7'), n_steps=10, temperature=300.0,
            label='run')
    x = X(str(tmp_path / 'fort.8'), n_atoms=3,
          x=np.arange(6.0).reshape(2, 3))
    return f1, f7, x


# ---------------------------------------------------------------------------
# _check_internal_consistency
# ---------------------------------------------------------------------------

def test_consistency_matching_numbers_returns_true_without_warning():
    a = FakeReader('a.txt', value=1.0)
    b = FakeReader('b.txt', value=1.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert occam_data._check_internal_consistency(a, b) is True
    assert len(record) == 0


def test_consistency_numeric_mismatch_warns_and_returns_false():
    a = FakeReader('a.txt', value=1.0)
    b = FakeReader('b.txt', value=2.0)
    with pytest.warns(UserWarning, match='not internally consistent'):
        result = occam_data._check_internal_consistency(a, b)
    assert result is False


def test_consistency_string_mismatch_warns_and_returns_false():
    a = FakeReader('a.txt', label='alpha')
    b = FakeReader('b.txt', label='beta')
    with pytest.warns(UserWarning):
        result = occam_data._check_internal_consistency(a, b)
    assert result is False


def test_consistency_all_true_when_all_pairs_match(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert occam_data._check_internal_consistency_all(f1, f7, x) is True
    assert len(record) == 0


def test_consistency_all_false_when_one_pair_mismatches(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    f7.temperature = 350.0  # disagrees with fort1 only
    with pytest.warns(UserWarning):
        assert occam_data._check_internal_consistency_all(f1, f7, x) is False


# ---------------------------------------------------------------------------
# _check_npy_dump_exists
# ---------------------------------------------------------------------------

def test_npy_dump_exists_for_directory_with_x_npy(tmp_path):
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'x.npy'), np.zeros(3))

    exists, path = occam_data._check_npy_dump_exists(str(tmp_path))

    assert exists is True
    assert path == os.path.join(str(tmp_path), 'class_data')


def test_npy_dump_exists_for_file_path_uses_parent_directory(tmp_path):
    fort1 = tmp_path / 'fort.1'
    fort1.write_text('')
    class_dir = tmp_path / 'class_data'
    class_dir.mkdir()
    np.save(str(class_dir / 'x.npy'), np.zeros(3))

    exists, path = occam_data._check_npy_dump_exists(str(fort1))

    assert exists is True
    assert path == os.path.join(str(tmp_path), 'class_data')


def test_npy_dump_exists_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        occam_data._check_npy_dump_exists(str(tmp_path / 'missing'))


def test_npy_dump_exists_false_when_directory_has_no_dump(tmp_path):
    exists, path = occam_data._check_npy_dump_exists(str(tmp_path))

    assert exists is False
    assert path == os.path.join(str(tmp_path), 'class_data')


# ---------------------------------------------------------------------------
# _check_constructor_input
# ---------------------------------------------------------------------------

def test_constructor_input_returns_objects_unchanged(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)

    result = occam_data._check_constructor_input(f1, f7, x)

    assert result[0] is f1
    assert result[1] is f7
    assert result[2] is x


def test_constructor_input_non_string_single_arg_raises_type_error():
    with pytest.raises(TypeError):
        occam_data._check_constructor_input(42)


def test_constructor_input_wrong_three_types_raises_value_error():
    with pytest.raises(ValueError):
        occam_data._check_constructor_input(1, 2, 3)


def test_constructor_input_missing_single_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        occam_data._check_constructor_input(str(tmp_path / 'nope.1'))


# ---------------------------------------------------------------------------
# _open_fort_files
# ---------------------------------------------------------------------------

def test_open_fort_files_without_which_uses_given_paths(tmp_path, fakes):
    F1, F7, X = fakes
    p1 = str(tmp_path / 'a.1')
    p7 = str(tmp_path / 'a.7')
    p8 = str(tmp_path / 'a.8')

    f1, f7, x = occam_data._open_fort_files(p1, p7, p8, silent=True)

    assert isinstance(f1, F1) and f1.file_name == p1
    assert isinstance(f7, F7) and f7.file_name == p7
    assert isinstance(x, X) and x.file_name == p8


def test_open_fort_files_derives_sibling_paths_for_which_0_and_2(tmp_path,
                                                                  fakes):
    p1 = str(tmp_path / 'fort.1')
    f1, f7, x = occam_data._open_fort_files(p1, 'unused', 'unused',
                                            silent=True, which=0)
    assert f1.file_name == p1
    assert f7.file_name == os.path.join(str(tmp_path), 'fort.7')
    assert x.file_name == os.path.join(str(tmp_path), 'fort.8')

    p8 = str(tmp_path / 'fort.8')
    f1, f7, x = occam_data._open_fort_files('unused', 'unused', p8,
                                            silent=True, which=2)
    assert f1.file_name == os.path.join(str(tmp_path), 'fort.1')
    assert f7.file_name == os.path.join(str(tmp_path), 'fort.7')
    assert x.file_name == p8


# ---------------------------------------------------------------------------
# OccamData construction (no npy cache involved)
# ---------------------------------------------------------------------------

def test_init_from_objects_merges_attributes_and_skips_ignored(tmp_path,
                                                               fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    f7.file_contents = 'raw text'
    f7.comment_format_known = True
    f7.n_time_steps_ = 10

    obj = occam_data.OccamData(f1, f7, x, save_to_npy=False)

    assert obj.n_atoms == 3
    assert obj.n_steps == 10
    assert obj.temperature == 300.0
    assert np.array_equal(obj.x, np.arange(6.0).reshape(2, 3))
    for key in ('file_name', 'num_lines', 'file_contents',
                'comment_format_known', 'n_time_steps_'):
        assert not hasattr(obj, key)


def test_init_records_file_names_and_consistent_flag(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)

    obj = occam_data.OccamData(f1, f7, x, save_to_npy=False)

    assert obj.fort1_file_name == f1.file_name
    assert obj.fort7_file_name == f7.file_name
    assert obj.xyz_file_name == x.file_name
    assert obj.consistent is True


def test_init_flags_inconsistent_data_with_warning(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    f7.temperature = 350.0

    with pytest.warns(UserWarning):
        obj = occam_data.OccamData(f1, f7, x, save_to_npy=False)

    assert obj.consistent is False


def test_init_single_fort1_path_derives_companion_files(tmp_path, fakes):
    fort1 = tmp_path / 'fort.1'
    fort1.write_text('')

    obj = occam_data.OccamData(str(fort1), save_to_npy=False)

    assert obj.fort1_file_name == str(fort1)
    assert obj.fort7_file_name == os.path.join(str(tmp_path), 'fort.7')
    assert obj.xyz_file_name == os.path.join(str(tmp_path), 'fort.8')


def test_init_nonexistent_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        occam_data.OccamData(str(tmp_path / 'missing'))


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------

def test_save_writes_npy_for_arrays_and_json_for_rest(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    obj = occam_data.OccamData(f1, f7, x, save_to_npy=False)

    assert obj.save() is True

    class_dir = tmp_path / 'class_data'
    assert (class_dir / 'x.npy').exists()
    assert not (class_dir / 'label.npy').exists()
    assert (class_dir / 'class.json').exists()

    with open(str(class_dir / 'class.json')) as handle:
        data = json.load(handle)
    assert data['n_atoms'] == 3
    assert 'x' not in data
    # Arrays are reloaded onto the object after saving.
    assert isinstance(obj.x, np.ndarray)


def test_save_respects_overwrite_flag(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    obj = occam_data.OccamData(f1, f7, x, save_to_npy=False)

    assert obj.save() is True
    assert obj.save() is False
    assert obj.save(overwrite=True) is True


def test_round_trip_load_from_directory_restores_object(tmp_path, fakes):
    f1, f7, x = _make_trio(tmp_path, fakes)
    original = occam_data.OccamData(f1, f7, x, save_to_npy=False)
    original.save()

    loaded = occam_data.OccamData(str(tmp_path), silent=True)

    np.testing.assert_array_equal(loaded.x, original.x)
    assert loaded.n_atoms == 3
    assert loaded.label == 'run'
    assert loaded.fort1_file_name == f1.file_name
    assert loaded.consistent is True
