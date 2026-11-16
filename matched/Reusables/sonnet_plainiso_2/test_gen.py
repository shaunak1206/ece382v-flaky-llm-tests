import os

import pytest

import reusables
from reusables.file_operations import directory_duplicates


def _write(path, content):
    parent = os.path.dirname(str(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(str(path), "wb") as f:
        f.write(content)
    return str(path)


def _normalize(groups):
    return sorted(sorted(os.path.abspath(p) for p in g) for g in groups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_no_duplicates_different_content(tmp_path):
    _write(tmp_path / "a.txt", b"hello")
    _write(tmp_path / "b.txt", b"world!!")
    assert directory_duplicates(str(tmp_path)) == []


def test_simple_duplicate_pair(tmp_path):
    a = _write(tmp_path / "a.txt", b"same content")
    b = _write(tmp_path / "b.txt", b"same content")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_same_size_different_content_not_duplicates(tmp_path):
    _write(tmp_path / "a.txt", b"abcde")
    _write(tmp_path / "b.txt", b"abcdf")
    assert directory_duplicates(str(tmp_path)) == []


def test_duplicates_in_subdirectories(tmp_path):
    a = _write(tmp_path / "x" / "a.txt", b"nested data")
    b = _write(tmp_path / "y" / "z" / "b.txt", b"nested data")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_three_duplicates_in_one_group(tmp_path):
    paths = [_write(tmp_path / (n + ".txt"), b"triple") for n in "abc"]
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted(paths)


def test_multiple_distinct_duplicate_groups(tmp_path):
    a1 = _write(tmp_path / "a1.txt", b"group one")
    a2 = _write(tmp_path / "a2.txt", b"group one")
    b1 = _write(tmp_path / "b1.txt", b"group two!")
    b2 = _write(tmp_path / "b2.txt", b"group two!")
    _write(tmp_path / "unique.txt", b"unique")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 2
    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_returns_list_of_lists_of_strings(tmp_path):
    _write(tmp_path / "a.txt", b"data")
    _write(tmp_path / "b.txt", b"data")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    for group in result:
        assert isinstance(group, list)
        for item in group:
            assert isinstance(item, str)


def test_empty_files_are_duplicates(tmp_path):
    a = _write(tmp_path / "e1", b"")
    b = _write(tmp_path / "e2", b"")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_single_file_no_duplicates(tmp_path):
    _write(tmp_path / "only.txt", b"alone")
    assert directory_duplicates(str(tmp_path)) == []


def test_hash_type_sha256(tmp_path):
    a = _write(tmp_path / "a.bin", b"\x00\x01\x02\x03")
    b = _write(tmp_path / "b.bin", b"\x00\x01\x02\x03")
    result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_invalid_hash_type_raises(tmp_path):
    _write(tmp_path / "a.txt", b"data")
    _write(tmp_path / "b.txt", b"data")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not_a_real_hash")


def test_ext_kwarg_filters_files(tmp_path):
    t1 = _write(tmp_path / "a.txt", b"filter me")
    t2 = _write(tmp_path / "b.txt", b"filter me")
    _write(tmp_path / "c.log", b"filter me")
    result = directory_duplicates(str(tmp_path), ext=".txt")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([t1, t2])


def test_name_kwarg_filters_files(tmp_path):
    k1 = _write(tmp_path / "keep_1.dat", b"payload")
    k2 = _write(tmp_path / "keep_2.dat", b"payload")
    _write(tmp_path / "other.dat", b"payload")
    result = directory_duplicates(str(tmp_path), name="keep")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([k1, k2])


def test_depth_kwarg_limits_search(tmp_path):
    _write(tmp_path / "top.txt", b"depth test")
    _write(tmp_path / "sub" / "deep.txt", b"depth test")
    shallow = directory_duplicates(str(tmp_path), depth=1)
    assert shallow == []
    full = directory_duplicates(str(tmp_path))
    assert len(full) == 1
    assert len(full[0]) == 2


def test_ext_invalid_type_raises_type_error(tmp_path):
    _write(tmp_path / "a.txt", b"x")
    with pytest.raises(TypeError):
        directory_duplicates(str(tmp_path), ext=123)


def test_exported_from_package():
    assert reusables.directory_duplicates is directory_duplicates
