import os

import pytest

import reusables
from reusables.file_operations import directory_duplicates


def _write(path, content):
    parent = os.path.dirname(str(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(str(path), mode) as f:
        f.write(content)
    return str(path)


def _normalize(groups):
    return sorted(sorted(os.path.abspath(p) for p in g) for g in groups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_single_file_no_duplicates(tmp_path):
    _write(tmp_path / "a.txt", "hello")
    assert directory_duplicates(str(tmp_path)) == []


def test_two_identical_files_found(tmp_path):
    a = _write(tmp_path / "a.txt", "same content")
    b = _write(tmp_path / "b.txt", "same content")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_different_content_same_size_not_duplicates(tmp_path):
    _write(tmp_path / "a.txt", "abcd")
    _write(tmp_path / "b.txt", "abce")
    assert directory_duplicates(str(tmp_path)) == []


def test_different_sizes_not_duplicates(tmp_path):
    _write(tmp_path / "a.txt", "abcd")
    _write(tmp_path / "b.txt", "abcde")
    assert directory_duplicates(str(tmp_path)) == []


def test_duplicates_in_subdirectories(tmp_path):
    a = _write(tmp_path / "x" / "a.txt", "dup data")
    b = _write(tmp_path / "y" / "z" / "b.txt", "dup data")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_three_identical_files_in_one_group(tmp_path):
    paths = [_write(tmp_path / (n + ".txt"), "triple") for n in "abc"]
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted(paths)


def test_multiple_duplicate_groups(tmp_path):
    a1 = _write(tmp_path / "a1.txt", "group a")
    a2 = _write(tmp_path / "a2.txt", "group a")
    b1 = _write(tmp_path / "b1.txt", "group bb")
    b2 = _write(tmp_path / "b2.txt", "group bb")
    _write(tmp_path / "unique.txt", "something unique")
    result = directory_duplicates(str(tmp_path))
    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_same_size_groups_separated_by_hash(tmp_path):
    a1 = _write(tmp_path / "a1.txt", "aaaa")
    a2 = _write(tmp_path / "a2.txt", "aaaa")
    b1 = _write(tmp_path / "b1.txt", "bbbb")
    b2 = _write(tmp_path / "b2.txt", "bbbb")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 2
    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_empty_files_are_duplicates(tmp_path):
    a = _write(tmp_path / "e1", "")
    b = _write(tmp_path / "e2", "")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_binary_duplicates(tmp_path):
    data = bytes(bytearray(range(256)))
    a = _write(tmp_path / "a.bin", data)
    b = _write(tmp_path / "b.bin", data)
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_hash_type_sha256(tmp_path):
    a = _write(tmp_path / "a.txt", "sha content")
    b = _write(tmp_path / "b.txt", "sha content")
    result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_invalid_hash_type_raises(tmp_path):
    _write(tmp_path / "a.txt", "content")
    _write(tmp_path / "b.txt", "content")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not_a_real_hash")


def test_ext_kwarg_filters_files(tmp_path):
    t1 = _write(tmp_path / "a.txt", "filter me")
    t2 = _write(tmp_path / "b.txt", "filter me")
    _write(tmp_path / "c.log", "filter me")
    _write(tmp_path / "d.log", "filter me")
    result = directory_duplicates(str(tmp_path), ext=".txt")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([t1, t2])


def test_name_kwarg_filters_files(tmp_path):
    k1 = _write(tmp_path / "keep_1.dat", "xyz")
    k2 = _write(tmp_path / "keep_2.dat", "xyz")
    _write(tmp_path / "other_1.dat", "xyz")
    _write(tmp_path / "other_2.dat", "xyz")
    result = directory_duplicates(str(tmp_path), name="keep")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([k1, k2])


def test_depth_kwarg_limits_search(tmp_path):
    _write(tmp_path / "top.txt", "depth data")
    _write(tmp_path / "sub" / "deep.txt", "depth data")
    result = directory_duplicates(str(tmp_path), depth=1)
    assert result == []


def test_returns_list_of_lists_of_strings(tmp_path):
    _write(tmp_path / "a.txt", "types")
    _write(tmp_path / "b.txt", "types")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    assert all(isinstance(g, list) for g in result)
    assert all(isinstance(p, str) for g in result for p in g)


def test_exposed_at_package_level():
    assert reusables.directory_duplicates is directory_duplicates
