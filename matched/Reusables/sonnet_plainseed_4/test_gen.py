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


def test_returns_list(tmp_path):
    _write(tmp_path / "a.txt", "hello")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_no_duplicates_different_content(tmp_path):
    _write(tmp_path / "a.txt", "aaa")
    _write(tmp_path / "b.txt", "bbbb")
    _write(tmp_path / "c.txt", "cc")
    assert directory_duplicates(str(tmp_path)) == []


def test_single_file_no_duplicates(tmp_path):
    _write(tmp_path / "only.txt", "unique")
    assert directory_duplicates(str(tmp_path)) == []


def test_simple_duplicate_pair(tmp_path):
    a = _write(tmp_path / "a.txt", "same content")
    b = _write(tmp_path / "b.txt", "same content")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_same_size_different_content_not_duplicates(tmp_path):
    _write(tmp_path / "a.txt", "abcdef")
    _write(tmp_path / "b.txt", "abcdeg")
    assert directory_duplicates(str(tmp_path)) == []


def test_duplicates_in_subdirectories(tmp_path):
    a = _write(tmp_path / "a.txt", "nested dup")
    b = _write(tmp_path / "sub" / "deep" / "b.txt", "nested dup")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_multiple_duplicate_groups(tmp_path):
    a1 = _write(tmp_path / "a1.txt", "group A")
    a2 = _write(tmp_path / "a2.txt", "group A")
    b1 = _write(tmp_path / "b1.txt", "group B!")
    b2 = _write(tmp_path / "b2.txt", "group B!")
    _write(tmp_path / "unique.txt", "something else entirely")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 2
    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_group_of_three_duplicates(tmp_path):
    paths = [_write(tmp_path / "f{}.txt".format(i), "triple") for i in range(3)]
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted(paths)


def test_same_size_groups_separated_by_hash(tmp_path):
    a1 = _write(tmp_path / "a1.txt", "xxxx")
    a2 = _write(tmp_path / "a2.txt", "xxxx")
    b1 = _write(tmp_path / "b1.txt", "yyyy")
    b2 = _write(tmp_path / "b2.txt", "yyyy")
    result = directory_duplicates(str(tmp_path))
    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_empty_files_are_duplicates(tmp_path):
    a = _write(tmp_path / "e1", "")
    b = _write(tmp_path / "e2", "")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_binary_content_duplicates(tmp_path):
    data = bytes(bytearray(range(256)))
    a = _write(tmp_path / "a.bin", data)
    b = _write(tmp_path / "b.bin", data)
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_hash_type_sha256(tmp_path):
    a = _write(tmp_path / "a.txt", "hash me")
    b = _write(tmp_path / "b.txt", "hash me")
    _write(tmp_path / "c.txt", "hash ya!")
    result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_invalid_hash_type_raises(tmp_path):
    _write(tmp_path / "a.txt", "dup")
    _write(tmp_path / "b.txt", "dup")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not_a_real_hash")


def test_invalid_hash_type_ignored_without_candidates(tmp_path):
    # No same-size files, so no hashing occurs
    _write(tmp_path / "a.txt", "one")
    _write(tmp_path / "b.txt", "three")
    assert directory_duplicates(str(tmp_path), hash_type="not_a_real_hash") == []


def test_ext_kwarg_filters_files(tmp_path):
    t1 = _write(tmp_path / "a.txt", "filter")
    t2 = _write(tmp_path / "b.txt", "filter")
    _write(tmp_path / "c.log", "filter")
    _write(tmp_path / "d.log", "filter")
    result = directory_duplicates(str(tmp_path), ext=".txt")
    assert len(result) == 1
    assert sorted(result[0]) == sorted([t1, t2])


def test_depth_kwarg_limits_search(tmp_path):
    a = _write(tmp_path / "a.txt", "depth test")
    b = _write(tmp_path / "b.txt", "depth test")
    _write(tmp_path / "x" / "y" / "c.txt", "depth test")
    result = directory_duplicates(str(tmp_path), depth=1)
    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_results_are_strings_not_pathlib(tmp_path):
    _write(tmp_path / "a.txt", "strs")
    _write(tmp_path / "b.txt", "strs")
    result = directory_duplicates(str(tmp_path))
    for group in result:
        for item in group:
            assert isinstance(item, str)


def test_exposed_on_reusables_package(tmp_path):
    assert reusables.directory_duplicates is directory_duplicates
    _write(tmp_path / "a.txt", "pkg")
    _write(tmp_path / "b.txt", "pkg")
    assert len(reusables.directory_duplicates(str(tmp_path))) == 1
