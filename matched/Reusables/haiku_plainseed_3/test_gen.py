import os

import pytest

import reusables.file_operations as file_ops


def _write(path, data):
    """Create parent directories as needed and write bytes to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _normalize(groups):
    """Order-insensitive representation of a list of duplicate groups."""
    return sorted(sorted(group) for group in groups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert file_ops.directory_duplicates(str(tmp_path)) == []


def test_single_file_has_no_duplicates(tmp_path):
    _write(tmp_path / "only.txt", b"hello")
    assert file_ops.directory_duplicates(str(tmp_path)) == []


def test_two_identical_files_form_one_group(tmp_path):
    a = _write(tmp_path / "a.txt", b"same content")
    b = _write(tmp_path / "b.txt", b"same content")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[a, b]])


def test_return_type_is_list_of_lists_of_strings(tmp_path):
    _write(tmp_path / "a.txt", b"x")
    _write(tmp_path / "b.txt", b"x")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(p, str) for p in result[0])


def test_same_size_different_content_not_grouped(tmp_path):
    _write(tmp_path / "a.txt", b"aaaa")
    _write(tmp_path / "b.txt", b"bbbb")

    assert file_ops.directory_duplicates(str(tmp_path)) == []


def test_mixed_same_size_only_matching_content_grouped(tmp_path):
    a = _write(tmp_path / "a.txt", b"match")
    b = _write(tmp_path / "b.txt", b"match")
    _write(tmp_path / "c.txt", b"other")  # same size, different content

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[a, b]])


def test_multiple_independent_groups_found(tmp_path):
    a1 = _write(tmp_path / "a1.txt", b"first group")
    a2 = _write(tmp_path / "a2.txt", b"first group")
    b1 = _write(tmp_path / "b1.dat", b"second")
    b2 = _write(tmp_path / "b2.dat", b"second")
    _write(tmp_path / "unique.dat", b"unique!")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[a1, a2], [b1, b2]])


def test_three_identical_files_form_single_group(tmp_path):
    paths = [_write(tmp_path / name, b"triple") for name in
             ("one.txt", "two.txt", "three.txt")]

    result = file_ops.directory_duplicates(str(tmp_path))

    assert len(result) == 1
    assert _normalize(result) == _normalize([paths])


def test_duplicates_found_in_subdirectories(tmp_path):
    top = _write(tmp_path / "top.txt", b"nested dup")
    deep = _write(tmp_path / "sub" / "deeper" / "copy.txt", b"nested dup")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[top, deep]])


def test_duplicate_detection_ignores_file_names(tmp_path):
    a = _write(tmp_path / "completely_different_name.xyz", b"payload")
    b = _write(tmp_path / "zzz.abc", b"payload")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[a, b]])


def test_empty_files_are_treated_as_duplicates(tmp_path):
    e1 = _write(tmp_path / "empty1", b"")
    e2 = _write(tmp_path / "empty2", b"")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert _normalize(result) == _normalize([[e1, e2]])


def test_default_hash_type_is_md5(tmp_path, monkeypatch):
    _write(tmp_path / "a.txt", b"hash me")
    _write(tmp_path / "b.txt", b"hash me")

    calls = []
    original = file_ops.file_hash

    def spy(path, hash_type="md5", *args, **kwargs):
        calls.append(hash_type)
        return original(path, hash_type, *args, **kwargs)

    monkeypatch.setattr(file_ops, "file_hash", spy)
    file_ops.directory_duplicates(str(tmp_path))

    assert calls and all(h == "md5" for h in calls)


def test_sha256_hash_type_gives_same_grouping(tmp_path):
    a = _write(tmp_path / "a.txt", b"sha content")
    b = _write(tmp_path / "b.txt", b"sha content")
    _write(tmp_path / "c.txt", b"sha contenX")

    result = file_ops.directory_duplicates(str(tmp_path), hash_type="sha256")

    assert _normalize(result) == _normalize([[a, b]])


def test_unsupported_hash_type_raises_value_error(tmp_path):
    # Two identical files are needed so that hashing actually happens.
    _write(tmp_path / "a.txt", b"will be hashed")
    _write(tmp_path / "b.txt", b"will be hashed")

    with pytest.raises(ValueError):
        file_ops.directory_duplicates(str(tmp_path),
                                      hash_type="not-a-real-hash")


def test_unique_sizes_are_not_hashed(tmp_path, monkeypatch):
    unique = _write(tmp_path / "unique.txt", b"aaa")  # only file of size 3
    _write(tmp_path / "b.txt", b"aaaa")
    _write(tmp_path / "c.txt", b"aaaa")  # size 4 duplicates

    hashed = []
    original = file_ops.file_hash

    def spy(path, *args, **kwargs):
        hashed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(file_ops, "file_hash", spy)
    file_ops.directory_duplicates(str(tmp_path))

    assert unique not in hashed
    assert len(hashed) == 2


def test_ext_kwarg_limits_candidate_files(tmp_path):
    a = _write(tmp_path / "a.txt", b"ext test")
    b = _write(tmp_path / "b.txt", b"ext test")
    _write(tmp_path / "c.bin", b"ext test")  # identical but excluded by ext

    result = file_ops.directory_duplicates(str(tmp_path), ext=".txt")

    assert _normalize(result) == _normalize([[a, b]])


def test_name_kwarg_limits_candidate_files(tmp_path):
    d1 = _write(tmp_path / "dup_1.bin", b"name test")
    d2 = _write(tmp_path / "dup_2.bin", b"name test")
    _write(tmp_path / "other.bin", b"name test")

    result = file_ops.directory_duplicates(str(tmp_path), name="dup")

    assert _normalize(result) == _normalize([[d1, d2]])


def test_invalid_ext_type_raises_type_error(tmp_path):
    _write(tmp_path / "a.txt", b"x")
    _write(tmp_path / "b.txt", b"x")

    with pytest.raises(TypeError):
        file_ops.directory_duplicates(str(tmp_path), ext=5)


def test_depth_kwarg_limits_search_depth(tmp_path):
    top_a = _write(tmp_path / "a.txt", b"depth test")
    top_b = _write(tmp_path / "b.txt", b"depth test")
    _write(tmp_path / "sub" / "c.txt", b"depth test")

    result = file_ops.directory_duplicates(str(tmp_path), depth=1)

    assert _normalize(result) == _normalize([[top_a, top_b]])


def test_abspath_kwarg_returns_absolute_paths(tmp_path):
    _write(tmp_path / "a.txt", b"abs")
    _write(tmp_path / "b.txt", b"abs")

    result = file_ops.directory_duplicates(str(tmp_path), abspath=True)

    assert len(result) == 1
    assert all(os.path.isabs(p) for p in result[0])


def test_returned_paths_exist_inside_directory(tmp_path):
    _write(tmp_path / "a.txt", b"exists")
    _write(tmp_path / "sub" / "b.txt", b"exists")

    result = file_ops.directory_duplicates(str(tmp_path))

    assert len(result) == 1
    for path in result[0]:
        assert os.path.exists(path)
        assert os.path.commonpath([os.path.abspath(path),
                                   str(tmp_path)]) == str(tmp_path)


def test_nonexistent_directory_returns_empty_list(tmp_path):
    missing = str(tmp_path / "does_not_exist")

    assert file_ops.directory_duplicates(missing) == []
