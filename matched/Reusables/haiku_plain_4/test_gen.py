import hashlib

import pytest

import reusables.file_operations as file_ops
from reusables.file_operations import directory_duplicates


def _write(path, data):
    """Create a file (and parent directories) with the given bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _norm(groups):
    """Sort group members and the groups themselves for order-independent comparison."""
    return sorted(sorted(group) for group in groups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_nonexistent_directory_returns_empty_list(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert directory_duplicates(str(missing)) == []


def test_single_file_has_no_duplicates(tmp_path):
    _write(tmp_path / "only.txt", b"hello")
    assert directory_duplicates(str(tmp_path)) == []


def test_files_with_unique_sizes_have_no_duplicates(tmp_path):
    _write(tmp_path / "a.txt", b"a")
    _write(tmp_path / "b.txt", b"bb")
    _write(tmp_path / "c.txt", b"ccc")
    assert directory_duplicates(str(tmp_path)) == []


def test_same_size_different_content_is_not_duplicate(tmp_path):
    _write(tmp_path / "x.txt", b"aaaa")
    _write(tmp_path / "y.txt", b"bbbb")
    assert directory_duplicates(str(tmp_path)) == []


def test_identical_pair_in_same_directory(tmp_path):
    a = _write(tmp_path / "a.txt", b"same content")
    b = _write(tmp_path / "b.txt", b"same content")
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a, b]])


def test_duplicates_detected_regardless_of_file_name(tmp_path):
    a = _write(tmp_path / "completely_different_name.dat", b"payload")
    b = _write(tmp_path / "zzz.bin", b"payload")
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a, b]])


def test_identical_pair_across_subdirectories(tmp_path):
    a = _write(tmp_path / "photos" / "img.jpg", b"pixels" * 10)
    b = _write(tmp_path / "backup" / "deep" / "img_copy.jpg", b"pixels" * 10)
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a, b]])


def test_three_identical_files_form_one_group(tmp_path):
    a = _write(tmp_path / "1.txt", b"triple")
    b = _write(tmp_path / "2.txt", b"triple")
    c = _write(tmp_path / "sub" / "3.txt", b"triple")
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert _norm(result) == _norm([[a, b, c]])


def test_multiple_independent_duplicate_groups(tmp_path):
    a1 = _write(tmp_path / "a1.txt", b"group-one")
    a2 = _write(tmp_path / "a2.txt", b"group-one")
    b1 = _write(tmp_path / "b1.txt", b"group-two-longer")
    b2 = _write(tmp_path / "b2.txt", b"group-two-longer")
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a1, a2], [b1, b2]])


def test_unique_files_excluded_from_mixed_directory(tmp_path):
    a = _write(tmp_path / "dup1.txt", b"shared")
    b = _write(tmp_path / "dup2.txt", b"shared")
    _write(tmp_path / "unique_short.txt", b"xy")
    _write(tmp_path / "unique_long.txt", b"something else entirely")
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a, b]])


def test_same_size_same_prefix_different_tail_not_duplicate(tmp_path):
    # Identical first bytes and size, but the full hash must differ.
    _write(tmp_path / "left.bin", b"HEADER" + b"1" * 100)
    _write(tmp_path / "right.bin", b"HEADER" + b"2" * 100)
    assert directory_duplicates(str(tmp_path)) == []


def test_empty_files_are_treated_as_duplicates(tmp_path):
    a = _write(tmp_path / "empty1", b"")
    b = _write(tmp_path / "nested" / "empty2", b"")
    result = directory_duplicates(str(tmp_path))
    assert _norm(result) == _norm([[a, b]])


def test_return_type_is_list_of_lists_with_min_two_members(tmp_path):
    _write(tmp_path / "p.txt", b"abc")
    _write(tmp_path / "q.txt", b"abc")
    _write(tmp_path / "solo.txt", b"unique!!")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    for group in result:
        assert isinstance(group, list)
        assert len(group) > 1


def test_returned_paths_are_inside_searched_directory(tmp_path):
    _write(tmp_path / "x.txt", b"inside")
    _write(tmp_path / "sub" / "y.txt", b"inside")
    result = directory_duplicates(str(tmp_path))
    assert result
    for group in result:
        for path in group:
            assert path.startswith(str(tmp_path))


def test_ext_filter_restricts_candidates(tmp_path):
    a = _write(tmp_path / "a.txt", b"matching")
    b = _write(tmp_path / "a2.txt", b"matching")
    _write(tmp_path / "c.bin", b"ignored!")
    _write(tmp_path / "d.bin", b"ignored!")
    result = directory_duplicates(str(tmp_path), ext=".txt")
    assert _norm(result) == _norm([[a, b]])


def test_name_filter_restricts_candidates(tmp_path):
    a = _write(tmp_path / "report_1.txt", b"quarterly")
    b = _write(tmp_path / "report_2.txt", b"quarterly")
    _write(tmp_path / "other_1.txt", b"quarterly")
    _write(tmp_path / "other_2.txt", b"quarterly")
    result = directory_duplicates(str(tmp_path), name="report")
    assert _norm(result) == _norm([[a, b]])


def test_depth_limits_search_to_top_level(tmp_path):
    top_a = _write(tmp_path / "a.txt", b"top level")
    top_b = _write(tmp_path / "a_copy.txt", b"top level")
    _write(tmp_path / "sub" / "b.txt", b"nested dup")
    _write(tmp_path / "sub" / "b_copy.txt", b"nested dup")

    everything = directory_duplicates(str(tmp_path))
    assert len(everything) == 2

    limited = directory_duplicates(str(tmp_path), depth=1)
    assert _norm(limited) == _norm([[top_a, top_b]])


def test_sha256_hash_type_gives_same_grouping(tmp_path):
    a = _write(tmp_path / "a.txt", b"hash me")
    b = _write(tmp_path / "b.txt", b"hash me")
    _write(tmp_path / "c.txt", b"different!")
    md5_result = directory_duplicates(str(tmp_path))
    sha_result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert _norm(md5_result) == _norm(sha_result) == _norm([[a, b]])


def test_default_hash_type_is_md5(tmp_path, monkeypatch):
    _write(tmp_path / "a.txt", b"check hash type")
    _write(tmp_path / "b.txt", b"check hash type")

    seen = []
    original = file_ops.file_hash

    def spy(path, hash_type="md5", **kwargs):
        seen.append(hash_type)
        return original(path, hash_type=hash_type, **kwargs)

    monkeypatch.setattr(file_ops, "file_hash", spy)
    directory_duplicates(str(tmp_path))
    assert seen
    assert set(seen) == {"md5"}


def test_invalid_hash_type_raises_value_error(tmp_path):
    _write(tmp_path / "a.txt", b"needs hashing")
    _write(tmp_path / "b.txt", b"needs hashing")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not-a-real-hash")


def test_hashing_skipped_when_no_size_collisions(tmp_path, monkeypatch):
    _write(tmp_path / "a.txt", b"1")
    _write(tmp_path / "b.txt", b"22")
    _write(tmp_path / "c.txt", b"333")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("file_hash should not be called for unique sizes")

    monkeypatch.setattr(file_ops, "file_hash", fail_if_called)
    assert directory_duplicates(str(tmp_path)) == []
