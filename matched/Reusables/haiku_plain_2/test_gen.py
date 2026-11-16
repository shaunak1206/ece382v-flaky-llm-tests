import os

import pytest

from reusables import file_operations
from reusables.file_operations import directory_duplicates


def _write(path, data):
    """Write bytes to path, creating parent directories as needed."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _names(group):
    """Sorted base names of the files in one duplicate group."""
    return sorted(os.path.basename(p) for p in group)


def _name_groups(dups):
    """Duplicate result as a sorted list of sorted name lists."""
    return sorted(_names(g) for g in dups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_single_file_has_no_duplicates(tmp_path):
    _write(tmp_path / "only.txt", b"hello")
    assert directory_duplicates(str(tmp_path)) == []


def test_identical_pair_is_grouped(tmp_path):
    _write(tmp_path / "a.txt", b"same content")
    _write(tmp_path / "b.txt", b"same content")
    dups = directory_duplicates(str(tmp_path))
    assert _name_groups(dups) == [["a.txt", "b.txt"]]


def test_same_size_different_content_is_not_duplicate(tmp_path):
    _write(tmp_path / "a.bin", b"AAAA")
    _write(tmp_path / "b.bin", b"BBBB")
    assert directory_duplicates(str(tmp_path)) == []


def test_different_sizes_are_not_duplicates(tmp_path):
    _write(tmp_path / "a.txt", b"abc")
    _write(tmp_path / "b.txt", b"abcd")
    assert directory_duplicates(str(tmp_path)) == []


def test_three_identical_files_form_one_group(tmp_path):
    for name in ("a.txt", "b.txt", "c.txt"):
        _write(tmp_path / name, b"triplet")
    dups = directory_duplicates(str(tmp_path))
    assert len(dups) == 1
    assert _names(dups[0]) == ["a.txt", "b.txt", "c.txt"]


def test_multiple_independent_duplicate_groups(tmp_path):
    # Same size for all four files, but two different contents.
    _write(tmp_path / "x1.dat", b"xxxx")
    _write(tmp_path / "x2.dat", b"xxxx")
    _write(tmp_path / "y1.dat", b"yyyy")
    _write(tmp_path / "y2.dat", b"yyyy")
    dups = directory_duplicates(str(tmp_path))
    assert _name_groups(dups) == [["x1.dat", "x2.dat"], ["y1.dat", "y2.dat"]]


def test_unique_files_are_excluded_from_results(tmp_path):
    _write(tmp_path / "dup1.txt", b"match")
    _write(tmp_path / "dup2.txt", b"match")
    _write(tmp_path / "solo.txt", b"other!")  # same size, different content
    dups = directory_duplicates(str(tmp_path))
    assert _name_groups(dups) == [["dup1.txt", "dup2.txt"]]


def test_search_is_recursive(tmp_path):
    _write(tmp_path / "top.bin", b"deep-content")
    _write(tmp_path / "level1" / "level2" / "deep.bin", b"deep-content")
    dups = directory_duplicates(str(tmp_path))
    assert _name_groups(dups) == [["deep.bin", "top.bin"]]


def test_empty_files_are_duplicates_of_each_other(tmp_path):
    _write(tmp_path / "empty1", b"")
    _write(tmp_path / "empty2", b"")
    dups = directory_duplicates(str(tmp_path))
    assert _name_groups(dups) == [["empty1", "empty2"]]


def test_files_differing_only_after_first_bytes_are_not_duplicates(tmp_path):
    prefix = b"A" * 100
    _write(tmp_path / "f1", prefix + b"x")
    _write(tmp_path / "f2", prefix + b"y")
    assert directory_duplicates(str(tmp_path)) == []


def test_sha256_hash_type_gives_same_groups(tmp_path):
    _write(tmp_path / "a.txt", b"hash me")
    _write(tmp_path / "b.txt", b"hash me")
    _write(tmp_path / "c.txt", b"other!!")
    md5_result = directory_duplicates(str(tmp_path), hash_type="md5")
    sha_result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert _name_groups(sha_result) == _name_groups(md5_result)
    assert _name_groups(sha_result) == [["a.txt", "b.txt"]]


def test_default_hash_type_is_md5(tmp_path, monkeypatch):
    _write(tmp_path / "a.txt", b"spy")
    _write(tmp_path / "b.txt", b"spy")
    original = file_operations.file_hash
    used_types = []

    def spy(path, hash_type="md5", **kwargs):
        used_types.append(hash_type)
        return original(path, hash_type=hash_type, **kwargs)

    monkeypatch.setattr(file_operations, "file_hash", spy)
    directory_duplicates(str(tmp_path))
    assert used_types and all(t == "md5" for t in used_types)


def test_invalid_hash_type_raises_value_error(tmp_path):
    _write(tmp_path / "a.txt", b"dupe")
    _write(tmp_path / "b.txt", b"dupe")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not-a-real-hash")


def test_invalid_hash_type_unused_when_no_size_candidates(tmp_path):
    # No two files share a size, so nothing is hashed and no error occurs.
    _write(tmp_path / "a.txt", b"1")
    _write(tmp_path / "b.txt", b"12")
    assert directory_duplicates(str(tmp_path), hash_type="not-a-real-hash") == []


def test_nonexistent_directory_returns_empty_list(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    assert directory_duplicates(missing) == []


def test_ext_kwarg_restricts_candidate_files(tmp_path):
    _write(tmp_path / "a.txt", b"same")
    _write(tmp_path / "b.txt", b"same")
    _write(tmp_path / "c.bin", b"same")
    dups = directory_duplicates(str(tmp_path), ext=".txt")
    assert _name_groups(dups) == [["a.txt", "b.txt"]]


def test_name_kwarg_restricts_candidate_files(tmp_path):
    _write(tmp_path / "report1.txt", b"same")
    _write(tmp_path / "report2.txt", b"same")
    _write(tmp_path / "other.txt", b"same")
    dups = directory_duplicates(str(tmp_path), name="report")
    assert _name_groups(dups) == [["report1.txt", "report2.txt"]]


def test_abspath_kwarg_returns_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "a.txt", b"abs")
    _write(tmp_path / "b.txt", b"abs")
    dups = directory_duplicates(".", abspath=True)
    assert len(dups) == 1
    assert all(os.path.isabs(p) for group in dups for p in group)


def test_each_file_appears_in_at_most_one_group(tmp_path):
    _write(tmp_path / "x1", b"aaaa")
    _write(tmp_path / "x2", b"aaaa")
    _write(tmp_path / "y1", b"bbbb")
    _write(tmp_path / "y2", b"bbbb")
    _write(tmp_path / "z1", b"cccc")
    dups = directory_duplicates(str(tmp_path))
    all_paths = [p for group in dups for p in group]
    assert len(all_paths) == len(set(all_paths)) == 4


def test_grouped_files_have_identical_contents(tmp_path):
    _write(tmp_path / "a", b"content-A")
    _write(tmp_path / "b", b"content-A")
    _write(tmp_path / "c", b"content-B")
    _write(tmp_path / "d", b"content-B")
    for group in directory_duplicates(str(tmp_path)):
        contents = set()
        for path in group:
            with open(path, "rb") as f:
                contents.add(f.read())
        assert len(contents) == 1


def test_files_are_not_modified_or_removed(tmp_path):
    a = _write(tmp_path / "a.txt", b"keep me")
    b = _write(tmp_path / "b.txt", b"keep me")
    directory_duplicates(str(tmp_path))
    assert os.path.exists(a) and os.path.exists(b)
    with open(a, "rb") as f:
        assert f.read() == b"keep me"


def test_same_name_in_different_directories_are_duplicates(tmp_path):
    _write(tmp_path / "a" / "x.dat", b"payload")
    _write(tmp_path / "b" / "x.dat", b"payload")
    dups = directory_duplicates(str(tmp_path))
    assert len(dups) == 1
    assert _names(dups[0]) == ["x.dat", "x.dat"]
    assert dups[0][0] != dups[0][1]


def test_return_type_is_list_of_lists_of_strings(tmp_path):
    _write(tmp_path / "a.txt", b"type")
    _write(tmp_path / "b.txt", b"type")
    dups = directory_duplicates(str(tmp_path))
    assert isinstance(dups, list)
    for group in dups:
        assert isinstance(group, list)
        assert len(group) > 1
        assert all(isinstance(p, str) for p in group)
