import hashlib
import os

import pytest

import reusables.file_operations as fo


def write(path, content):
    """Create parent directories and write bytes to path. Returns the path as str."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def basenames(groups):
    """Normalize a list of groups into a sorted structure of base file names."""
    return sorted(sorted(os.path.basename(p) for p in group) for group in groups)


def test_empty_directory_returns_empty_list(tmp_path):
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_nonexistent_directory_returns_empty_list(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert fo.directory_duplicates(str(missing)) == []


def test_single_file_has_no_duplicates(tmp_path):
    write(tmp_path / "only.txt", b"hello")
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_two_identical_files_are_duplicates(tmp_path):
    write(tmp_path / "a.txt", b"same content")
    write(tmp_path / "b.txt", b"same content")
    result = fo.directory_duplicates(str(tmp_path))
    assert basenames(result) == [["a.txt", "b.txt"]]


def test_same_size_different_content_not_duplicates(tmp_path):
    write(tmp_path / "a.txt", b"aaaa")
    write(tmp_path / "b.txt", b"bbbb")
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_unique_sizes_are_not_reported(tmp_path):
    write(tmp_path / "a.txt", b"abc")
    write(tmp_path / "b.txt", b"abcd")
    write(tmp_path / "c.txt", b"abcd")
    result = fo.directory_duplicates(str(tmp_path))
    assert basenames(result) == [["b.txt", "c.txt"]]


def test_three_identical_files_form_one_group(tmp_path):
    for name in ("one.bin", "two.bin", "three.bin"):
        write(tmp_path / name, b"triplicate")
    result = fo.directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert basenames(result) == [["one.bin", "three.bin", "two.bin"]]


def test_multiple_duplicate_groups_in_same_size_bucket(tmp_path):
    # Both pairs are 5 bytes, so they share a size bucket but differ in hash.
    write(tmp_path / "a1.txt", b"alpha")
    write(tmp_path / "a2.txt", b"alpha")
    write(tmp_path / "b1.txt", b"bravo")
    write(tmp_path / "b2.txt", b"bravo")
    result = fo.directory_duplicates(str(tmp_path))
    assert basenames(result) == [["a1.txt", "a2.txt"], ["b1.txt", "b2.txt"]]


def test_duplicates_across_subdirectories(tmp_path):
    write(tmp_path / "top.dat", b"shared data")
    write(tmp_path / "sub" / "deep" / "copy.dat", b"shared data")
    result = fo.directory_duplicates(str(tmp_path))
    assert basenames(result) == [["copy.dat", "top.dat"]]


def test_result_is_list_of_lists_with_at_least_two_items(tmp_path):
    write(tmp_path / "x.txt", b"xyz")
    write(tmp_path / "y.txt", b"xyz")
    write(tmp_path / "z.txt", b"unique!")
    result = fo.directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    for group in result:
        assert isinstance(group, list)
        assert len(group) > 1


def test_empty_files_are_duplicates_of_each_other(tmp_path):
    write(tmp_path / "empty1", b"")
    write(tmp_path / "empty2", b"")
    write(tmp_path / "not_empty", b"x")
    result = fo.directory_duplicates(str(tmp_path))
    assert basenames(result) == [["empty1", "empty2"]]


def test_returned_paths_exist_and_live_under_directory(tmp_path):
    write(tmp_path / "p.txt", b"path check")
    write(tmp_path / "nested" / "q.txt", b"path check")
    result = fo.directory_duplicates(str(tmp_path))
    assert result
    for group in result:
        for path in group:
            assert os.path.exists(path)
            assert path.startswith(str(tmp_path))


def test_hash_computed_only_for_same_size_candidates(tmp_path, monkeypatch):
    write(tmp_path / "a.txt", b"xx")
    write(tmp_path / "b.txt", b"xx")
    write(tmp_path / "c.txt", b"yyy")
    write(tmp_path / "d.txt", b"zzzz")

    hashed = []
    original = fo.file_hash

    def spy(path, hash_type="md5", **kwargs):
        hashed.append(os.path.basename(path))
        return original(path, hash_type=hash_type, **kwargs)

    monkeypatch.setattr(fo, "file_hash", spy)
    result = fo.directory_duplicates(str(tmp_path))

    assert basenames(result) == [["a.txt", "b.txt"]]
    assert sorted(hashed) == ["a.txt", "b.txt"]


def test_default_hash_type_is_md5(tmp_path, monkeypatch):
    write(tmp_path / "a.txt", b"hash me")
    write(tmp_path / "b.txt", b"hash me")

    used = []
    original = fo.file_hash

    def spy(path, hash_type="md5", **kwargs):
        used.append(hash_type)
        return original(path, hash_type=hash_type, **kwargs)

    monkeypatch.setattr(fo, "file_hash", spy)
    result = fo.directory_duplicates(str(tmp_path))

    assert used and set(used) == {"md5"}
    assert basenames(result) == [["a.txt", "b.txt"]]


def test_sha256_hash_type_is_passed_through(tmp_path, monkeypatch):
    write(tmp_path / "a.txt", b"hash me too")
    write(tmp_path / "b.txt", b"hash me too")

    used = []
    original = fo.file_hash

    def spy(path, hash_type="md5", **kwargs):
        used.append(hash_type)
        return original(path, hash_type=hash_type, **kwargs)

    monkeypatch.setattr(fo, "file_hash", spy)
    result = fo.directory_duplicates(str(tmp_path), hash_type="sha256")

    assert used and set(used) == {"sha256"}
    assert basenames(result) == [["a.txt", "b.txt"]]


def test_invalid_hash_type_raises_when_candidates_exist(tmp_path):
    write(tmp_path / "a.txt", b"dup")
    write(tmp_path / "b.txt", b"dup")
    with pytest.raises(ValueError):
        fo.directory_duplicates(str(tmp_path), hash_type="not-a-real-hash")


def test_invalid_hash_type_unused_without_candidates(tmp_path):
    write(tmp_path / "lonely.txt", b"only one of me")
    assert fo.directory_duplicates(str(tmp_path), hash_type="not-a-real-hash") == []


def test_ext_kwarg_limits_duplicate_search(tmp_path):
    write(tmp_path / "a.txt", b"text dup")
    write(tmp_path / "b.txt", b"text dup")
    write(tmp_path / "c.log", b"log dup!")
    write(tmp_path / "d.log", b"log dup!")
    result = fo.directory_duplicates(str(tmp_path), ext=".txt")
    assert basenames(result) == [["a.txt", "b.txt"]]


def test_name_kwarg_limits_duplicate_search(tmp_path):
    write(tmp_path / "x_copy.dat", b"named dup")
    write(tmp_path / "y_copy.dat", b"named dup")
    write(tmp_path / "z.dat", b"named dup")
    result = fo.directory_duplicates(str(tmp_path), name="copy")
    assert basenames(result) == [["x_copy.dat", "y_copy.dat"]]


def test_depth_kwarg_excludes_nested_duplicates(tmp_path):
    write(tmp_path / "a.txt", b"depth test")
    write(tmp_path / "b.txt", b"depth test")
    write(tmp_path / "sub" / "c.txt", b"depth test")
    result = fo.directory_duplicates(str(tmp_path), depth=1)
    assert basenames(result) == [["a.txt", "b.txt"]]
