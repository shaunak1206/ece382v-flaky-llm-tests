import os
from pathlib import Path

import pytest

from reusables import file_operations as fo


def _write_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_directory_duplicates_returns_empty_for_empty_directory(tmp_path):
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_returns_empty_for_directory_with_unique_files(tmp_path):
    _write_file(tmp_path / "a.txt", b"alpha")
    _write_file(tmp_path / "b.txt", b"bravo")
    _write_file(tmp_path / "c.txt", b"charlie")

    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_finds_duplicates_in_same_directory(tmp_path):
    _write_file(tmp_path / "a.txt", b"same content")
    _write_file(tmp_path / "b.txt", b"same content")
    _write_file(tmp_path / "c.txt", b"different")

    dups = fo.directory_duplicates(str(tmp_path))
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])


def test_directory_duplicates_finds_duplicates_across_subdirectories(tmp_path):
    _write_file(tmp_path / "one" / "a.txt", b"dup")
    _write_file(tmp_path / "two" / "b.txt", b"dup")
    _write_file(tmp_path / "three" / "c.txt", b"unique")

    dups = fo.directory_duplicates(str(tmp_path))
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "one" / "a.txt"), str(tmp_path / "two" / "b.txt")])


def test_directory_duplicates_ignores_same_size_but_different_content(tmp_path):
    _write_file(tmp_path / "a.txt", b"abcd")
    _write_file(tmp_path / "b.txt", b"wxyz")  # same size, different content
    _write_file(tmp_path / "c.txt", b"1234")  # same size, different content

    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_groups_more_than_two_duplicates(tmp_path):
    payload = b"repeated payload"
    _write_file(tmp_path / "a.txt", payload)
    _write_file(tmp_path / "b.txt", payload)
    _write_file(tmp_path / "nested" / "c.txt", payload)

    dups = fo.directory_duplicates(str(tmp_path))
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted(
        [str(tmp_path / "a.txt"), str(tmp_path / "b.txt"), str(tmp_path / "nested" / "c.txt")]
    )


def test_directory_duplicates_respects_find_files_kwargs_ext_filter(tmp_path):
    _write_file(tmp_path / "a.txt", b"same")
    _write_file(tmp_path / "b.txt", b"same")
    _write_file(tmp_path / "c.log", b"same")
    _write_file(tmp_path / "d.log", b"same")

    dups = fo.directory_duplicates(str(tmp_path), ext=".txt")
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])


def test_directory_duplicates_respects_find_files_kwargs_name_filter(tmp_path):
    _write_file(tmp_path / "match_1.txt", b"dup")
    _write_file(tmp_path / "match_2.txt", b"dup")
    _write_file(tmp_path / "other.txt", b"dup")

    dups = fo.directory_duplicates(str(tmp_path), name="match")
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "match_1.txt"), str(tmp_path / "match_2.txt")])


def test_directory_duplicates_forwards_hash_type_to_file_hash(tmp_path, monkeypatch):
    _write_file(tmp_path / "a.txt", b"same")
    _write_file(tmp_path / "b.txt", b"same")

    seen = []

    real_file_hash = fo.file_hash

    def fake_file_hash(path, hash_type="md5", *args, **kwargs):
        seen.append((os.path.basename(path), hash_type))
        return real_file_hash(path, hash_type=hash_type, *args, **kwargs)

    monkeypatch.setattr(fo, "file_hash", fake_file_hash)

    dups = fo.directory_duplicates(str(tmp_path), hash_type="sha1")
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert seen == [("a.txt", "sha1"), ("b.txt", "sha1")]


def test_directory_duplicates_passes_disable_pathlib_true_to_find_files(tmp_path, monkeypatch):
    _write_file(tmp_path / "a.txt", b"x")
    _write_file(tmp_path / "b.txt", b"x")

    captured = {}

    def fake_find_files(directory, **kwargs):
        captured["directory"] = directory
        captured["kwargs"] = kwargs
        return [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo, "file_hash", lambda path, hash_type="md5": "hash")

    dups = fo.directory_duplicates(str(tmp_path), ext=".txt")

    assert dups == [[str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]]
    assert captured["directory"] == str(tmp_path)
    assert captured["kwargs"]["disable_pathlib"] is True
    assert captured["kwargs"]["ext"] == ".txt"


def test_directory_duplicates_does_not_call_file_hash_for_single_candidate_size_bucket(tmp_path, monkeypatch):
    _write_file(tmp_path / "a.txt", b"same")
    _write_file(tmp_path / "b.txt", b"unique1")
    _write_file(tmp_path / "c.txt", b"unique2")

    call_count = 0

    def fake_file_hash(path, hash_type="md5"):
        nonlocal call_count
        call_count += 1
        return fo.file_hash(path, hash_type=hash_type)

    monkeypatch.setattr(fo, "file_hash", fake_file_hash)

    assert fo.directory_duplicates(str(tmp_path)) == []
    # Only the two same-sized files should be hashed; the unique sizes are skipped.
    assert call_count == 2


def test_directory_duplicates_returns_multiple_groups(tmp_path):
    _write_file(tmp_path / "group1_a.txt", b"aaa")
    _write_file(tmp_path / "group1_b.txt", b"aaa")
    _write_file(tmp_path / "group2_a.txt", b"bbb")
    _write_file(tmp_path / "group2_b.txt", b"bbb")
    _write_file(tmp_path / "unique.txt", b"ccc")

    dups = fo.directory_duplicates(str(tmp_path))
    assert len(dups) == 2

    sorted_groups = [sorted(group) for group in dups]
    assert sorted([str(tmp_path / "group1_a.txt"), str(tmp_path / "group1_b.txt")]) in sorted_groups
    assert sorted([str(tmp_path / "group2_a.txt"), str(tmp_path / "group2_b.txt")]) in sorted_groups


def test_directory_duplicates_can_handle_nonexistent_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert fo.directory_duplicates(str(missing)) == []


def test_directory_duplicates_handles_binary_files(tmp_path):
    _write_file(tmp_path / "a.bin", b"\x00\x01\x02\x03")
    _write_file(tmp_path / "b.bin", b"\x00\x01\x02\x03")
    _write_file(tmp_path / "c.bin", b"\x00\x01\x02\x04")

    dups = fo.directory_duplicates(str(tmp_path), ext=".bin")
    assert len(dups) == 1
    assert sorted(dups[0]) == sorted([str(tmp_path / "a.bin"), str(tmp_path / "b.bin")])
