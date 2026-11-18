import os
from pathlib import Path

import pytest

import reusables.file_operations as fo


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def test_directory_duplicates_finds_duplicates_by_content(tmp_path):
    root = tmp_path / "root"
    a = _write(root / "a.txt", b"same-content")
    b = _write(root / "nested" / "b.txt", b"same-content")
    _write(root / "c.txt", b"different")

    result = fo.directory_duplicates(str(root))

    assert len(result) == 1
    assert sorted(result[0]) == sorted([a, b])


def test_directory_duplicates_ignores_unique_files(tmp_path):
    root = tmp_path / "root"
    _write(root / "a.txt", b"a")
    _write(root / "b.txt", b"bb")
    _write(root / "c.txt", b"ccc")

    assert fo.directory_duplicates(str(root)) == []


def test_directory_duplicates_groups_duplicates_across_subdirectories(tmp_path):
    root = tmp_path / "root"
    f1 = _write(root / "one" / "same.bin", b"123456")
    f2 = _write(root / "two" / "same.bin", b"123456")
    _write(root / "two" / "other.bin", b"654321")

    result = fo.directory_duplicates(str(root))

    assert len(result) == 1
    assert sorted(result[0]) == sorted([f1, f2])


def test_directory_duplicates_uses_hash_type_argument(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write(root / "a.txt", b"same")
    _write(root / "b.txt", b"same")

    seen = []

    def fake_hash(path, hash_type="md5"):
        seen.append((os.path.basename(path), hash_type))
        return "hash-" + hash_type

    monkeypatch.setattr(fo, "file_hash", fake_hash)

    result = fo.directory_duplicates(str(root), hash_type="sha256")

    assert len(result) == 1
    assert sorted(result[0]) == sorted(
        [str(root / "a.txt"), str(root / "b.txt")]
    )
    assert seen == [("a.txt", "sha256"), ("b.txt", "sha256")]


def test_directory_duplicates_passes_kwargs_to_find_files(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _write(root / "a.txt", b"same")
    _write(root / "b.log", b"same")
    _write(root / "c.txt", b"different")

    called = {}

    def fake_find_files(directory, **kwargs):
        called["directory"] = directory
        called["kwargs"] = kwargs
        return iter([str(root / "a.txt"), str(root / "c.txt")])

    monkeypatch.setattr(fo, "find_files", fake_find_files)

    result = fo.directory_duplicates(str(root), name="*.txt", abspath=True)

    assert called["directory"] == str(root)
    assert called["kwargs"]["disable_pathlib"] is True
    assert called["kwargs"]["name"] == "*.txt"
    assert called["kwargs"]["abspath"] is True
    assert result == []


def test_directory_duplicates_returns_empty_when_no_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    assert fo.directory_duplicates(str(root)) == []


def test_directory_duplicates_returns_empty_for_single_file(tmp_path):
    root = tmp_path / "root"
    _write(root / "only.txt", b"solo")

    assert fo.directory_duplicates(str(root)) == []


def test_directory_duplicates_allows_ext_filtering(tmp_path):
    root = tmp_path / "root"
    f1 = _write(root / "a.txt", b"same")
    f2 = _write(root / "b.txt", b"same")
    _write(root / "c.log", b"same")

    result = fo.directory_duplicates(str(root), ext=".txt")

    assert len(result) == 1
    assert sorted(result[0]) == sorted([f1, f2])


def test_directory_duplicates_respects_name_filtering(tmp_path):
    root = tmp_path / "root"
    f1 = _write(root / "keep_one.txt", b"same")
    f2 = _write(root / "nested" / "keep_two.txt", b"same")
    _write(root / "skip.txt", b"same")

    result = fo.directory_duplicates(str(root), name="keep")

    assert len(result) == 1
    assert sorted(result[0]) == sorted([f1, f2])


def test_directory_duplicates_returns_list_of_lists(tmp_path):
    root = tmp_path / "root"
    _write(root / "a.txt", b"same-1")
    _write(root / "b.txt", b"same-1")
    _write(root / "c.txt", b"same-2")
    _write(root / "d.txt", b"same-2")

    result = fo.directory_duplicates(str(root))

    assert isinstance(result, list)
    assert all(isinstance(group, list) for group in result)
    assert len(result) == 2


def test_directory_duplicates_does_not_group_same_size_different_content(tmp_path):
    root = tmp_path / "root"
    _write(root / "a.txt", b"abcd")
    _write(root / "b.txt", b"wxyz")
    _write(root / "c.txt", b"1234")

    assert fo.directory_duplicates(str(root)) == []


def test_directory_duplicates_with_nested_structure_and_multiple_groups(tmp_path):
    root = tmp_path / "root"
    g1a = _write(root / "g1" / "a.bin", b"group-one")
    g1b = _write(root / "g2" / "b.bin", b"group-one")
    g2a = _write(root / "g3" / "c.bin", b"group-two")
    g2b = _write(root / "g4" / "d.bin", b"group-two")
    _write(root / "unique.bin", b"unique")

    result = fo.directory_duplicates(str(root))

    assert len(result) == 2
    normalized = [sorted(group) for group in result]
    assert sorted(normalized) == sorted([sorted([g1a, g1b]), sorted([g2a, g2b])])


def test_directory_duplicates_same_name_different_content_not_grouped(tmp_path):
    root = tmp_path / "root"
    _write(root / "a" / "dup.txt", b"content-1")
    _write(root / "b" / "dup.txt", b"content-2")

    assert fo.directory_duplicates(str(root)) == []
