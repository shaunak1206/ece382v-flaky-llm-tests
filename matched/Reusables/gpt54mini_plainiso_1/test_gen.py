import os
from pathlib import Path

import pytest

import reusables.file_operations as fo


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def test_directory_duplicates_returns_empty_list_for_empty_directory(tmp_path):
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_returns_empty_list_when_all_files_are_unique(tmp_path):
    _write(tmp_path / "a.txt", b"aaa")
    _write(tmp_path / "b.txt", b"bbb")
    _write(tmp_path / "c.txt", b"cccc")
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_finds_simple_duplicates(tmp_path):
    f1 = _write(tmp_path / "one.txt", b"duplicate content")
    f2 = _write(tmp_path / "two.txt", b"duplicate content")
    _write(tmp_path / "three.txt", b"different content")

    result = fo.directory_duplicates(str(tmp_path))

    assert result == [[f1, f2]]


def test_directory_duplicates_groups_duplicates_across_subdirectories(tmp_path):
    f1 = _write(tmp_path / "root.txt", b"same")
    f2 = _write(tmp_path / "sub" / "nested.txt", b"same")
    _write(tmp_path / "other.txt", b"unique")

    result = fo.directory_duplicates(str(tmp_path))

    assert result == [[f1, f2]]


def test_directory_duplicates_ignores_files_with_same_size_but_different_content(tmp_path):
    _write(tmp_path / "one.txt", b"abcd")
    _write(tmp_path / "two.txt", b"wxyz")
    _write(tmp_path / "three.txt", b"1234")

    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_returns_multiple_duplicate_groups(tmp_path):
    a1 = _write(tmp_path / "a1.txt", b"group-a")
    a2 = _write(tmp_path / "dir" / "a2.txt", b"group-a")
    b1 = _write(tmp_path / "b1.txt", b"group-b!!")
    b2 = _write(tmp_path / "dir2" / "b2.txt", b"group-b!!")
    _write(tmp_path / "unique.txt", b"unique")

    result = fo.directory_duplicates(str(tmp_path))

    assert result == [[a1, a2], [b1, b2]]


def test_directory_duplicates_passes_kwargs_to_find_files(tmp_path, monkeypatch):
    captured = {}

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        captured["directory"] = directory
        captured["disable_pathlib"] = disable_pathlib
        captured["kwargs"] = kwargs
        return iter([])

    monkeypatch.setattr(fo, "find_files", fake_find_files)

    result = fo.directory_duplicates(str(tmp_path), ext=".txt", depth=2, match_case=True)

    assert result == []
    assert captured["directory"] == str(tmp_path)
    assert captured["disable_pathlib"] is True
    assert captured["kwargs"] == {"ext": ".txt", "depth": 2, "match_case": True}


def test_directory_duplicates_uses_specified_hash_type(tmp_path, monkeypatch):
    files = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    for path in files:
        _write(Path(path), b"same-content")

    seen = []

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        return iter(files)

    def fake_file_hash(path, hash_type="md5", **kwargs):
        seen.append((path, hash_type))
        return "hash-value"

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo, "file_hash", fake_file_hash)

    result = fo.directory_duplicates(str(tmp_path), hash_type="sha256")

    assert result == [files]
    assert seen == [(files[0], "sha256"), (files[1], "sha256")]


def test_directory_duplicates_only_hashes_items_with_duplicate_sizes(tmp_path, monkeypatch):
    files = [
        str(tmp_path / "a.txt"),
        str(tmp_path / "b.txt"),
        str(tmp_path / "c.txt"),
    ]
    _write(Path(files[0]), b"same-size")
    _write(Path(files[1]), b"same-size")
    _write(Path(files[2]), b"different")

    hashed = []

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        return iter(files)

    def fake_getsize(path):
        return 9 if path != files[2] else 99

    def fake_file_hash(path, hash_type="md5", **kwargs):
        hashed.append(path)
        return "shared" if path in files[:2] else "unique"

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo.os.path, "getsize", fake_getsize)
    monkeypatch.setattr(fo, "file_hash", fake_file_hash)

    result = fo.directory_duplicates(str(tmp_path))

    assert result == [files[:2]]
    assert hashed == files[:2]


def test_directory_duplicates_excludes_singletons_from_hash_groups(tmp_path, monkeypatch):
    files = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt"), str(tmp_path / "c.txt")]

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        return iter(files)

    size_map = {files[0]: 1, files[1]: 1, files[2]: 2}
    hash_map = {files[0]: "X", files[1]: "Y"}

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo.os.path, "getsize", lambda p: size_map[p])
    monkeypatch.setattr(fo, "file_hash", lambda p, hash_type="md5", **kwargs: hash_map[p])

    result = fo.directory_duplicates(str(tmp_path))

    assert result == []


def test_directory_duplicates_returns_empty_list_when_hashes_do_not_match(tmp_path, monkeypatch):
    files = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        return iter(files)

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo.os.path, "getsize", lambda p: 10)
    monkeypatch.setattr(fo, "file_hash", lambda p, hash_type="md5", **kwargs: p)

    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_preserves_order_within_duplicate_group(tmp_path, monkeypatch):
    files = [str(tmp_path / "first.txt"), str(tmp_path / "second.txt"), str(tmp_path / "third.txt")]

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        return iter(files)

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo.os.path, "getsize", lambda p: 5)
    monkeypatch.setattr(fo, "file_hash", lambda p, hash_type="md5", **kwargs: "same" if p != files[2] else "different")

    result = fo.directory_duplicates(str(tmp_path))

    assert result == [[files[0], files[1]]]


def test_directory_duplicates_with_filename_filter_can_limit_results(tmp_path):
    txt1 = _write(tmp_path / "a.txt", b"same")
    txt2 = _write(tmp_path / "sub" / "b.txt", b"same")
    _write(tmp_path / "a.bin", b"same")
    _write(tmp_path / "sub" / "b.bin", b"same")

    result = fo.directory_duplicates(str(tmp_path), ext=".txt")

    assert result == [[txt1, txt2]]


def test_directory_duplicates_can_use_custom_directory_argument(tmp_path, monkeypatch):
    subdir = tmp_path / "target"
    subdir.mkdir()
    files = [str(subdir / "a.txt"), str(subdir / "b.txt")]

    def fake_find_files(directory, disable_pathlib=True, **kwargs):
        assert directory == str(subdir)
        return iter(files)

    monkeypatch.setattr(fo, "find_files", fake_find_files)
    monkeypatch.setattr(fo.os.path, "getsize", lambda p: 8)
    monkeypatch.setattr(fo, "file_hash", lambda p, hash_type="md5", **kwargs: "duplicate")

    assert fo.directory_duplicates(str(subdir)) == [files]
