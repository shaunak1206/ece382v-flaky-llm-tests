import os
from pathlib import Path

import pytest

import reusables.file_operations as fo


def _write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sorted_grouped(result):
    return sorted(sorted(group) for group in result)


def test_directory_duplicates_empty_directory_returns_empty_list(tmp_path):
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_single_file_returns_empty_list(tmp_path):
    _write_file(tmp_path / "a.txt", b"hello")
    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_detects_duplicate_files_default_md5(tmp_path):
    _write_file(tmp_path / "a.txt", b"same content")
    _write_file(tmp_path / "sub" / "b.txt", b"same content")

    result = fo.directory_duplicates(str(tmp_path))

    assert _sorted_grouped(result) == [[
        os.path.abspath(str(tmp_path / "a.txt")),
        os.path.abspath(str(tmp_path / "sub" / "b.txt")),
    ]]


def test_directory_duplicates_ignores_files_with_same_size_but_different_content(tmp_path):
    _write_file(tmp_path / "a.txt", b"abcd")
    _write_file(tmp_path / "b.txt", b"wxyz")

    assert fo.directory_duplicates(str(tmp_path)) == []


def test_directory_duplicates_finds_multiple_duplicate_groups(tmp_path):
    _write_file(tmp_path / "group1" / "a1.txt", b"aaa")
    _write_file(tmp_path / "group1" / "a2.txt", b"aaa")
    _write_file(tmp_path / "group2" / "b1.txt", b"bbb")
    _write_file(tmp_path / "group2" / "b2.txt", b"bbb")
    _write_file(tmp_path / "unique.txt", b"unique")

    result = fo.directory_duplicates(str(tmp_path))

    expected = [
        sorted([os.path.abspath(str(tmp_path / "group1" / "a1.txt")),
                os.path.abspath(str(tmp_path / "group1" / "a2.txt"))]),
        sorted([os.path.abspath(str(tmp_path / "group2" / "b1.txt")),
                os.path.abspath(str(tmp_path / "group2" / "b2.txt"))]),
    ]
    assert _sorted_grouped(result) == sorted(expected)


def test_directory_duplicates_same_size_different_hash_not_grouped(tmp_path):
    _write_file(tmp_path / "a.txt", b"1234")
    _write_file(tmp_path / "b.txt", b"5678")
    _write_file(tmp_path / "c.txt", b"1234")  # duplicate of a.txt

    result = fo.directory_duplicates(str(tmp_path))

    expected = [[
        os.path.abspath(str(tmp_path / "a.txt")),
        os.path.abspath(str(tmp_path / "c.txt")),
    ]]
    assert _sorted_grouped(result) == _sorted_grouped(expected)


def test_directory_duplicates_respects_find_files_kwargs_ext_filter(tmp_path):
    _write_file(tmp_path / "a.txt", b"same")
    _write_file(tmp_path / "b.txt", b"same")
    _write_file(tmp_path / "c.md", b"same")
    _write_file(tmp_path / "d.md", b"same")

    result = fo.directory_duplicates(str(tmp_path), ext=".txt")

    expected = [[
        os.path.abspath(str(tmp_path / "a.txt")),
        os.path.abspath(str(tmp_path / "b.txt")),
    ]]
    assert _sorted_grouped(result) == _sorted_grouped(expected)


def test_directory_duplicates_respects_find_files_kwargs_name_filter(tmp_path):
    _write_file(tmp_path / "keep_1.txt", b"dup")
    _write_file(tmp_path / "keep_2.txt", b"dup")
    _write_file(tmp_path / "skip_1.txt", b"dup")
    _write_file(tmp_path / "skip_2.txt", b"dup")

    result = fo.directory_duplicates(str(tmp_path), name="keep")

    expected = [[
        os.path.abspath(str(tmp_path / "keep_1.txt")),
        os.path.abspath(str(tmp_path / "keep_2.txt")),
    ]]
    assert _sorted_grouped(result) == _sorted_grouped(expected)


def test_directory_duplicates_passes_hash_type_to_file_hash(tmp_path, monkeypatch):
    _write_file(tmp_path / "a.txt", b"abc")
    _write_file(tmp_path / "b.txt", b"abc")

    seen_hash_types = []

    def fake_file_hash(path, hash_type="md5"):
        seen_hash_types.append(hash_type)
        return "same-hash"

    monkeypatch.setattr(fo, "file_hash", fake_file_hash)

    result = fo.directory_duplicates(str(tmp_path), hash_type="sha256")

    assert _sorted_grouped(result) == [[
        os.path.abspath(str(tmp_path / "a.txt")),
        os.path.abspath(str(tmp_path / "b.txt")),
    ]]
    assert seen_hash_types == ["sha256", "sha256"]


def test_directory_duplicates_uses_disable_pathlib_true_in_find_files(tmp_path, monkeypatch):
    _write_file(tmp_path / "a.txt", b"same")
    _write_file(tmp_path / "b.txt", b"same")

    calls = []

    def fake_find_files(directory, **kwargs):
        calls.append((directory, kwargs))
        yield str(tmp_path / "a.txt")
        yield str(tmp_path / "b.txt")

    monkeypatch.setattr(fo, "find_files", fake_find_files)

    result = fo.directory_duplicates(str(tmp_path))

    assert len(calls) == 1
    assert calls[0][0] == str(tmp_path)
    assert calls[0][1]["disable_pathlib"] is True
    assert result == [[str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]]


def test_directory_duplicates_works_with_relative_directory_argument(tmp_path, monkeypatch):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _write_file(Path("x") / "a.txt", b"same")
        _write_file(Path("x") / "b.txt", b"same")

        result = fo.directory_duplicates("x")
    finally:
        os.chdir(cwd)

    expected = [[
        os.path.abspath(str(tmp_path / "x" / "a.txt")),
        os.path.abspath(str(tmp_path / "x" / "b.txt")),
    ]]
    assert _sorted_grouped(result) == _sorted_grouped(expected)


def test_directory_duplicates_preserves_duplicate_files_across_nested_directories(tmp_path):
    _write_file(tmp_path / "dir1" / "a.txt", b"content")
    _write_file(tmp_path / "dir2" / "nested" / "b.txt", b"content")

    result = fo.directory_duplicates(str(tmp_path))

    assert _sorted_grouped(result) == [[
        os.path.abspath(str(tmp_path / "dir1" / "a.txt")),
        os.path.abspath(str(tmp_path / "dir2" / "nested" / "b.txt")),
    ]]
