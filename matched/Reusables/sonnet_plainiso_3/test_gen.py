import os

import pytest

import reusables
from reusables import file_operations
from reusables.file_operations import directory_duplicates


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def normalize(groups):
    return sorted(sorted(g) for g in groups)


def test_exposed_at_package_level():
    assert reusables.directory_duplicates is directory_duplicates


def test_empty_directory_returns_empty_list(tmp_path):
    assert directory_duplicates(str(tmp_path)) == []


def test_nonexistent_directory_returns_empty_list(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    assert directory_duplicates(missing) == []


def test_single_file_is_not_duplicate(tmp_path):
    write(tmp_path / "only.txt", b"hello")
    assert directory_duplicates(str(tmp_path)) == []


def test_two_identical_files_found(tmp_path):
    a = write(tmp_path / "a.txt", b"same content")
    b = write(tmp_path / "b.txt", b"same content")
    result = directory_duplicates(str(tmp_path))
    assert normalize(result) == [sorted([a, b])]


def test_same_size_different_content_not_duplicates(tmp_path):
    write(tmp_path / "a.txt", b"abcdef")
    write(tmp_path / "b.txt", b"fedcba")
    assert directory_duplicates(str(tmp_path)) == []


def test_different_size_files_not_duplicates(tmp_path):
    write(tmp_path / "a.txt", b"abc")
    write(tmp_path / "b.txt", b"abcd")
    assert directory_duplicates(str(tmp_path)) == []


def test_duplicates_in_subdirectories(tmp_path):
    a = write(tmp_path / "a.txt", b"data data")
    b = write(tmp_path / "sub" / "deeper" / "b.txt", b"data data")
    result = directory_duplicates(str(tmp_path))
    assert normalize(result) == [sorted([a, b])]


def test_three_way_duplicate_grouped_together(tmp_path):
    paths = [
        write(tmp_path / "one.bin", b"xyz"),
        write(tmp_path / "two.bin", b"xyz"),
        write(tmp_path / "dir" / "three.bin", b"xyz"),
    ]
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert sorted(result[0]) == sorted(paths)


def test_multiple_distinct_groups(tmp_path):
    a1 = write(tmp_path / "a1", b"AAAA")
    a2 = write(tmp_path / "a2", b"AAAA")
    b1 = write(tmp_path / "b1", b"BBBB")
    b2 = write(tmp_path / "b2", b"BBBB")
    write(tmp_path / "unique", b"CCCC")
    result = directory_duplicates(str(tmp_path))
    assert normalize(result) == normalize([[a1, a2], [b1, b2]])


def test_empty_files_are_duplicates(tmp_path):
    a = write(tmp_path / "e1", b"")
    b = write(tmp_path / "e2", b"")
    result = directory_duplicates(str(tmp_path))
    assert normalize(result) == [sorted([a, b])]


def test_results_are_plain_strings(tmp_path):
    write(tmp_path / "a.txt", b"zzz")
    write(tmp_path / "b.txt", b"zzz")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    for group in result:
        assert isinstance(group, list)
        for item in group:
            assert isinstance(item, str)


def test_custom_hash_type_sha256(tmp_path):
    a = write(tmp_path / "a.txt", b"hash me")
    b = write(tmp_path / "b.txt", b"hash me")
    result = directory_duplicates(str(tmp_path), hash_type="sha256")
    assert normalize(result) == [sorted([a, b])]


def test_invalid_hash_type_raises_when_candidates_exist(tmp_path):
    write(tmp_path / "a.txt", b"same")
    write(tmp_path / "b.txt", b"same")
    with pytest.raises(ValueError):
        directory_duplicates(str(tmp_path), hash_type="not_a_real_hash")


def test_ext_kwarg_filters_files(tmp_path):
    t1 = write(tmp_path / "a.txt", b"text")
    t2 = write(tmp_path / "b.txt", b"text")
    write(tmp_path / "c.log", b"logs")
    write(tmp_path / "d.log", b"logs")
    result = directory_duplicates(str(tmp_path), ext=".txt")
    assert normalize(result) == [sorted([t1, t2])]


def test_name_kwarg_filters_files(tmp_path):
    c1 = write(tmp_path / "copy1.bin", b"1111")
    c2 = write(tmp_path / "copy2.bin", b"1111")
    write(tmp_path / "orig1.bin", b"2222")
    write(tmp_path / "orig2.bin", b"2222")
    result = directory_duplicates(str(tmp_path), name="copy")
    assert normalize(result) == [sorted([c1, c2])]


def test_depth_kwarg_limits_search(tmp_path):
    top = write(tmp_path / "top.txt", b"depth")
    write(tmp_path / "sub" / "nested.txt", b"depth")
    # Only the top-level directory is searched, so no duplicates are seen
    assert directory_duplicates(str(tmp_path), depth=1) == []
    # Without a depth limit both are found
    result = directory_duplicates(str(tmp_path))
    assert len(result) == 1
    assert top in result[0]


def test_invalid_ext_type_raises_type_error(tmp_path):
    write(tmp_path / "a.txt", b"x")
    with pytest.raises(TypeError):
        directory_duplicates(str(tmp_path), ext=5)


def test_hash_only_computed_for_same_size_candidates(tmp_path, monkeypatch):
    a = write(tmp_path / "a.txt", b"12345")
    b = write(tmp_path / "b.txt", b"12345")
    write(tmp_path / "unique.txt", b"a much longer unique file")

    hashed = []
    real_hash = file_operations.file_hash

    def spy(path, *args, **kwargs):
        hashed.append(path)
        return real_hash(path, *args, **kwargs)

    monkeypatch.setattr(file_operations, "file_hash", spy)
    result = directory_duplicates(str(tmp_path))

    assert normalize(result) == [sorted([a, b])]
    assert sorted(hashed) == sorted([a, b])
    assert os.path.join(str(tmp_path), "unique.txt") not in hashed
