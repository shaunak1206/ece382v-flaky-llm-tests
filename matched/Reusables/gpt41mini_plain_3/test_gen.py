# test_file_operations_directory_duplicates.py

import os
import tempfile
import shutil
import hashlib
import pytest
from reusables.file_operations import directory_duplicates


def write_file(path, content=b""):
    with open(path, "wb") as f:
        f.write(content)


def compute_md5(content: bytes):
    hasher = hashlib.md5()
    hasher.update(content)
    return hasher.hexdigest()


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_directory_duplicates_no_files(tmp_dir):
    # empty directory returns empty list
    dups = directory_duplicates(tmp_dir)
    assert dups == []


def test_directory_duplicates_single_file(tmp_dir):
    path = os.path.join(tmp_dir, "file1.txt")
    write_file(path, b"Hello world")
    dups = directory_duplicates(tmp_dir)
    # Only one file, no duplicates
    assert dups == []


def test_directory_duplicates_two_identical_files(tmp_dir):
    content = b"Duplicate content"
    path1 = os.path.join(tmp_dir, "file1.txt")
    path2 = os.path.join(tmp_dir, "file2.txt")
    write_file(path1, content)
    write_file(path2, content)

    dups = directory_duplicates(tmp_dir)
    # There should be one group of duplicates with both files
    assert isinstance(dups, list)
    assert any(set(group) == {path1, path2} for group in dups)
    # Each group must have at least 2 files
    for group in dups:
        assert len(group) > 1


def test_directory_duplicates_multiple_groups(tmp_dir):
    content_a = b"abc"
    content_b = b"xyz"
    # group 1
    p1a = os.path.join(tmp_dir, "a1.txt")
    p1b = os.path.join(tmp_dir, "a2.txt")
    # group 2
    p2a = os.path.join(tmp_dir, "b1.txt")
    p2b = os.path.join(tmp_dir, "b2.txt")
    # unique file
    p3 = os.path.join(tmp_dir, "unique.txt")

    write_file(p1a, content_a)
    write_file(p1b, content_a)
    write_file(p2a, content_b)
    write_file(p2b, content_b)
    write_file(p3, b"no duplicates here")

    dups = directory_duplicates(tmp_dir)

    # Expect exactly two groups, each with 2 files
    assert len(dups) == 2

    flattened = [set(g) for g in dups]
    expected_groups = [{p1a, p1b}, {p2a, p2b}]
    for expected_group in expected_groups:
        assert any(exp_group == expected_group for exp_group in flattened)

    # unique file should not be in duplicates
    for group in dups:
        assert p3 not in group


def test_directory_duplicates_different_sizes_no_duplicates(tmp_dir):
    # Different sizes should never be duplicates
    p1 = os.path.join(tmp_dir, "file1.txt")
    p2 = os.path.join(tmp_dir, "file2.txt")
    write_file(p1, b"x" * 10)
    write_file(p2, b"x" * 11)
    dups = directory_duplicates(tmp_dir)
    assert dups == []


def test_directory_duplicates_empty_files(tmp_dir):
    # Empty files have 0 size; they could be detected as duplicates
    p1 = os.path.join(tmp_dir, "empty1.txt")
    p2 = os.path.join(tmp_dir, "empty2.txt")
    # create two empty files
    write_file(p1, b"")
    write_file(p2, b"")
    dups = directory_duplicates(tmp_dir)
    # Both should appear as duplicates together
    assert any(set(group) == {p1, p2} for group in dups)


def test_directory_duplicates_with_filter_extension(tmp_dir):
    # Only files with .txt extension should be considered
    content = b"same text"
    txt1 = os.path.join(tmp_dir, "foo.txt")
    txt2 = os.path.join(tmp_dir, "bar.txt")
    py1 = os.path.join(tmp_dir, "code.py")
    write_file(txt1, content)
    write_file(txt2, content)
    write_file(py1, content)

    dups = directory_duplicates(tmp_dir, ext=".txt")

    # should not consider py files
    assert any(set(group) == {txt1, txt2} for group in dups)
    # py1 should not be part of duplicates as filtered by ext
    for group in dups:
        assert py1 not in group


def test_directory_duplicates_with_name_filter(tmp_dir):
    # Only files with a name containing "match" will be considered
    content = b"contents"
    match1 = os.path.join(tmp_dir, "match1.txt")
    match2 = os.path.join(tmp_dir, "match2.txt")
    nomatch = os.path.join(tmp_dir, "other.txt")
    write_file(match1, content)
    write_file(match2, content)
    write_file(nomatch, content)

    dups = directory_duplicates(tmp_dir, name="match")

    assert any(set(group) == {match1, match2} for group in dups)
    # nomatch file should not be in the result
    for group in dups:
        assert nomatch not in group


def test_directory_duplicates_with_depth(tmp_dir):
    # Test that nested files are handled correctly with depth option passed
    os.makedirs(os.path.join(tmp_dir, "level1", "level2"))
    p_root = os.path.join(tmp_dir, "file.txt")
    p_level1 = os.path.join(tmp_dir, "level1", "file.txt")
    p_level2 = os.path.join(tmp_dir, "level1", "level2", "file.txt")

    content = b"hello"
    write_file(p_root, content)
    write_file(p_level1, content)
    write_file(p_level2, content)

    # depth 0 means only files in root directory
    dups_depth0 = directory_duplicates(tmp_dir, depth=0)
    assert dups_depth0 == []

    # depth 1 includes level1 files; root and level1 are duplicates
    dups_depth1 = directory_duplicates(tmp_dir, depth=1)
    # Should include the files at root and level1 but not level2
    sets = [set(files) for files in dups_depth1]
    expected = {p_root, p_level1}
    # There should be a duplicate group containing root and level1 files
    assert any(expected.issubset(s) for s in sets)
    # level2 not included (due to depth=1)


def test_directory_duplicates_hash_type_sha1(tmp_dir):
    # Test passing a different hash_type (sha1)
    content = b"sha1 hash test"
    p1 = os.path.join(tmp_dir, "file1.txt")
    p2 = os.path.join(tmp_dir, "file2.txt")
    write_file(p1, content)
    write_file(p2, content)

    dups = directory_duplicates(tmp_dir, hash_type="sha1")

    assert any(set(group) == {p1, p2} for group in dups)


def test_directory_duplicates_with_nonexistent_directory():
    # Should raise OSError when directory does not exist
    with pytest.raises(OSError):
        directory_duplicates("/path/does/not/exist")


def test_directory_duplicates_symlink_files(tmp_dir):
    # Create two identical files, and a symlink to one of them
    content = b"symlink contents"
    target = os.path.join(tmp_dir, "target.txt")
    link = os.path.join(tmp_dir, "link.txt")
    write_file(target, content)

    if hasattr(os, "symlink"):
        os.symlink(target, link)
        dups = directory_duplicates(tmp_dir)
        # The symlink should be detected as duplicate of the target
        # The returned list includes both target and link
        found = False
        for group in dups:
            if target in group and link in group:
                found = True
        assert found


def test_directory_duplicates_ignores_directories(tmp_dir):
    # Create a directory with the same name as a file and a file with some content
    os.makedirs(os.path.join(tmp_dir, "file_dir"))
    file_path = os.path.join(tmp_dir, "file_dir")
    # Since the directory has same name as file_path, but no file exists, no dup found
    dups = directory_duplicates(tmp_dir)
    # No duplicates because no actual files
    assert dups == []


def test_directory_duplicates_handles_large_file(tmp_dir):
    # Create two large files with identical contents
    large_content = b"A" * 1024 * 1024  # 1 MB of 'A's
    f1 = os.path.join(tmp_dir, "large1.bin")
    f2 = os.path.join(tmp_dir, "large2.bin")
    write_file(f1, large_content)
    write_file(f2, large_content)

    dups = directory_duplicates(tmp_dir)
    # The two large files should be identified as duplicates
    assert any(set(group) == {f1, f2} for group in dups)


def test_directory_duplicates_mixed_filetypes(tmp_dir):
    # Create files with same content but different extension
    content = b"Same content"
    f1 = os.path.join(tmp_dir, "a.txt")
    f2 = os.path.join(tmp_dir, "b.md")
    write_file(f1, content)
    write_file(f2, content)

    # Without filter, duplicates found
    dups_all = directory_duplicates(tmp_dir)
    assert any(set(group) == {f1, f2} for group in dups_all)

    # With ext filter matching only '.txt', duplicates should be empty
    dups_txt = directory_duplicates(tmp_dir, ext=".txt")
    assert dups_txt == []


def test_directory_duplicates_unicode_filenames(tmp_dir):
    # Test with filenames containing unicode characters
    content = b"unicode content"
    u1 = os.path.join(tmp_dir, "unićode1.txt")
    u2 = os.path.join(tmp_dir, "unićode2.txt")
    write_file(u1, content)
    write_file(u2, content)
    dups = directory_duplicates(tmp_dir)
    assert any(set(group) == {u1, u2} for group in dups)


def test_directory_duplicates_ignores_broken_symlink(tmp_dir):
    # Create a broken symlink (should be ignored safely)
    if hasattr(os, "symlink"):
        broken_link = os.path.join(tmp_dir, "broken_symlink")
        target = os.path.join(tmp_dir, "nonexistent.file")
        os.symlink(target, broken_link)
        # Create one real file that will not have duplicates
        real_file = os.path.join(tmp_dir, "real.txt")
        write_file(real_file, b"content")

        dups = directory_duplicates(tmp_dir)
        # Because broken symlink points nowhere, no duplicates
        # The real file alone means no dups found
        assert dups == []


def test_directory_duplicates_raises_type_error_for_bad_ext_arg(tmp_dir):
    # ext argument must be string or list/tuple; passing int should raise TypeError from find_files
    with pytest.raises(TypeError):
        directory_duplicates(tmp_dir, ext=123)


def test_directory_duplicates_returns_lists_of_lists(tmp_dir):
    # Confirm output type is list of lists of strings
    content = b"dup list test"
    f1 = os.path.join(tmp_dir, "filea.txt")
    f2 = os.path.join(tmp_dir, "fileb.txt")
    write_file(f1, content)
    write_file(f2, content)
    result = directory_duplicates(tmp_dir)
    assert isinstance(result, list)
    for group in result:
        assert isinstance(group, list)
        for file_path in group:
            assert isinstance(file_path, str)
            assert os.path.exists(file_path)
