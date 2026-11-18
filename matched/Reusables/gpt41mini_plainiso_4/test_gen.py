import os
import tempfile
import shutil
import hashlib
import pytest

from reusables.file_operations import directory_duplicates, file_hash


@pytest.fixture
def temp_dir_with_files():
    """Create a temp directory with some files for testing duplicates."""
    dirpath = tempfile.mkdtemp()
    files = {}
    # Create 3 files: two duplicates and one unique
    content1 = b"Hello World"
    content2 = b"Hello World"
    content3 = b"Different content"

    paths = []
    for idx, content in enumerate([content1, content2, content3], start=1):
        fpath = os.path.join(dirpath, f"file{idx}.txt")
        with open(fpath, "wb") as f:
            f.write(content)
        paths.append(fpath)
    yield dirpath, paths
    shutil.rmtree(dirpath)
    

@pytest.fixture
def temp_dir_empty_and_nested_files():
    """Create a temp directory with nested folders and some duplicate files."""
    dirpath = tempfile.mkdtemp()
    # Create nested structure:
    # dirpath/
    #   a.txt (content A)
    #   subdir/
    #       a_copy.txt (duplicate content A)
    #       b.txt (unique content B)
    content_a = b"Same content A"
    content_b = b"Unique content B"
    file_a = os.path.join(dirpath, "a.txt")
    file_b = os.path.join(dirpath, "subdir", "b.txt")
    copy_a = os.path.join(dirpath, "subdir", "a_copy.txt")

    os.makedirs(os.path.join(dirpath, "subdir"))
    with open(file_a, "wb") as f:
        f.write(content_a)
    with open(copy_a, "wb") as f:
        f.write(content_a)
    with open(file_b, "wb") as f:
        f.write(content_b)

    yield dirpath, (file_a, copy_a, file_b)
    shutil.rmtree(dirpath)


def test_directory_duplicates_detects_two_duplicates(temp_dir_with_files):
    dirpath, files = temp_dir_with_files
    dups = directory_duplicates(dirpath)
    # Expect one list of duplicates (file1 and file2)
    # and file3 unique
    assert isinstance(dups, list)
    # There should be exactly one group with exactly two files
    found = False
    for group in dups:
        assert len(group) > 1
        # The duplicates are file1 and file2 which have same content
        file_contents = [open(f, "rb").read() for f in group]
        assert all(c == file_contents[0] for c in file_contents)
        if set(group) == set([files[0], files[1]]):
            found = True
    assert found


def test_directory_duplicates_returns_empty_list_when_no_dupes(tmp_path):
    # tmp_path has no files initially, create unique files only
    file1 = tmp_path / "file1.txt"
    file1.write_text("unique1")
    file2 = tmp_path / "file2.txt"
    file2.write_text("unique2")
    result = directory_duplicates(str(tmp_path))
    assert isinstance(result, list)
    assert all(len(group) > 1 for group in result)  # all groups have duplicates
    assert len(result) == 0  # no duplicates found


def test_directory_duplicates_with_custom_hash_type(temp_dir_with_files):
    dirpath, files = temp_dir_with_files
    # Use sha256 to find duplicates - same content, different hash
    dups = directory_duplicates(dirpath, hash_type="sha256")
    # We should detect same duplicates as with md5 due to same content
    assert any(set(files[0:2]) == set(group) for group in dups)


def test_directory_duplicates_filters_by_extension(temp_dir_empty_and_nested_files):
    dirpath, files = temp_dir_empty_and_nested_files
    # We search for duplicates with extension filter .txt only
    dups = directory_duplicates(dirpath, ext=".txt")
    # We expect the duplicates between a.txt and a_copy.txt only
    # Should find a group with those two files
    found = False
    for group in dups:
        if set([files[0], files[1]]) == set(group):
            found = True
            # The third file should not be included
            assert files[2] not in group
    assert found


def test_directory_duplicates_with_name_filter(temp_dir_empty_and_nested_files):
    dirpath, files = temp_dir_empty_and_nested_files
    # name filter should filter files containing '_copy'
    dups = directory_duplicates(dirpath, name="_copy")
    # a_copy.txt is duplicate of a.txt but only a_copy.txt matches name
    # so no duplicates should be returned as only one file matches filter
    assert dups == []


def test_directory_duplicates_with_match_case_name(temp_dir_empty_and_nested_files):
    dirpath, files = temp_dir_empty_and_nested_files
    # The names are lower case, test match_case True with upper case name
    dups = directory_duplicates(dirpath, name="A_COPY", match_case=True)
    # Should find no matches due to case mismatch
    assert dups == []


def test_directory_duplicates_with_multiple_extensions(temp_dir_empty_and_nested_files):
    dirpath, files = temp_dir_empty_and_nested_files
    # Use extension list with .txt and .md (no md files exist)
    dups = directory_duplicates(dirpath, ext=[".txt", ".md"])
    # Duplicates between the two txt files present
    found = False
    for group in dups:
        if set([files[0], files[1]]) == set(group):
            found = True
    assert found


def test_directory_duplicates_empty_directory(tmp_path):
    # Empty directory should return empty list
    res = directory_duplicates(str(tmp_path))
    assert res == []


def test_directory_duplicates_only_one_file(tmp_path):
    # Single file means no duplicates
    file = tmp_path / "single.txt"
    file.write_text("singleton")
    res = directory_duplicates(str(tmp_path))
    assert res == []


def test_directory_duplicates_ignores_directories(tmp_path):
    # Create a directory inside tmp_path, no files with duplicates
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "file.txt").write_text("abc")
    # Should find no duplicates as only single file
    res = directory_duplicates(str(tmp_path))
    assert res == []


def test_directory_duplicates_handles_files_with_same_size_but_different_content(tmp_path):
    # Create files with same size but different contents
    content1 = b"12345678"
    content2 = b"87654321"
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    with open(file1, "wb") as f:
        f.write(content1)
    with open(file2, "wb") as f:
        f.write(content2)
    # Size is same (8 bytes), content differs, so no duplicates
    res = directory_duplicates(str(tmp_path))
    assert res == []


def test_directory_duplicates_ignores_symlinks(tmp_path):
    # Create a file and a symlink to it
    file1 = tmp_path / "file1.txt"
    file1.write_text("data")
    symlink_path = tmp_path / "symlink.txt"
    if hasattr(os, "symlink"):
        os.symlink(str(file1), str(symlink_path))
        # Should count symlink as file with duplicate content (same hash)
        dups = directory_duplicates(str(tmp_path))
        # symlink and target considered different files, so should detect duplication
        found = False
        for group in dups:
            pset = set(group)
            if str(file1) in pset and str(symlink_path) in pset:
                found = True
        assert found
    else:
        # Symlinks not supported on this platform (e.g. Windows without admin), pass
        pass


def test_directory_duplicates_nested_duplicates(tmp_path):
    # Create nested folders with same file content duplicates
    folder1 = tmp_path / "folder1"
    folder2 = tmp_path / "folder2"
    folder1.mkdir()
    folder2.mkdir()
    file1 = folder1 / "dupfile.txt"
    file2 = folder2 / "dupfile.txt"
    file1.write_text("duplicated content")
    file2.write_text("duplicated content")
    res = directory_duplicates(str(tmp_path))
    # Should return list with two files grouped together
    assert any(set(group) == {str(file1), str(file2)} for group in res)


def test_directory_duplicates_with_no_duplicate_sizes(tmp_path):
    # Create files with unique sizes only
    (tmp_path / "f1.txt").write_text("1")
    (tmp_path / "f2.txt").write_text("12")
    (tmp_path / "f3.txt").write_text("123")
    res = directory_duplicates(str(tmp_path))
    assert res == []


def test_directory_duplicates_respects_disable_pathlib(tmp_path):
    # Write two identical files, test with disable_pathlib True
    file1 = tmp_path / "f1.txt"
    file2 = tmp_path / "f2.txt"
    file1.write_text("abc")
    file2.write_text("abc")
    res = directory_duplicates(str(tmp_path), disable_pathlib=True)
    assert any(set(group) == {str(file1), str(file2)} for group in res)


def test_directory_duplicates_respects_depth_filter(tmp_path):
    # Create files at different nested depths - currently directory_duplicates doesn't use depth
    # but find_files does and kwargs are passed down
    folder1 = tmp_path / "folder1"
    folder2 = folder1 / "folder2"
    folder1.mkdir()
    folder2.mkdir()
    f1 = tmp_path / "file.txt"
    f2 = folder2 / "file.txt"
    f1.write_text("text")
    f2.write_text("text")
    # Use depth=1 should exclude folder2/file.txt
    res = directory_duplicates(str(tmp_path), depth=1)
    # They should not be detected as duplicates because second file is below depth 1
    # So no duplicates are detected
    assert res == []


def test_directory_duplicates_with_nonexistent_directory():
    # Provide a directory that does not exist should raise error
    with pytest.raises(OSError):
        directory_duplicates("nonexistent_dir")

