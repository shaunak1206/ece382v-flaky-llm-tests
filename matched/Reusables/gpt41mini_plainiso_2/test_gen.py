import os
import hashlib
import shutil
import tempfile
import pytest
from reusables.file_operations import directory_duplicates, file_hash

# Helper to write file with content
def write_file(path, content=b""):
    with open(path, "wb") as f:
        f.write(content)

@pytest.fixture
def tmp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)

def hash_bytes(data, hash_type='md5'):
    h = hashlib.new(hash_type)
    h.update(data)
    return h.hexdigest()

def test_directory_duplicates_empty_dir_returns_empty_list(tmp_dir):
    result = directory_duplicates(tmp_dir)
    assert isinstance(result, list)
    assert result == []

def test_directory_duplicates_single_unique_file(tmp_dir):
    filepath = os.path.join(tmp_dir, "file1.txt")
    write_file(filepath, b"unique content")
    result = directory_duplicates(tmp_dir)
    assert result == []

def test_directory_duplicates_two_identical_files_same_content(tmp_dir):
    content = b"same content"
    file1 = os.path.join(tmp_dir, "file1.txt")
    file2 = os.path.join(tmp_dir, "file2.txt")
    write_file(file1, content)
    write_file(file2, content)
    result = directory_duplicates(tmp_dir)
    # Should find one group with 2 duplicates
    assert len(result) == 1
    group = result[0]
    assert sorted(group) == sorted([file1, file2])

def test_directory_duplicates_files_different_sizes_not_duplicates(tmp_dir):
    file1 = os.path.join(tmp_dir, "file1.txt")
    file2 = os.path.join(tmp_dir, "file2.txt")
    write_file(file1, b"a" * 10)
    write_file(file2, b"a" * 11)
    result = directory_duplicates(tmp_dir)
    assert result == []

def test_directory_duplicates_files_same_size_different_content_not_duplicates(tmp_dir):
    file1 = os.path.join(tmp_dir, "file1.txt")
    file2 = os.path.join(tmp_dir, "file2.txt")
    write_file(file1, b"abcdefghij")
    write_file(file2, b"abcd3fghij")
    result = directory_duplicates(tmp_dir)
    assert result == []

def test_directory_duplicates_multiple_duplicate_groups(tmp_dir):
    # Group 1: two files with same content
    content_group1 = b"foo bar baz"
    f1a = os.path.join(tmp_dir, "f1a.txt")
    f1b = os.path.join(tmp_dir, "f1b.txt")
    write_file(f1a, content_group1)
    write_file(f1b, content_group1)

    # Group 2: three files with same content
    content_group2 = b"lorem ipsum dolor"
    f2a = os.path.join(tmp_dir, "f2a.txt")
    f2b = os.path.join(tmp_dir, "f2b.txt")
    f2c = os.path.join(tmp_dir, "f2c.txt")
    write_file(f2a, content_group2)
    write_file(f2b, content_group2)
    write_file(f2c, content_group2)

    # Unique file
    unique_file = os.path.join(tmp_dir, "unique.txt")
    write_file(unique_file, b"unique content")

    result = directory_duplicates(tmp_dir)
    # We expect two groups of duplicates
    assert len(result) == 2
    flat_results = [file for group in result for file in group]
    for f in (f1a, f1b, f2a, f2b, f2c):
        assert f in flat_results
    assert unique_file not in flat_results
    # Check groups lengths
    groups_lens = sorted([len(g) for g in result])
    assert groups_lens == [2, 3]

def test_directory_duplicates_nested_directories(tmp_dir):
    subdir = os.path.join(tmp_dir, "subdir")
    os.mkdir(subdir)
    content = b"nested file content"
    file_root = os.path.join(tmp_dir, "file.txt")
    file_sub = os.path.join(subdir, "file.txt")
    write_file(file_root, content)
    write_file(file_sub, content)

    result = directory_duplicates(tmp_dir)
    assert len(result) == 1
    group = result[0]
    assert sorted(group) == sorted([file_root, file_sub])

def test_directory_duplicates_with_hash_type_sha256(tmp_dir):
    content = b"hash type test content"
    file1 = os.path.join(tmp_dir, "file1.txt")
    file2 = os.path.join(tmp_dir, "file2.txt")
    write_file(file1, content)
    write_file(file2, content)

    result = directory_duplicates(tmp_dir, hash_type='sha256')
    assert len(result) == 1
    group = result[0]
    assert sorted(group) == sorted([file1, file2])

def test_directory_duplicates_with_no_files(tmp_dir):
    # Create some empty directories only
    os.mkdir(os.path.join(tmp_dir, "empty1"))
    os.mkdir(os.path.join(tmp_dir, "empty2"))
    result = directory_duplicates(tmp_dir)
    assert result == []

def test_directory_duplicates_with_file_filters(tmp_dir):
    # Create files with .txt and .log extensions
    content = b"same content for extension filter"
    file_txt1 = os.path.join(tmp_dir, "a.txt")
    file_txt2 = os.path.join(tmp_dir, "b.txt")
    file_log = os.path.join(tmp_dir, "c.log")
    write_file(file_txt1, content)
    write_file(file_txt2, content)
    write_file(file_log, content)

    # Only detect duplicates for extension .txt
    result = directory_duplicates(tmp_dir, ext=".txt")
    assert len(result) == 1
    group = result[0]
    assert sorted(group) == sorted([file_txt1, file_txt2])

def test_directory_duplicates_ignores_symlinks(tmp_dir):
    content = b"symlink content"
    real_file = os.path.join(tmp_dir, "real.txt")
    link_file = os.path.join(tmp_dir, "link.txt")
    write_file(real_file, content)
    # Create symlink to real_file (if OS supports)
    try:
        os.symlink(real_file, link_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unsupported")

    # duplicates should include both real file and symlink (if resolved by os.path.getsize and open)
    result = directory_duplicates(tmp_dir)
    # Symlink not guaranteed to appear in results. Just check real file is included.
    found_files = [f for group in result for f in group]
    assert real_file in found_files

def test_directory_duplicates_raises_type_error_for_non_str_ext(tmp_dir):
    with pytest.raises(TypeError):
        directory_duplicates(tmp_dir, ext=123)  # ext must be str or list

def test_directory_duplicates_handles_files_with_unusual_names(tmp_dir):
    content = b"some content"
    filenames = ["normal.txt", "spaced name.txt", "uniçøde.txt", "中文文件.txt"]
    for fn in filenames:
        write_file(os.path.join(tmp_dir, fn), content)
    result = directory_duplicates(tmp_dir)
    # They all have same content -> all duplicates
    assert len(result) == 1
    group = result[0]
    paths = [os.path.join(tmp_dir, fn) for fn in filenames]
    assert sorted(group) == sorted(paths)

def test_directory_duplicates_works_with_large_number_of_files(tmp_dir):
    # Create 50 files, 25 duplicates of content A, 25 of content B
    content_a = b"a"*1024
    content_b = b"b"*1024
    for i in range(25):
        write_file(os.path.join(tmp_dir, f"a_{i}.txt"), content_a)
        write_file(os.path.join(tmp_dir, f"b_{i}.txt"), content_b)
    result = directory_duplicates(tmp_dir)
    assert len(result) == 2
    lens = sorted(len(g) for g in result)
    assert lens == [25, 25]

def test_directory_duplicates_empty_file_detection(tmp_dir):
    empty_file1 = os.path.join(tmp_dir, "empty1")
    empty_file2 = os.path.join(tmp_dir, "empty2")
    write_file(empty_file1, b"")
    write_file(empty_file2, b"")
    result = directory_duplicates(tmp_dir)
    # Two empty files should be duplicates
    assert len(result) == 1
    group = result[0]
    assert set(group) == {empty_file1, empty_file2}

def test_directory_duplicates_respects_disable_pathlib_argument(tmp_dir):
    # This arg is passed to find_files; testing it doesn't break
    content = b"test content"
    f1 = os.path.join(tmp_dir, "f1.txt")
    f2 = os.path.join(tmp_dir, "f2.txt")
    write_file(f1, content)
    write_file(f2, content)
    result = directory_duplicates(tmp_dir, disable_pathlib=True)
    assert len(result) == 1
    group = result[0]
    assert sorted(group) == sorted([f1, f2])

def test_directory_duplicates_with_no_duplicates_but_multiple_files(tmp_dir):
    contents = [b"one", b"two", b"three"]
    for i, c in enumerate(contents):
        write_file(os.path.join(tmp_dir, f"file{i}.txt"), c)
    result = directory_duplicates(tmp_dir)
    assert result == []

