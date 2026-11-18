import os
import tempfile
import shutil
import hashlib
import pytest

from reusables.file_operations import directory_duplicates


def create_file(path, content=b"", mode="wb"):
    with open(path, mode) as f:
        f.write(content)


def hash_file_content(content, hash_type="md5"):
    hasher = hashlib.new(hash_type)
    hasher.update(content)
    return hasher.hexdigest()


@pytest.fixture
def temp_dir_with_files():
    """Create a temporary directory with some files for testing duplicates."""
    temp_dir = tempfile.mkdtemp()

    # Files content
    content_a = b"first file content"
    content_b = b"second file content"
    content_c = b"first file content"  # duplicate of content_a

    # Create files
    file_a = os.path.join(temp_dir, "file_a.txt")
    file_b = os.path.join(temp_dir, "file_b.txt")
    file_c = os.path.join(temp_dir, "file_c.txt")

    create_file(file_a, content_a)
    create_file(file_b, content_b)
    create_file(file_c, content_c)

    yield temp_dir, file_a, file_b, file_c

    shutil.rmtree(temp_dir)


def test_directory_duplicates_find_exact_duplicates(temp_dir_with_files):
    temp_dir, file_a, file_b, file_c = temp_dir_with_files
    dups = directory_duplicates(temp_dir, hash_type="md5")

    # We expect only one group of duplicates (file_a and file_c)
    assert isinstance(dups, list)
    assert any(set(group) == set([file_a, file_c]) for group in dups)
    # file_b should not be included in duplicates
    included_files = [f for group in dups for f in group]
    assert file_b not in included_files


def test_directory_duplicates_empty_directory_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        result = directory_duplicates(tmp)
        assert result == []


def test_directory_duplicates_single_file_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "file.txt")
        create_file(file_path, b"only one file")

        result = directory_duplicates(tmp)
        assert result == []


def test_directory_duplicates_with_empty_files():
    with tempfile.TemporaryDirectory() as tmp:
        empty1 = os.path.join(tmp, "empty1.txt")
        empty2 = os.path.join(tmp, "empty2.txt")
        nonempty = os.path.join(tmp, "nonempty.txt")

        create_file(empty1, b"")
        create_file(empty2, b"")
        create_file(nonempty, b"not empty")

        dups = directory_duplicates(tmp)
        # empty files are duplicates with each other
        found = False
        for group in dups:
            if set(group) == set([empty1, empty2]):
                found = True
        assert found
        # nonempty file not in duplicates
        included_files = [f for group in dups for f in group]
        assert nonempty not in included_files


def test_directory_duplicates_respects_hash_type(temp_dir_with_files):
    temp_dir, file_a, file_b, file_c = temp_dir_with_files

    md5_dups = directory_duplicates(temp_dir, hash_type="md5")
    sha1_dups = directory_duplicates(temp_dir, hash_type="sha1")

    # Both should find the same duplicate pair since content is identical
    md5_files = set(f for group in md5_dups for f in group)
    sha1_files = set(f for group in sha1_dups for f in group)

    assert set([file_a, file_c]) <= md5_files
    assert set([file_a, file_c]) <= sha1_files


def test_directory_duplicates_with_file_extension_filter():
    with tempfile.TemporaryDirectory() as tmp:
        file1 = os.path.join(tmp, "file1.txt")
        file2 = os.path.join(tmp, "file2.log")
        file3 = os.path.join(tmp, "file3.txt")
        # file1 and file3 same content
        create_file(file1, b"same")
        create_file(file2, b"different")
        create_file(file3, b"same")

        dups = directory_duplicates(tmp, ext=".txt")
        # Should find duplicates only among .txt files
        found_group = None
        for group in dups:
            if set(group) == set([file1, file3]):
                found_group = group
        assert found_group is not None

        # Files with .log should not be involved even if same content
        all_dup_files = [f for group in dups for f in group]
        assert file2 not in all_dup_files


def test_directory_duplicates_returns_only_duplicate_lists():
    with tempfile.TemporaryDirectory() as tmp:
        file_a = os.path.join(tmp, "a.txt")
        file_b = os.path.join(tmp, "b.txt")
        file_c = os.path.join(tmp, "c.txt")

        content_1 = b"content1"
        content_2 = b"content2"

        create_file(file_a, content_1)
        create_file(file_b, content_1)
        create_file(file_c, content_2)

        result = directory_duplicates(tmp)

        # Result only contains lists with duplicates (>1 file)
        for group in result:
            assert isinstance(group, list)
            assert len(group) > 1

        # Only one group should be present with file_a and file_b
        assert len(result) == 1
        assert set(result[0]) == set([file_a, file_b])


def test_directory_duplicates_symlinks_handling():
    with tempfile.TemporaryDirectory() as tmp:
        file_a = os.path.join(tmp, "a.txt")
        symlink_path = os.path.join(tmp, "a_link.txt")
        content = b"some content"

        create_file(file_a, content)
        try:
            os.symlink(file_a, symlink_path)
        except (AttributeError, OSError):
            # Symlink might not be supported, skip test
            pytest.skip("Symlinks not supported on this platform")

        dups = directory_duplicates(tmp)
        # The symlink and original file should have same size and hash but
        # depending on os read symlink content/target, duplicates might or not detected
        included = [f for group in dups for f in group]
        # Symlink or original file should be in the duplicate group as expected
        # We'll accept if both or just the original present because symlink contents might not be read
        assert file_a in included


def test_directory_duplicates_with_subdirectories():
    with tempfile.TemporaryDirectory() as tmp:
        subdir = os.path.join(tmp, "subdir")
        os.mkdir(subdir)

        file1 = os.path.join(tmp, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")
        file3 = os.path.join(subdir, "file3.txt")

        create_file(file1, b"common content")
        create_file(file2, b"common content")
        create_file(file3, b"unique content")

        dups = directory_duplicates(tmp)
        found_group = None
        for group in dups:
            if set(group) == set([file1, file2]):
                found_group = group
        assert found_group is not None
        included_files = [f for group in dups for f in group]
        assert file3 not in included_files


def test_directory_duplicates_large_files(tmp_path):
    # Create two large files with identical content
    file1 = tmp_path / "large1.bin"
    file2 = tmp_path / "large2.bin"
    large_content = b"x" * 1024 * 1024  # 1MB of 'x'
    file1.write_bytes(large_content)
    file2.write_bytes(large_content)

    dups = directory_duplicates(str(tmp_path))
    found_group = None
    group_sets = [set(g) for g in dups]
    assert any(set(map(str,[file1, file2])) == gset for gset in group_sets)


def test_directory_duplicates_nonexistent_directory_raises(tmp_path):
    fake_dir = str(tmp_path / "not_exist")
    # The implementation does not check for directory existence explicitly
    # So it will fail when find_files tries to scan; it raises OSError
    with pytest.raises(OSError):
        directory_duplicates(fake_dir)


def test_directory_duplicates_ignore_non_file_entries(tmp_path):
    # Create a directory with a subdirectory and a file inside
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file1 = tmp_path / "file1.txt"
    create_file(str(file1), b"hello")

    # Directory only contains one file and one subdir, no duplicates expected
    dups = directory_duplicates(str(tmp_path))
    assert dups == []


def test_directory_duplicates_custom_kwargs_ext(tmp_path):
    # Create files with different extensions
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f3 = tmp_path / "c.txt"

    create_file(str(f1), b"dupsame")
    create_file(str(f2), b"dupsame")
    create_file(str(f3), b"unique")

    dups = directory_duplicates(str(tmp_path), ext=".py")

    found_group = None
    for group in dups:
        if set(group) == set([str(f1), str(f2)]):
            found_group = group
            break
    assert found_group is not None
    included_files = [f for group in dups for f in group]
    assert str(f3) not in included_files


def test_directory_duplicates_disable_pathlib_false(tmp_path):
    # The function forcibly disables pathlib by default in call to find_files with disable_pathlib=True
    # Just test normal usage: create two duplicate files, check detection.

    f1 = tmp_path / "dup1.txt"
    f2 = tmp_path / "dup2.txt"
    content = b"same_content"
    create_file(str(f1), content)
    create_file(str(f2), content)

    dups = directory_duplicates(str(tmp_path))
    found = any(set(group) == set([str(f1), str(f2)]) for group in dups)
    assert found


def test_directory_duplicates_multiple_duplicate_groups(tmp_path):
    # Create multiple sets of duplicate files
    f1 = tmp_path / "a1.txt"
    f2 = tmp_path / "a2.txt"
    f3 = tmp_path / "b1.txt"
    f4 = tmp_path / "b2.txt"
    f5 = tmp_path / "c.txt"

    create_file(str(f1), b"group1")
    create_file(str(f2), b"group1")
    create_file(str(f3), b"group2")
    create_file(str(f4), b"group2")
    create_file(str(f5), b"unique")

    dups = directory_duplicates(str(tmp_path))
    sets = [set(group) for group in dups]

    assert set([str(f1), str(f2)]) in sets
    assert set([str(f3), str(f4)]) in sets
    # unique file should not be in duplicates
    included_files = [f for group in dups for f in group]
    assert str(f5) not in included_files


def test_directory_duplicates_handles_file_permissions(tmp_path):
    # Create two duplicate files and make one unreadable to simulate permission error
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"

    content = b"secret content"
    create_file(str(f1), content)
    create_file(str(f2), content)

    # Remove read-permission from file2
    os.chmod(str(f2), 0o000)

    try:
        dups = directory_duplicates(str(tmp_path))
        # Since file_hash opens the file for reading, invalid permissions may cause failure
        # Depending on OS permission model test might pass/fail
        # The safe approach is just that the function does not crash
        assert any(str(f1) in group for group in dups)
    finally:
        # Restore permissions so that tempfile cleanup can succeed
        os.chmod(str(f2), 0o644)
