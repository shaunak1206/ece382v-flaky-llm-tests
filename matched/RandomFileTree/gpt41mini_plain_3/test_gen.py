import os
import random
import string
import pytest
from pathlib import Path
from randomfiletree import core


@pytest.fixture
def tmp_tree(tmp_path):
    # Setup a simple directory tree for tests
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    (tmp_path / "file1.txt").write_text("file1")
    (subfolder / "file2.txt").write_text("file2")
    return tmp_path


def test_random_string_length_and_chars():
    for _ in range(100):
        s = core.random_string(5, 10)
        assert 5 <= len(s) <= 10
        assert all(c in string.ascii_uppercase + string.digits for c in s)


def test_iterative_tree_creates_folders_and_files(tmp_path):
    def nfolders(depth):
        return 1 if depth == 0 else 0

    def nfiles(depth):
        return 2 if depth == 0 else 0

    dirs, files = core.iterative_tree(tmp_path, nfolders, nfiles, repeat=1)
    # Should include one new folder at depth 0 and two files
    assert len(dirs) == 1
    assert all(isinstance(d, Path) for d in dirs)
    assert len(files) == 2
    assert all(isinstance(f, Path) for f in files)
    # Check that the files and folders actually exist
    for d in dirs:
        assert d.is_dir()
    for f in files:
        assert f.is_file()


def test_iterative_tree_respects_maxdepth(tmp_path):
    def nf(depth):
        return 1

    def nfi(depth):
        return 1

    dirs1, files1 = core.iterative_tree(tmp_path, nf, nfi, repeat=1, maxdepth=2)
    # All created dirs should be at depth < 2
    for d in dirs1:
        depth = d.relative_to(tmp_path).parts
        assert len(depth) < 2


def test_iterative_tree_payload_function(tmp_path):
    # Payload generator yields created files in the directory
    def payload_gen(dir_path):
        # Let's create a file named 'fileX' where X increments
        count = 0
        while True:
            f = dir_path / f"file{count}.txt"
            f.write_text("content")
            count += 1
            yield f

    def nf(depth):
        return 1

    def nfi(depth):
        return 2

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=nf,
        nfiles_func=nfi,
        repeat=1,
        payload=payload_gen
    )
    # Should create 1 folder and 2 files for every folder walked (including base)
    # Because only base, expect 1 folder + 2 files
    # Files must contain created files by payload
    assert len(dirs) == 1
    assert len(files) >= 2
    # Files created by payload have content
    for f in files:
        assert f.is_file()
        assert f.read_text() == "content"


def test_iterative_gaussian_tree_creates(tmp_path):
    dirs, files = core.iterative_gaussian_tree(tmp_path, nfiles=3, nfolders=2, repeat=1)
    # It should create some directories and files
    assert all(isinstance(d, Path) for d in dirs)
    assert all(isinstance(f, Path) for f in files)
    # They should physically exist
    for d in dirs:
        assert d.exists() and d.is_dir()
    for f in files:
        assert f.exists() and f.is_file()


def test_choose_random_elements_selects(tmp_tree):
    # Choose existing directories and files
    dirs, files = core.choose_random_elements(tmp_tree, 1, 1)
    assert len(dirs) == 1
    assert len(files) == 1
    for d in dirs:
        assert d.is_dir()
    for f in files:
        assert f.is_file()


def test_choose_random_elements_raises_no_dirs(tmp_path):
    # Directory contains no subfolders
    (tmp_path / "file.txt").touch()
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, 1, 0)


def test_choose_random_elements_raises_no_files(tmp_path):
    # Directory contains no files
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, 0, 1)


def test_choose_random_elements_ignore_onfail(tmp_path):
    (tmp_path / "sub").mkdir()
    # No files but onfail ignore, should not raise but return empty file list
    dirs, files = core.choose_random_elements(tmp_path, 1, 1, onfail="ignore")
    assert len(files) == 0


def test_sample_random_elements_selects_unique(tmp_tree):
    dirs, files = core.sample_random_elements(tmp_tree, 1, 1)
    assert len(dirs) == 1
    assert len(files) == 1
    # Ensure those dirs and files are present on disk
    for d in dirs:
        assert d.exists() and d.is_dir()
    for f in files:
        assert f.exists() and f.is_file()


def test_sample_random_elements_raises_not_enough_dirs(tmp_tree):
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_tree, 1000, 0, onfail="raise")


def test_sample_random_elements_raises_not_enough_files(tmp_tree):
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_tree, 0, 1000, onfail="raise")


def test_sample_random_elements_ignore_onfail(tmp_tree):
    # Request more dirs/files than available with onfail ignore should return less
    dirs, files = core.sample_random_elements(tmp_tree, 1000, 1000, onfail="ignore")
    # Should be less or equal to actual number available
    assert len(dirs) <= 2
    assert len(files) <= 2


def test_choose_random_elements_unknown_onfail_raises(tmp_tree):
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_tree, 0, 1, onfail="unknown")


def test_sample_random_elements_unknown_onfail_raises(tmp_tree):
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_tree, 0, 1, onfail="unknown")


def test_iterative_tree_repeat_creates_more(tmp_path):
    def nf(depth):
        return 1

    def nfi(depth):
        return 1

    dirs1, files1 = core.iterative_tree(tmp_path, nf, nfi, repeat=1)
    dirs2, files2 = core.iterative_tree(tmp_path, nf, nfi, repeat=3)
    # Repeat 3 times should create more or equal files and folders than repeat=1
    assert len(dirs2) >= len(dirs1)
    assert len(files2) >= len(files1)


def test_iterative_tree_with_filename_func(tmp_path):
    def nf(depth):
        return 0

    def nfi(depth):
        return 3

    def static_filename():
        return "myfile.txt"

    dirs, files = core.iterative_tree(tmp_path, nf, nfi, filename=static_filename, repeat=1)
    # Because filename is always same, files list should have only one entry due to touch overwrite
    assert len(files) == 1
    filepath = tmp_path / "myfile.txt"
    assert filepath.exists()


def test_iterative_gaussian_tree_min_values(tmp_path):
    dirs, files = core.iterative_gaussian_tree(
        tmp_path,
        nfiles=0,
        nfolders=0,
        sigma_folders=5,
        sigma_files=5,
        min_folders=2,
        min_files=3,
        repeat=1,
    )
    # Minimums are respected
    assert len(dirs) >= 2
    assert len(files) >= 3


def test_payload_generator_yields_files(tmp_path):
    callcount = 0

    def payload_gen(dir_path):
        nonlocal callcount
        while True:
            path = dir_path / f"file_{callcount}.txt"
            path.write_text("x" * callcount)
            callcount += 1
            yield path

    def nf(depth):
        return 0

    def nfi(depth):
        return 3

    _, files = core.iterative_tree(tmp_path, nf, nfi, payload=payload_gen)
    assert len(files) == 3
    for i, f in enumerate(files):
        assert f.exists()
        assert f.read_text() == "x" * i


def test_iterative_tree_maxdepth_limits_creation(tmp_path):
    depth_created = []

    def nf(depth):
        depth_created.append(depth)
        return 1

    def nfi(depth):
        depth_created.append(depth)
        return 0

    core.iterative_tree(tmp_path, nf, nfi, maxdepth=2)

    # Only depths 0 and 1 should be included (maxdepth=2 means depth >= 1 stops)
    assert all(d < 2 for d in depth_created)
