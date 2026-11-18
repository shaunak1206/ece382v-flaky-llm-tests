import os
import random
import string
import pytest
from pathlib import Path
from randomfiletree import core


def test_iterative_tree_creates_files_and_dirs(tmp_path):
    # Setup nfolders_func and nfiles_func to create fixed numbers
    def nfolders_func(depth): return 1 if depth == 0 else 0
    def nfiles_func(depth): return 2 if depth == 0 else 0

    dirs, files = core.iterative_tree(tmp_path, nfolders_func, nfiles_func, repeat=1)
    # There should be exactly one new folder and two new files in root
    assert isinstance(dirs, list)
    assert isinstance(files, list)
    assert len(dirs) == 1
    assert len(files) == 2
    for d in dirs:
        assert d.is_dir()
    for f in files:
        assert f.is_file()


def test_iterative_tree_respects_maxdepth(tmp_path):
    def nfolders_func(depth): return 1
    def nfiles_func(depth): return 1

    # Depth limit to 1 means only create items in root, not in subfolders
    dirs, files = core.iterative_tree(tmp_path, nfolders_func, nfiles_func, maxdepth=1, repeat=1)
    # Only root level items, so all folders have depth 0; no subfolders inside created subfolders
    # Thus, only one folder created (at depth 0), no deeper folders
    assert all(d.parent == tmp_path for d in dirs)
    assert all(f.parent == tmp_path for f in files)


def test_iterative_tree_payload_creates_files(tmp_path):
    def nfolders_func(depth): return 0
    def nfiles_func(depth): return 3

    def payload_generator(dir_path):
        # Generator that creates 3 files with predetermined names
        def gen():
            for i in range(3):
                fp = dir_path / f"file_{i}.txt"
                fp.write_text(f"content {i}")
                yield fp
        return gen()

    dirs, files = core.iterative_tree(tmp_path, nfolders_func, nfiles_func, payload=payload_generator)
    assert dirs == []
    assert len(files) == 3
    for f in files:
        assert f.exists()
        # Files created by payload have content
        assert f.read_text().startswith("content")


def test_iterative_tree_payload_overrides_filename(tmp_path):
    # Payload creates file with fixed name; filename function returns random string but should not be used.
    def nfolders_func(depth): return 0
    def nfiles_func(depth): return 1

    def payload_generator(dir_path):
        def gen():
            fp = dir_path / "fixed_name.txt"
            fp.write_text("hello")
            yield fp
        return gen()

    dirs, files = core.iterative_tree(tmp_path, nfolders_func, nfiles_func, payload=payload_generator)
    assert len(files) == 1
    assert files[0].name == "fixed_name.txt"
    assert files[0].read_text() == "hello"


def test_iterative_gaussian_tree_creates_files_and_dirs(tmp_path):
    # Set seed for reproducibility
    random.seed(12345)
    dirs, files = core.iterative_gaussian_tree(tmp_path, nfiles=3, nfolders=2, repeat=1, sigma_folders=0, sigma_files=0)
    # With sigma=0 number of files/folders should be fixed at avg (nfiles/nfolders)
    assert len(dirs) == 2
    assert len(files) == 3
    for d in dirs:
        assert d.is_dir()
    for f in files:
        assert f.is_file()


def test_choose_random_elements_raises_on_empty_dirs_files(tmp_path):
    # Directory empty, should raise when asking for dirs or files
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignore_onfail_no_dirs_files(tmp_path):
    # No dirs or files but onfail="ignore" should return empty lists
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=1, n_files=1, onfail="ignore")
    assert dirs == []
    assert files == []


def test_choose_random_elements_selects_existing(tmp_path):
    # Prepare directory with one subfolder and one file
    subdir = tmp_path / "sub"
    subdir.mkdir()
    file1 = tmp_path / "file1.txt"
    file1.write_text("content")
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=1, n_files=1, onfail="raise")
    assert dirs[0] == subdir
    assert files[0] == file1


def test_sample_random_elements_selects_distinct(tmp_path):
    # Create 3 folders and 3 files
    for i in range(3):
        (tmp_path / f"dir{i}").mkdir()
        (tmp_path / f"file{i}.txt").write_text(f"content {i}")
    dirs, files = core.sample_random_elements(tmp_path, n_dirs=2, n_files=2, onfail="raise")
    assert len(dirs) == 2
    assert len(files) == 2
    # Selected are distinct, and from created sets
    all_dirs = {tmp_path / f"dir{i}" for i in range(3)}
    all_files = {tmp_path / f"file{i}.txt" for i in range(3)}
    assert all(d in all_dirs for d in dirs)
    assert len(set(dirs)) == len(dirs)
    assert all(f in all_files for f in files)
    assert len(set(files)) == len(files)


def test_sample_random_elements_raises_when_not_enough(tmp_path):
    # Only 1 folder and 1 file created
    (tmp_path / "onlydir").mkdir()
    (tmp_path / "onlyfile.txt").write_text("c")

    # Request more than available with onfail='raise'
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=1, onfail="raise")
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_path, n_dirs=1, n_files=2, onfail="raise")


def test_sample_random_elements_ignores_when_not_enough(tmp_path):
    # Create only 2 folders and 2 files
    for i in range(2):
        (tmp_path / f"d{i}").mkdir()
        (tmp_path / f"f{i}.txt").write_text("x")
    # Request 3 dirs and 3 files with onfail='ignore' returns only 2 each
    dirs, files = core.sample_random_elements(tmp_path, n_dirs=3, n_files=3, onfail="ignore")
    assert len(dirs) == 2
    assert len(files) == 2


def test_choose_random_elements_unknown_onfail_raises(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=1, onfail="unknown")


def test_sample_random_elements_unknown_onfail_raises(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_path, n_dirs=1, n_files=1, onfail="unknown")


def test_iterative_tree_repeat_creates_multiple_levels(tmp_path):
    # Creates one folder and one file each iteration, repeat=2 should create nested structure
    def nfolders(depth): return 1 if depth < 2 else 0
    def nfiles(depth): return 1

    dirs, files = core.iterative_tree(tmp_path, nfolders, nfiles, repeat=2, maxdepth=3)
    # There should be multiple folders nested, check nesting
    assert len(dirs) >= 2
    assert len(files) >= 2
    # All dirs and files exist in proper nesting
    for d in dirs:
        assert d.is_dir()
        assert tmp_path in d.parents or d == tmp_path
    for f in files:
        assert f.is_file()
        assert tmp_path in f.parents


def test_iterative_tree_filename_callable_used(tmp_path):
    # filename callable that returns fixed name each call (plus index to generate different files)
    def nfolders(depth): return 0
    def nfiles(depth): return 3
    counter = -1

    def filename():
        nonlocal counter
        counter += 1
        return f"fixedfile{counter}.txt"

    dirs, files = core.iterative_tree(tmp_path, nfolders, nfiles, filename=filename)
    assert len(files) == 3
    expected_names = {f"fixedfile{i}.txt" for i in range(3)}
    actual_names = {f.name for f in files}
    assert actual_names == expected_names


def test_iterative_gaussian_tree_minimums_respected(tmp_path):
    random.seed(0)

    # Use sigma=0 to fix number of files/folders near to mean
    dirs, files = core.iterative_gaussian_tree(
        tmp_path, nfiles=2, nfolders=3, sigma_folders=0, sigma_files=0,
        min_folders=5, min_files=5, repeat=1
    )
    # Because min_folders and min_files are higher than mean, created should be at least those minimums
    assert len(dirs) >= 5
    assert len(files) >= 5


def test_choose_random_elements_multiple_selection(tmp_path):
    # Create multiple folders and files
    for i in range(5):
        (tmp_path / f"d{i}").mkdir()
        (tmp_path / f"f{i}.txt").write_text(str(i))
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=3, n_files=4)
    assert len(dirs) == 3
    assert len(files) == 4
    # No guarantees about uniqueness in choose_random_elements
    # Check that all selected exist in the directory tree
    all_dirs = {tmp_path / f"d{i}" for i in range(5)}
    all_files = {tmp_path / f"f{i}.txt" for i in range(5)}
    assert all(d in all_dirs for d in dirs)
    assert all(f in all_files for f in files)


def test_sample_random_elements_zero_selection(tmp_path):
    # Create some dirs and files
    (tmp_path / "a").mkdir()
    (tmp_path / "b.txt").write_text("text")
    # Ask for zero dirs and zero files: should return empty lists without error
    dirs, files = core.sample_random_elements(tmp_path, n_dirs=0, n_files=0)
    assert dirs == []
    assert files == []


def test_choose_random_elements_zero_selection(tmp_path):
    # Create some dirs and files
    (tmp_path / "a").mkdir()
    (tmp_path / "b.txt").write_text("text")
    # Ask for zero dirs and zero files: should return empty lists without error
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=0, n_files=0)
    assert dirs == []
    assert files == []
