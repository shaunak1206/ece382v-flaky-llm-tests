import os
import random
import string
import tempfile
import shutil
from pathlib import Path
import pytest

from randomfiletree.core import (
    iterative_tree,
    iterative_gaussian_tree,
    choose_random_elements,
    sample_random_elements,
    random_string,
)


def test_random_string_length_and_charset():
    s = random_string(5, 10)
    assert 5 <= len(s) <= 10
    for ch in s:
        assert ch in string.ascii_uppercase + string.digits


def test_iterative_tree_creates_files_and_dirs(tmp_path):
    def folders(depth):  # Always create 1 folder per directory
        return 1

    def files(depth):  # Always create 2 files per directory
        return 2

    dirs, files_ = iterative_tree(tmp_path, folders, files, repeat=1, maxdepth=2)

    # We expect some directories and files created
    assert all(isinstance(d, Path) for d in dirs)
    assert all(isinstance(f, Path) for f in files_)

    # Check folders exist on disk
    for d in dirs:
        assert d.exists() and d.is_dir()

    # Check files exist on disk
    for f in files_:
        assert f.exists() and f.is_file()

    # Because repeat=1 and maxdepth=2, root + 1 subdir level
    # With 1 folder each, should have 1 + (1 folder created) = 2 dirs minimum (tmp_path + subdir)
    # But only folders created by iterative_tree are in `dirs`, root excluded
    assert len(dirs) >= 1
    assert len(files_) >= 2


def test_iterative_tree_with_payload(tmp_path):
    def folders(depth):
        return 1

    def files(depth):
        return 3

    # Define a simple payload generator: yields 3 files, writing content
    def payload_gen(dirpath):
        count = 0
        while True:
            if count >= 3:
                break
            p = dirpath / f"payload_{count}.txt"
            with open(p, "w") as f:
                f.write("content\n")
            yield p
            count += 1

    # Must pass a generator func that returns an iterator, so wrap in function
    def payload(dirpath):
        return payload_gen(dirpath)

    dirs, files_ = iterative_tree(
        tmp_path, folders, files, repeat=1, maxdepth=2, payload=payload
    )

    # Payload files should contain data
    for f in files_:
        assert f.exists() and f.is_file()
        with open(f, "r") as file:
            content = file.read()
        assert content == "content\n"


def test_iterative_tree_maxdepth_limits_creation(tmp_path):
    calls = {"folders": [], "files": []}

    def folders(depth):
        calls["folders"].append(depth)
        return 1

    def files(depth):
        calls["files"].append(depth)
        return 1

    # maxdepth=2 means depth can be 0 and 1 but not 2 or more
    iterative_tree(tmp_path, folders, files, repeat=1, maxdepth=2)

    # Check no depth larger than 1 was used
    assert all(d <= 1 for d in calls["folders"])
    assert all(d <= 1 for d in calls["files"])


def test_iterative_gaussian_tree_creates_files_and_dirs(tmp_path):
    dirs, files_ = iterative_gaussian_tree(tmp_path, nfiles=3, nfolders=2, repeat=1, maxdepth=2)

    assert all(isinstance(d, Path) for d in dirs)
    assert all(isinstance(f, Path) for f in files_)

    for d in dirs:
        assert d.exists() and d.is_dir()

    for f in files_:
        assert f.exists() and f.is_file()


def test_iterative_gaussian_tree_with_payload(tmp_path):
    # Similar payload function as above
    def payload_gen(dirpath):
        count = 0
        while True:
            if count >= 2:
                break
            p = dirpath / f"gpayload_{count}.txt"
            p.write_text("gaussian\n")
            yield p
            count += 1

    def payload(dirpath):
        return payload_gen(dirpath)

    dirs, files_ = iterative_gaussian_tree(
        tmp_path,
        nfiles=2,
        nfolders=1,
        repeat=1,
        maxdepth=2,
        payload=payload,
    )

    for f in files_:
        assert f.exists() and f.is_file()
        assert f.read_text() == "gaussian\n"


def test_choose_random_elements_raises_when_no_dirs(tmp_path):
    # Create only files, no dirs
    f = tmp_path / "testfile.txt"
    f.write_text("test")

    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0, onfail="raise")


def test_choose_random_elements_no_dirs_ignore(tmp_path):
    # Create only files, no dirs
    f = tmp_path / "testfile.txt"
    f.write_text("test")

    dirs, files_ = choose_random_elements(tmp_path, 1, 1, onfail="ignore")
    assert dirs == []  # no dirs but onfail=ignore -> empty
    assert files_ and all(f.exists() for f in files_)


def test_choose_random_elements_raises_when_no_files(tmp_path):
    # Create only dir, no file
    d = tmp_path / "subdir"
    d.mkdir()

    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="raise")


def test_choose_random_elements_unknown_onfail_value(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    f = tmp_path / "file.txt"
    f.write_text("ok")

    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 1, onfail="unknown")


def test_choose_random_elements_returns_requested_number(tmp_path):
    # Create dirs and files
    dirs = [tmp_path / f"dir{i}" for i in range(3)]
    for d in dirs:
        d.mkdir()
    files = [tmp_path / f"file{i}.txt" for i in range(4)]
    for f in files:
        f.write_text("x")

    selected_dirs, selected_files = choose_random_elements(tmp_path, 2, 3)
    assert len(selected_dirs) == 2
    assert len(selected_files) == 3


def test_sample_random_elements_raises_when_not_enough_dirs(tmp_path):
    d = tmp_path / "onlydir"
    d.mkdir()

    with pytest.raises(ValueError):
        sample_random_elements(tmp_path, 2, 0, onfail="raise")


def test_sample_random_elements_raises_when_not_enough_files(tmp_path):
    f = tmp_path / "onlyfile.txt"
    f.write_text("ok")

    with pytest.raises(ValueError):
        sample_random_elements(tmp_path, 0, 2, onfail="raise")


def test_sample_random_elements_ignore_returns_less_dirs_and_files(tmp_path):
    d1 = tmp_path / "dir1"
    d1.mkdir()
    d2 = tmp_path / "dir2"
    d2.mkdir()
    f1 = tmp_path / "file1.txt"
    f1.write_text("ok")

    # Request more dirs and files than available, onfail=ignore: should get all existing dirs and files
    selected_dirs, selected_files = sample_random_elements(tmp_path, 5, 3, onfail="ignore")
    assert set(selected_dirs) == {d1, d2}
    assert set(selected_files) == {f1}


def test_sample_random_elements_returns_exact_number_when_enough(tmp_path):
    dirs = [tmp_path / f"dir{i}" for i in range(4)]
    for d in dirs:
        d.mkdir()
    files = [tmp_path / f"file{i}.txt" for i in range(6)]
    for f in files:
        f.write_text("ok")

    selected_dirs, selected_files = sample_random_elements(tmp_path, 3, 5)
    assert len(selected_dirs) == 3
    assert len(selected_files) == 5


def test_sample_random_elements_unknown_onfail_value(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    f = tmp_path / "file.txt"
    f.write_text("ok")

    with pytest.raises(ValueError):
        sample_random_elements(tmp_path, 1, 1, onfail="invalid")


def test_choose_and_sample_random_elements_with_empty_basedir(tmp_path):
    # Empty dir: choose_random_elements with onfail=ignore returns empty lists
    dirs, files_ = choose_random_elements(tmp_path, 1, 1, onfail="ignore")
    assert dirs == []
    assert files_ == []

    # sample_random_elements with onfail=ignore returns empty lists as well
    dirs, files_ = sample_random_elements(tmp_path, 1, 1, onfail="ignore")
    assert dirs == []
    assert files_ == []


def test_iterative_tree_filename_callable_used(tmp_path):
    # Provide a filename callable that produces predictable names
    def folders(depth):
        return 0

    def files(depth):
        return 3

    def filename():
        return "fixedname.txt"

    dirs, files_ = iterative_tree(tmp_path, folders, files, repeat=1, filename=filename)
    assert len(files_) == 3
    for f in files_:
        assert f.name == "fixedname.txt"


def test_iterative_tree_payload_generator_is_iterable(tmp_path):
    def folders(depth):
        return 0

    def files(depth):
        return 2

    # Payload yields the exact file paths needed each iteration from generator
    def payload_gen(dirpath):
        for i in range(2):
            p = dirpath / f"pgen_{i}.txt"
            p.write_text("payload data")
            yield p

    def payload(dirpath):
        return payload_gen(dirpath)

    dirs, files_ = iterative_tree(tmp_path, folders, files, repeat=1, payload=payload)
    assert len(files_) == 2
    for f in files_:
        assert f.exists()
        assert f.read_text() == "payload data"
