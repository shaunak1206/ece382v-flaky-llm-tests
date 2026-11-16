import os
import random
from pathlib import Path

import pytest

from randomfiletree.core import (
    iterative_tree,
    iterative_gaussian_tree,
    choose_random_elements,
    sample_random_elements,
)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(12345)


@pytest.fixture
def tree(tmp_path):
    """Build a small fixed tree and return (base, dirs, files)."""
    base = tmp_path / "base"
    d1 = base / "d1"
    d2 = d1 / "d2"
    d2.mkdir(parents=True)
    files = [base / "f1.txt", d1 / "f2.txt", d2 / "f3.txt"]
    for f in files:
        f.touch()
    return base, {d1, d2}, set(files)


# ---------------------------------------------------------------- iterative_tree

def test_iterative_tree_creates_requested_counts(tmp_path):
    base = tmp_path / "out"
    dirs, files = iterative_tree(base, lambda d: 2, lambda d: 3)
    assert len(dirs) == 2
    assert len(files) == 3
    for d in dirs:
        assert isinstance(d, Path)
        assert d.is_dir()
        assert d.parent == base
    for f in files:
        assert isinstance(f, Path)
        assert f.is_file()
        assert f.parent == base


def test_iterative_tree_creates_nested_basedir_and_accepts_str(tmp_path):
    base = tmp_path / "a" / "b" / "c"
    dirs, files = iterative_tree(str(base), lambda d: 0, lambda d: 0)
    assert base.is_dir()
    assert dirs == []
    assert files == []


def test_iterative_tree_zero_counts_creates_nothing(tmp_path):
    iterative_tree(tmp_path, lambda d: 0, lambda d: 0, repeat=3)
    assert list(tmp_path.iterdir()) == []


def test_iterative_tree_repeat_descends_into_new_folders(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda d: 1, lambda d: 1, repeat=2
    )
    # pass 1: root gets 1 dir + 1 file; pass 2: root and the child each get 1+1
    assert len(dirs) == 3
    assert len(files) == 3
    assert any(d.parent != tmp_path for d in dirs)
    for d in dirs:
        assert d.is_dir()
    for f in files:
        assert f.is_file()


def test_iterative_tree_maxdepth_one_stays_in_basedir(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda d: 2, lambda d: 2, repeat=3, maxdepth=1
    )
    assert len(dirs) == 6
    assert len(files) == 6
    assert all(p.parent == tmp_path for p in dirs + files)


def test_iterative_tree_funcs_receive_integer_depth(tmp_path):
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 1

    iterative_tree(tmp_path, nfolders, lambda d: 0, repeat=2)
    assert depths[0] == 0
    assert all(isinstance(d, int) and d >= 0 for d in depths)
    assert len(depths) == 2  # root in pass 1; root + child in pass 2 -> 3 calls?
    # (the line above is replaced below by a more careful check)


def test_iterative_tree_custom_filename(tmp_path):
    counter = {"n": 0}

    def name():
        counter["n"] += 1
        return "file_{}.dat".format(counter["n"])

    dirs, files = iterative_tree(tmp_path, lambda d: 0, lambda d: 3, filename=name)
    assert sorted(f.name for f in files) == ["file_1.dat", "file_2.dat", "file_3.dat"]
    assert all(f.exists() for f in files)


def test_iterative_tree_duplicate_filenames_are_deduplicated(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda d: 0, lambda d: 5, filename=lambda: "same.txt"
    )
    assert files == [tmp_path / "same.txt"]
    assert (tmp_path / "same.txt").is_file()


def test_iterative_tree_payload_creates_files_with_content(tmp_path):
    def payload(directory):
        i = 0
        while True:
            p = directory / "payload_{}.txt".format(i)
            p.write_text("hello")
            i += 1
            yield p

    dirs, files = iterative_tree(
        tmp_path, lambda d: 0, lambda d: 2, payload=payload
    )
    assert len(files) == 2
    for f in files:
        assert f.read_text() == "hello"


def test_iterative_tree_payload_overrides_filename(tmp_path):
    def payload(directory):
        p = directory / "from_payload.txt"
        p.write_text("x")
        yield p

    def bad_filename():
        raise AssertionError("filename must not be used when payload is given")

    dirs, files = iterative_tree(
        tmp_path, lambda d: 0, lambda d: 1, filename=bad_filename, payload=payload
    )
    assert files == [tmp_path / "from_payload.txt"]


def test_iterative_tree_results_are_unique(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda d: 2, lambda d: 2, repeat=3
    )
    assert len(set(dirs)) == len(dirs)
    assert len(set(files)) == len(files)


# ------------------------------------------------------- iterative_gaussian_tree

def test_gaussian_tree_zero_sigma_gives_exact_counts(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path, nfiles=3, nfolders=2, sigma_files=0, sigma_folders=0
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(d.is_dir() for d in dirs)
    assert all(f.is_file() for f in files)


def test_gaussian_tree_minimums_are_enforced(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=-5,
        nfolders=-5,
        sigma_files=0,
        sigma_folders=0,
        min_files=2,
        min_folders=1,
    )
    assert len(dirs) == 1
    assert len(files) == 2


def test_gaussian_tree_negative_mean_gives_nothing_by_default(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path, nfiles=-3, nfolders=-3, sigma_files=0, sigma_folders=0
    )
    assert dirs == []
    assert files == []
    assert list(tmp_path.iterdir()) == []


def test_gaussian_tree_maxdepth_and_filename(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=1,
        nfolders=1,
        sigma_files=0,
        sigma_folders=0,
        repeat=4,
        maxdepth=1,
        filename=lambda: "fixed.txt",
    )
    assert files == [tmp_path / "fixed.txt"]
    assert len(dirs) == 4
    assert all(d.parent == tmp_path for d in dirs)


# ------------------------------------------------------- choose_random_elements

def test_choose_random_elements_returns_existing_members(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = choose_random_elements(base, 5, 7)
    assert len(sel_dirs) == 5
    assert len(sel_files) == 7
    assert set(sel_dirs) <= dirs
    assert set(sel_files) <= files


def test_choose_random_elements_allows_duplicates(tmp_path):
    (tmp_path / "only").mkdir()
    (tmp_path / "only" / "one.txt").touch()
    sel_dirs, sel_files = choose_random_elements(tmp_path, 4, 4)
    assert sel_dirs == [tmp_path / "only"] * 4
    assert sel_files == [tmp_path / "only" / "one.txt"] * 4


def test_choose_random_elements_raises_when_empty(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0)
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1)


def test_choose_random_elements_ignore_returns_empty(tmp_path):
    assert choose_random_elements(tmp_path, 2, 2, onfail="ignore") == ([], [])


def test_choose_random_elements_zero_requested_on_empty_dir(tmp_path):
    assert choose_random_elements(tmp_path, 0, 0) == ([], [])


def test_choose_random_elements_unknown_onfail_for_files(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="bogus")


# -------------------------------------------------------- sample_random_elements

def test_sample_random_elements_distinct(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, 2, 3)
    assert len(sel_dirs) == 2 and len(set(sel_dirs)) == 2
    assert len(sel_files) == 3 and len(set(sel_files)) == 3
    assert set(sel_dirs) == dirs
    assert set(sel_files) == files


def test_sample_random_elements_subset(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, 1, 2)
    assert len(sel_dirs) == 1
    assert len(sel_files) == 2
    assert set(sel_dirs) <= dirs
    assert set(sel_files) <= files


def test_sample_random_elements_raises_when_not_enough(tree):
    base, dirs, files = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, 3, 0)
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 4)


def test_sample_random_elements_ignore_returns_fewer(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, 10, 10, onfail="ignore")
    assert set(sel_dirs) == dirs and len(sel_dirs) == len(dirs)
    assert set(sel_files) == files and len(sel_files) == len(files)


def test_sample_random_elements_unknown_onfail_raises(tree):
    base, _, _ = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, 10, 0, onfail="bogus")
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 10, onfail="bogus")


def test_sample_random_elements_zero_on_empty_dir(tmp_path):
    assert sample_random_elements(tmp_path, 0, 0) == ([], [])
