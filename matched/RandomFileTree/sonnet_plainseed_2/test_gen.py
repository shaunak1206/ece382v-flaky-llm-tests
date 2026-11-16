import random
from pathlib import Path

import pytest

from randomfiletree.core import (
    choose_random_elements,
    iterative_gaussian_tree,
    iterative_tree,
    sample_random_elements,
)


@pytest.fixture
def sample_tree(tmp_path):
    """3 directories (a, b, a/c) and 4 files."""
    (tmp_path / "a" / "c").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    files = [
        tmp_path / "f1.txt",
        tmp_path / "a" / "f2.txt",
        tmp_path / "b" / "f3.txt",
        tmp_path / "a" / "c" / "f4.txt",
    ]
    for f in files:
        f.touch()
    dirs = [tmp_path / "a", tmp_path / "b", tmp_path / "a" / "c"]
    return tmp_path, set(dirs), set(files)


# ---------------------------------------------------------------- iterative_tree

def test_iterative_tree_creates_requested_counts(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda depth: 2, lambda depth: 3, repeat=1
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(isinstance(p, Path) for p in dirs + files)
    for d in dirs:
        assert d.is_dir()
        assert d.parent == tmp_path
    for f in files:
        assert f.is_file()
        assert f.stat().st_size == 0
        assert f.parent == tmp_path


def test_iterative_tree_creates_missing_basedir_from_str(tmp_path):
    base = tmp_path / "x" / "y" / "z"
    dirs, files = iterative_tree(str(base), lambda d: 1, lambda d: 1)
    assert base.is_dir()
    assert len(dirs) == 1
    assert len(files) == 1
    assert dirs[0].parent == base


def test_iterative_tree_zero_counts_returns_empty(tmp_path):
    dirs, files = iterative_tree(tmp_path, lambda d: 0, lambda d: 0, repeat=3)
    assert dirs == []
    assert files == []
    assert list(tmp_path.iterdir()) == []


def test_iterative_tree_funcs_called_with_depth_zero_for_root(tmp_path):
    folder_calls = []
    file_calls = []

    def nfolders(depth):
        folder_calls.append(depth)
        return 0

    def nfiles(depth):
        file_calls.append(depth)
        return 0

    iterative_tree(tmp_path, nfolders, nfiles, repeat=1)
    assert folder_calls == [0]
    assert file_calls == [0]


def test_iterative_tree_repeat_descends_into_new_folders(tmp_path):
    dirs, files = iterative_tree(tmp_path, lambda d: 1, lambda d: 0, repeat=2)
    # iteration 1: 1 dir in root; iteration 2: 1 more in root and 1 in the old dir
    assert len(dirs) == 3
    assert files == []
    nested = [d for d in dirs if d.parent != tmp_path]
    assert len(nested) == 1
    assert nested[0].parent.parent == tmp_path
    for d in dirs:
        assert d.is_dir()


def test_iterative_tree_maxdepth_one_stays_in_basedir(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, lambda d: 2, lambda d: 1, repeat=3, maxdepth=1
    )
    assert len(dirs) == 6
    assert len(files) == 3
    assert all(p.parent == tmp_path for p in dirs + files)


def test_iterative_tree_custom_filename(tmp_path):
    counter = {"n": 0}

    def name():
        counter["n"] += 1
        return "custom_{}.dat".format(counter["n"])

    _, files = iterative_tree(
        tmp_path, lambda d: 0, lambda d: 3, filename=name
    )
    assert sorted(f.name for f in files) == [
        "custom_1.dat",
        "custom_2.dat",
        "custom_3.dat",
    ]
    for f in files:
        assert f.exists()


def test_iterative_tree_payload_creates_files_with_content(tmp_path):
    def payload(directory):
        i = 0
        while True:
            p = directory / "payload_{}.txt".format(i)
            p.write_text("hello")
            yield p
            i += 1

    def bad_filename():
        raise AssertionError("filename must be ignored when payload is given")

    dirs, files = iterative_tree(
        tmp_path,
        lambda d: 0,
        lambda d: 2,
        filename=bad_filename,
        payload=payload,
    )
    assert dirs == []
    assert sorted(f.name for f in files) == ["payload_0.txt", "payload_1.txt"]
    for f in files:
        assert f.read_text() == "hello"


# ---------------------------------------------------- iterative_gaussian_tree

def test_gaussian_tree_zero_sigma_gives_exact_counts(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=3,
        nfolders=2,
        sigma_files=0,
        sigma_folders=0,
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(d.is_dir() for d in dirs)
    assert all(f.is_file() for f in files)


def test_gaussian_tree_min_values_enforced(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=-10,
        nfolders=-10,
        sigma_files=0,
        sigma_folders=0,
        min_files=4,
        min_folders=2,
    )
    assert len(dirs) == 2
    assert len(files) == 4


def test_gaussian_tree_negative_mean_gives_nothing(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=-5,
        nfolders=-5,
        sigma_files=0,
        sigma_folders=0,
    )
    assert dirs == []
    assert files == []


def test_gaussian_tree_maxdepth_and_payload_passthrough(tmp_path):
    def payload(directory):
        i = 0
        while True:
            p = directory / "pl_{}.bin".format(i)
            p.write_bytes(b"xyz")
            yield p
            i += 1

    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=1,
        nfolders=1,
        sigma_files=0,
        sigma_folders=0,
        repeat=3,
        maxdepth=1,
        payload=payload,
    )
    assert len(dirs) == 3
    assert len(files) == 1  # same name each time in root, deduplicated
    assert all(p.parent == tmp_path for p in dirs + files)
    assert all(f.read_bytes() == b"xyz" for f in files)


# ------------------------------------------------------ choose_random_elements

def test_choose_random_elements_returns_requested_counts(sample_tree):
    base, all_dirs, all_files = sample_tree
    dirs, files = choose_random_elements(base, 5, 6)
    assert len(dirs) == 5
    assert len(files) == 6
    assert set(dirs) <= all_dirs
    assert set(files) <= all_files
    assert all(isinstance(p, Path) for p in dirs + files)


def test_choose_random_elements_allows_duplicates(tmp_path):
    (tmp_path / "only_dir").mkdir()
    (tmp_path / "only_file").touch()
    dirs, files = choose_random_elements(tmp_path, 4, 4)
    assert dirs == [tmp_path / "only_dir"] * 4
    assert files == [tmp_path / "only_file"] * 4


def test_choose_random_elements_zero_on_empty_dir(tmp_path):
    assert choose_random_elements(tmp_path, 0, 0) == ([], [])


def test_choose_random_elements_empty_raises_or_ignores(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0)
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1)
    assert choose_random_elements(tmp_path, 2, 2, onfail="ignore") == ([], [])


def test_choose_random_elements_unknown_onfail_for_files_raises(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="bogus")


# ------------------------------------------------------ sample_random_elements

def test_sample_random_elements_distinct_and_valid(sample_tree):
    base, all_dirs, all_files = sample_tree
    random.seed(1234)
    dirs, files = sample_random_elements(base, 2, 3)
    assert len(dirs) == 2 and len(set(dirs)) == 2
    assert len(files) == 3 and len(set(files)) == 3
    assert set(dirs) <= all_dirs
    assert set(files) <= all_files


def test_sample_random_elements_all_available(sample_tree):
    base, all_dirs, all_files = sample_tree
    dirs, files = sample_random_elements(base, 3, 4)
    assert set(dirs) == all_dirs
    assert set(files) == all_files
    assert len(dirs) == 3 and len(files) == 4


def test_sample_random_elements_too_many_raises(sample_tree):
    base, _, _ = sample_tree
    with pytest.raises(ValueError):
        sample_random_elements(base, 4, 0)
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 5)


def test_sample_random_elements_ignore_returns_fewer(sample_tree):
    base, all_dirs, all_files = sample_tree
    dirs, files = sample_random_elements(base, 10, 20, onfail="ignore")
    assert set(dirs) == all_dirs and len(dirs) == len(all_dirs)
    assert set(files) == all_files and len(files) == len(all_files)


def test_sample_random_elements_unknown_onfail_raises(sample_tree):
    base, _, _ = sample_tree
    with pytest.raises(ValueError):
        sample_random_elements(base, 10, 0, onfail="bogus")
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 10, onfail="bogus")


def test_sample_random_elements_zero_on_empty_dir(tmp_path):
    assert sample_random_elements(tmp_path, 0, 0) == ([], [])
