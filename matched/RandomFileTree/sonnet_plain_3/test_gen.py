import os
from pathlib import Path

import pytest

from randomfiletree.core import (
    iterative_tree,
    iterative_gaussian_tree,
    choose_random_elements,
    sample_random_elements,
)


@pytest.fixture
def tree(tmp_path):
    """Fixed tree: 3 dirs and 4 files."""
    base = tmp_path / "base"
    dirs = [base / "a", base / "b", base / "a" / "c"]
    for d in dirs:
        d.mkdir(parents=True)
    files = [
        base / "f1.txt",
        base / "a" / "f2.txt",
        base / "a" / "c" / "f3.txt",
        base / "b" / "f4.txt",
    ]
    for f in files:
        f.touch()
    return base, set(dirs), set(files)


def _counter_filename():
    state = {"i": 0}

    def name():
        state["i"] += 1
        return "file_{}.dat".format(state["i"])

    return name


# ---------------------------------------------------------------- iterative_tree

def test_iterative_tree_creates_basedir_and_returns_paths(tmp_path):
    base = tmp_path / "x" / "y" / "z"
    dirs, files = iterative_tree(base, lambda d: 2, lambda d: 3)
    assert base.is_dir()
    assert len(dirs) == 2
    assert len(files) == 3
    for d in dirs:
        assert isinstance(d, Path)
        assert d.is_dir()
    for f in files:
        assert isinstance(f, Path)
        assert f.is_file()
        assert f.stat().st_size == 0


def test_iterative_tree_accepts_string_basedir(tmp_path):
    base = str(tmp_path / "strbase")
    dirs, files = iterative_tree(base, lambda d: 1, lambda d: 1)
    assert len(dirs) == 1
    assert len(files) == 1
    assert dirs[0].parent == Path(base)
    assert files[0].parent == Path(base)


def test_iterative_tree_zero_counts(tmp_path):
    dirs, files = iterative_tree(tmp_path / "empty", lambda d: 0, lambda d: 0)
    assert dirs == []
    assert files == []
    assert list((tmp_path / "empty").iterdir()) == []


def test_iterative_tree_custom_filename(tmp_path):
    dirs, files = iterative_tree(
        tmp_path / "t", lambda d: 0, lambda d: 4, filename=_counter_filename()
    )
    assert dirs == []
    assert sorted(f.name for f in files) == [
        "file_1.dat", "file_2.dat", "file_3.dat", "file_4.dat"
    ]
    for f in files:
        assert f.exists()


def test_iterative_tree_single_repeat_only_populates_base(tmp_path):
    base = tmp_path / "t"
    dirs, files = iterative_tree(base, lambda d: 2, lambda d: 2)
    assert all(d.parent == base for d in dirs)
    assert all(f.parent == base for f in files)


def test_iterative_tree_repeat_descends_into_new_folders(tmp_path):
    base = tmp_path / "t"
    dirs, files = iterative_tree(
        base, lambda d: 1, lambda d: 1, repeat=2, filename=_counter_filename()
    )
    # pass 1: base gets 1 dir + 1 file; pass 2: base and the new subdir each
    # get 1 dir + 1 file.
    assert len(dirs) == 3
    assert len(files) == 3
    assert any(d.parent != base for d in dirs)
    assert any(f.parent != base for f in files)
    for p in dirs + files:
        assert base in p.parents


def test_iterative_tree_maxdepth_one_limits_to_basedir(tmp_path):
    base = tmp_path / "t"
    dirs, files = iterative_tree(
        base, lambda d: 2, lambda d: 2, repeat=4, maxdepth=1
    )
    assert dirs and files
    assert all(d.parent == base for d in dirs)
    assert all(f.parent == base for f in files)


def test_iterative_tree_nfuncs_receive_depth(tmp_path):
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 1

    iterative_tree(tmp_path / "t", nfolders, lambda d: 0)
    assert depths == [0]


def test_iterative_tree_payload_creates_files_with_content(tmp_path):
    def payload(directory):
        i = 0
        while True:
            i += 1
            p = directory / "payload_{}.txt".format(i)
            p.write_text("hello")
            yield p

    dirs, files = iterative_tree(
        tmp_path / "t", lambda d: 0, lambda d: 3, payload=payload
    )
    assert len(files) == 3
    for f in files:
        assert f.read_text() == "hello"
    assert sorted(f.name for f in files) == [
        "payload_1.txt", "payload_2.txt", "payload_3.txt"
    ]


def test_iterative_tree_payload_overrides_filename(tmp_path):
    def payload(directory):
        p = directory / "from_payload.bin"
        p.write_bytes(b"abc")
        yield p

    def bad_filename():
        raise AssertionError("filename should not be used when payload given")

    dirs, files = iterative_tree(
        tmp_path / "t", lambda d: 0, lambda d: 1,
        filename=bad_filename, payload=payload,
    )
    assert [f.name for f in files] == ["from_payload.bin"]


# ------------------------------------------------------ iterative_gaussian_tree

def test_gaussian_tree_exact_counts_with_zero_sigma(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path / "g", nfiles=3, nfolders=2, sigma_files=0, sigma_folders=0
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(d.is_dir() for d in dirs)
    assert all(f.is_file() for f in files)


def test_gaussian_tree_min_files_and_folders_enforced(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path / "g",
        nfiles=0, nfolders=0,
        sigma_files=0, sigma_folders=0,
        min_files=4, min_folders=2,
    )
    assert len(dirs) == 2
    assert len(files) == 4


def test_gaussian_tree_negative_mean_clamped_to_zero(tmp_path):
    base = tmp_path / "g"
    dirs, files = iterative_gaussian_tree(
        base, nfiles=-5, nfolders=-5, sigma_files=0, sigma_folders=0
    )
    assert dirs == []
    assert files == []
    assert base.is_dir()


def test_gaussian_tree_maxdepth_and_filename(tmp_path):
    base = tmp_path / "g"
    dirs, files = iterative_gaussian_tree(
        base, nfiles=2, nfolders=2, sigma_files=0, sigma_folders=0,
        repeat=3, maxdepth=1, filename=_counter_filename(),
    )
    assert all(d.parent == base for d in dirs)
    assert all(f.parent == base for f in files)
    assert all(f.name.startswith("file_") for f in files)


# ------------------------------------------------------- choose_random_elements

def test_choose_returns_requested_number_from_existing(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = choose_random_elements(base, 5, 6)
    assert len(sel_dirs) == 5
    assert len(sel_files) == 6
    assert set(sel_dirs) <= dirs
    assert set(sel_files) <= files


def test_choose_allows_duplicates(tmp_path):
    (tmp_path / "only.txt").touch()
    (tmp_path / "sub").mkdir()
    sel_dirs, sel_files = choose_random_elements(tmp_path, 4, 4)
    assert sel_dirs == [tmp_path / "sub"] * 4
    assert sel_files == [tmp_path / "only.txt"] * 4


def test_choose_zero_requested_on_empty_dir(tmp_path):
    assert choose_random_elements(tmp_path, 0, 0) == ([], [])


def test_choose_raises_when_nothing_to_pick(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0)
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1)


def test_choose_onfail_ignore_returns_empty(tmp_path):
    sel_dirs, sel_files = choose_random_elements(tmp_path, 2, 2, onfail="ignore")
    assert sel_dirs == []
    assert sel_files == []


# ------------------------------------------------------ sample_random_elements

def test_sample_returns_distinct_existing_elements(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, 2, 3)
    assert len(sel_dirs) == 2 and len(set(sel_dirs)) == 2
    assert len(sel_files) == 3 and len(set(sel_files)) == 3
    assert set(sel_dirs) <= dirs
    assert set(sel_files) <= files


def test_sample_all_elements_when_exact_count(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, len(dirs), len(files))
    assert set(sel_dirs) == dirs
    assert set(sel_files) == files


def test_sample_raises_when_not_enough(tree):
    base, dirs, files = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, len(dirs) + 1, 0)
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, len(files) + 1)


def test_sample_onfail_ignore_returns_all_available(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(
        base, len(dirs) + 5, len(files) + 5, onfail="ignore"
    )
    assert set(sel_dirs) == dirs
    assert set(sel_files) == files
    assert len(sel_dirs) == len(dirs)
    assert len(sel_files) == len(files)


def test_sample_unknown_onfail_raises_value_error(tree):
    base, dirs, files = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, len(dirs) + 1, 0, onfail="bogus")
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, len(files) + 1, onfail="bogus")


def test_sample_zero_requested_on_empty_dir(tmp_path):
    assert sample_random_elements(tmp_path, 0, 0) == ([], [])
