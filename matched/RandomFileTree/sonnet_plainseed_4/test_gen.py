import itertools
import random
from pathlib import Path

import pytest

from randomfiletree.core import (
    choose_random_elements,
    iterative_gaussian_tree,
    iterative_tree,
    sample_random_elements,
)


def make_counter_namer(prefix="file"):
    counter = itertools.count()
    return lambda: "{}_{}".format(prefix, next(counter))


def payload_generator(directory):
    i = 0
    while True:
        p = directory / "payload_{}.txt".format(i)
        p.write_text("hello")
        i += 1
        yield p


@pytest.fixture
def tree(tmp_path):
    """Fixed tree: 3 dirs (a, a/b, c) and 5 files."""
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "c").mkdir()
    files = [
        tmp_path / "f1.txt",
        tmp_path / "a" / "f2.txt",
        tmp_path / "a" / "b" / "f3.txt",
        tmp_path / "c" / "f4.txt",
        tmp_path / "c" / "f5.txt",
    ]
    for f in files:
        f.write_text("x")
    dirs = [tmp_path / "a", tmp_path / "a" / "b", tmp_path / "c"]
    return tmp_path, dirs, files


# ---------------------------------------------------------------- iterative_tree


def test_iterative_tree_creates_requested_counts(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 2,
        nfiles_func=lambda depth: 3,
        filename=make_counter_namer(),
    )
    assert len(dirs) == 2
    assert len(files) == 3
    for d in dirs:
        assert isinstance(d, Path)
        assert d.is_dir()
        assert d.parent == tmp_path
    for f in files:
        assert isinstance(f, Path)
        assert f.is_file()
        assert f.parent == tmp_path


def test_iterative_tree_creates_missing_basedir_with_str(tmp_path):
    base = tmp_path / "x" / "y" / "z"
    dirs, files = iterative_tree(
        str(base),
        nfolders_func=lambda depth: 1,
        nfiles_func=lambda depth: 1,
        filename=make_counter_namer(),
    )
    assert base.is_dir()
    assert len(dirs) == 1
    assert len(files) == 1
    assert dirs[0].parent == base


def test_iterative_tree_zero_counts(tmp_path):
    base = tmp_path / "empty"
    dirs, files = iterative_tree(
        base, nfolders_func=lambda d: 0, nfiles_func=lambda d: 0
    )
    assert dirs == []
    assert files == []
    assert base.is_dir()
    assert list(base.iterdir()) == []


def test_iterative_tree_default_filename_creates_empty_files(tmp_path):
    dirs, files = iterative_tree(
        tmp_path, nfolders_func=lambda d: 0, nfiles_func=lambda d: 4
    )
    assert len(files) >= 1
    for f in files:
        assert f.is_file()
        assert f.stat().st_size == 0
        assert f.parent == tmp_path


def test_iterative_tree_custom_filename_used(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        nfolders_func=lambda d: 0,
        nfiles_func=lambda d: 3,
        filename=make_counter_namer("custom"),
    )
    assert sorted(f.name for f in files) == ["custom_0", "custom_1", "custom_2"]


def test_iterative_tree_repeat_descends_into_new_folders(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        nfolders_func=lambda d: 1,
        nfiles_func=lambda d: 1,
        repeat=2,
        filename=make_counter_namer(),
    )
    # pass 1: 1 dir + 1 file in base
    # pass 2: 1 dir + 1 file in base, and 1 dir + 1 file in the first dir
    assert len(dirs) == 3
    assert len(files) == 3
    assert len(set(dirs)) == 3
    assert len(set(files)) == 3
    assert any(d.parent != tmp_path for d in dirs)
    assert all(p.exists() for p in dirs + files)


def test_iterative_tree_maxdepth_one_only_populates_base(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        nfolders_func=lambda d: 1,
        nfiles_func=lambda d: 1,
        repeat=3,
        maxdepth=1,
        filename=make_counter_namer(),
    )
    assert len(dirs) == 3
    assert len(files) == 3
    assert all(d.parent == tmp_path for d in dirs)
    assert all(f.parent == tmp_path for f in files)


def test_iterative_tree_depth_function_receives_ints(tmp_path):
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 1

    iterative_tree(tmp_path, nfolders, lambda d: 0, repeat=2)
    assert depths[0] == 0
    assert all(isinstance(d, int) and d >= 0 for d in depths)


def test_iterative_tree_payload_creates_files_with_content(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        nfolders_func=lambda d: 0,
        nfiles_func=lambda d: 3,
        payload=payload_generator,
    )
    assert sorted(f.name for f in files) == [
        "payload_0.txt",
        "payload_1.txt",
        "payload_2.txt",
    ]
    for f in files:
        assert f.read_text() == "hello"


# ------------------------------------------------------- iterative_gaussian_tree


def test_gaussian_tree_zero_sigma_gives_exact_counts(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=4,
        nfolders=2,
        sigma_files=0,
        sigma_folders=0,
        filename=make_counter_namer(),
    )
    assert len(dirs) == 2
    assert len(files) == 4
    assert all(d.is_dir() for d in dirs)
    assert all(f.is_file() for f in files)


def test_gaussian_tree_min_values_enforced(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=-10,
        nfolders=-10,
        sigma_files=0,
        sigma_folders=0,
        min_files=3,
        min_folders=2,
        filename=make_counter_namer(),
    )
    assert len(dirs) == 2
    assert len(files) == 3


def test_gaussian_tree_negative_mean_gives_nothing_by_default(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path, nfiles=-5, nfolders=-5, sigma_files=0, sigma_folders=0
    )
    assert dirs == []
    assert files == []


def test_gaussian_tree_reproducible_with_seed(tmp_path):
    random.seed(1234)
    a = tmp_path / "a"
    dirs_a, files_a = iterative_gaussian_tree(a, nfiles=3, nfolders=2, repeat=2)
    random.seed(1234)
    b = tmp_path / "b"
    dirs_b, files_b = iterative_gaussian_tree(b, nfiles=3, nfolders=2, repeat=2)
    rel_a = sorted(str(p.relative_to(a)) for p in dirs_a + files_a)
    rel_b = sorted(str(p.relative_to(b)) for p in dirs_b + files_b)
    assert rel_a == rel_b


def test_gaussian_tree_maxdepth_and_payload_passed_through(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=2,
        nfolders=1,
        sigma_files=0,
        sigma_folders=0,
        repeat=3,
        maxdepth=1,
        payload=payload_generator,
    )
    assert len(dirs) == 3
    assert all(d.parent == tmp_path for d in dirs)
    assert len(files) == 2
    assert all(f.read_text() == "hello" for f in files)


# ------------------------------------------------------- choose_random_elements


def test_choose_random_elements_returns_existing_items(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = choose_random_elements(base, 4, 6)
    assert len(sel_dirs) == 4
    assert len(sel_files) == 6
    assert all(d in dirs for d in sel_dirs)
    assert all(f in files for f in sel_files)


def test_choose_random_elements_zero_requested_on_empty_dir(tmp_path):
    assert choose_random_elements(tmp_path, 0, 0) == ([], [])


def test_choose_random_elements_raises_when_nothing_to_pick(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0)
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1)


def test_choose_random_elements_ignore_returns_empty(tmp_path):
    sel_dirs, sel_files = choose_random_elements(tmp_path, 2, 2, onfail="ignore")
    assert sel_dirs == []
    assert sel_files == []


def test_choose_random_elements_unknown_onfail_raises_for_files(tmp_path):
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="bogus")


# ------------------------------------------------------- sample_random_elements


def test_sample_random_elements_distinct_and_valid(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, 2, 4)
    assert len(sel_dirs) == 2
    assert len(set(sel_dirs)) == 2
    assert len(sel_files) == 4
    assert len(set(sel_files)) == 4
    assert all(d in dirs for d in sel_dirs)
    assert all(f in files for f in sel_files)


def test_sample_random_elements_can_select_everything(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(base, len(dirs), len(files))
    assert sorted(sel_dirs) == sorted(dirs)
    assert sorted(sel_files) == sorted(files)


def test_sample_random_elements_raises_when_not_enough(tree):
    base, dirs, files = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, len(dirs) + 1, 0)
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, len(files) + 1)


def test_sample_random_elements_ignore_returns_fewer(tree):
    base, dirs, files = tree
    sel_dirs, sel_files = sample_random_elements(
        base, len(dirs) + 5, len(files) + 5, onfail="ignore"
    )
    assert sorted(sel_dirs) == sorted(dirs)
    assert sorted(sel_files) == sorted(files)


def test_sample_random_elements_unknown_onfail_and_zero_on_empty(tmp_path, tree):
    base, dirs, files = tree
    with pytest.raises(ValueError):
        sample_random_elements(base, len(dirs) + 1, 0, onfail="bogus")
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert sample_random_elements(empty, 0, 0) == ([], [])
