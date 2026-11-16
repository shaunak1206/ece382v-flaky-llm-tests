import itertools
import random
from pathlib import Path

import pytest

from randomfiletree.core import (
    iterative_gaussian_tree,
    iterative_tree,
    choose_random_elements,
    sample_random_elements,
)


def _counter_filename(prefix="f"):
    """Return a filename callable that yields unique names."""
    counter = itertools.count()
    return lambda: "{}{}.txt".format(prefix, next(counter))


def _build_sample_tree(base):
    """
    Create a fixed tree:
        base/sub1/deep/
        base/sub2/
        base/top.txt
        base/sub1/one.txt
        base/sub1/deep/two.txt
    """
    (base / "sub1" / "deep").mkdir(parents=True)
    (base / "sub2").mkdir()
    (base / "top.txt").write_text("t")
    (base / "sub1" / "one.txt").write_text("o")
    (base / "sub1" / "deep" / "two.txt").write_text("w")
    return base


def _expected_entries(base):
    dirs = {base / "sub1", base / "sub2", base / "sub1" / "deep"}
    files = {
        base / "top.txt",
        base / "sub1" / "one.txt",
        base / "sub1" / "deep" / "two.txt",
    }
    return dirs, files


# ---------------------------------------------------------------------------
# iterative_tree
# ---------------------------------------------------------------------------

def test_iterative_tree_creates_missing_basedir(tmp_path):
    base = tmp_path / "new" / "nested"
    iterative_tree(base, lambda depth: 0, lambda depth: 0)
    assert base.is_dir()


def test_single_pass_creates_folders_only_in_basedir(tmp_path):
    base = tmp_path / "t"
    dirs, files = iterative_tree(
        base,
        lambda depth: 2 if depth == 0 else 0,
        lambda depth: 0,
    )
    assert files == []
    assert len(dirs) == 2
    assert all(isinstance(p, Path) for p in dirs)
    assert all(p.parent == base and p.is_dir() for p in dirs)


def test_repeat_populates_folders_created_in_earlier_pass(tmp_path):
    # Every walked directory creates one subfolder. With repeat=2:
    # pass 1 creates A in base; pass 2 creates B in base and C inside A.
    dirs, _ = iterative_tree(
        tmp_path,
        lambda depth: 1,
        lambda depth: 0,
        repeat=2,
    )
    assert len(dirs) == 3
    nested = [p for p in dirs if p.parent != tmp_path]
    assert len(nested) == 1
    assert nested[0].parent.parent == tmp_path


def test_files_are_created_empty_with_expected_count(tmp_path):
    dirs, files = iterative_tree(
        tmp_path,
        lambda depth: 0,
        lambda depth: 2,
        filename=_counter_filename(),
    )
    assert dirs == []
    assert len(files) == 2
    for p in files:
        assert isinstance(p, Path)
        assert p.parent == tmp_path
        assert p.is_file()
        assert p.stat().st_size == 0


def test_payload_generator_is_used_and_overrides_filename(tmp_path):
    def payload(directory):
        def gen():
            i = 0
            while True:
                p = directory / "payload_{}.bin".format(i)
                p.write_bytes(b"data")
                i += 1
                yield p
        return gen()

    def filename():
        raise AssertionError("filename must not be used when payload is given")

    dirs, files = iterative_tree(
        tmp_path,
        lambda depth: 0,
        lambda depth: 3,
        filename=filename,
        payload=payload,
    )
    assert dirs == []
    assert len(files) == 3
    assert all(p.parent == tmp_path for p in files)
    assert all(p.read_bytes() == b"data" for p in files)


def test_depth_passed_to_callables_matches_existing_tree(tmp_path):
    # Depth is the count of separators in the path relative to basedir:
    # base (0), base/a (0), base/a/b (1), base/a/b/c (2).
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 0

    iterative_tree(tmp_path, nfolders, lambda depth: 0)
    assert sorted(depths) == [0, 0, 1, 2]


def test_maxdepth_prevents_descending_into_deeper_folders(tmp_path):
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 0

    iterative_tree(tmp_path, nfolders, lambda depth: 0, maxdepth=2)
    # base (0), a (0), a/b (1) are visited; a/b/c (depth 2) is pruned.
    assert sorted(depths) == [0, 0, 1]


# ---------------------------------------------------------------------------
# iterative_gaussian_tree
# ---------------------------------------------------------------------------

def test_gaussian_zero_sigma_creates_exact_counts(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=3,
        nfolders=2,
        sigma_files=0,
        sigma_folders=0,
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(p.parent == tmp_path for p in dirs + files)


def test_gaussian_passes_mean_and_sigma_to_random_gauss(tmp_path, monkeypatch):
    calls = []

    def fake_gauss(mu, sigma):
        calls.append((mu, sigma))
        return mu

    monkeypatch.setattr(random, "gauss", fake_gauss)

    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=4,
        nfolders=2,
        sigma_files=0.5,
        sigma_folders=0.25,
    )
    assert sorted(calls) == sorted([(2, 0.25), (4, 0.5)])
    assert len(dirs) == 2
    assert len(files) == 4


def test_gaussian_min_files_clamps_low_draws(tmp_path):
    dirs, files = iterative_gaussian_tree(
        tmp_path,
        nfiles=-3,
        nfolders=0,
        sigma_files=0,
        sigma_folders=0,
        min_files=2,
    )
    assert dirs == []
    assert len(files) == 2


# ---------------------------------------------------------------------------
# choose_random_elements
# ---------------------------------------------------------------------------

def test_choose_returns_requested_counts_with_replacement(tmp_path):
    base = _build_sample_tree(tmp_path)
    expected_dirs, expected_files = _expected_entries(base)

    dirs, files = choose_random_elements(base, 10, 10)

    assert len(dirs) == 10
    assert len(files) == 10
    assert set(dirs) <= expected_dirs
    assert set(files) <= expected_files
    assert all(isinstance(p, Path) for p in dirs + files)


def test_choose_can_repeat_single_file(tmp_path):
    f = tmp_path / "only.txt"
    f.write_text("x")

    dirs, files = choose_random_elements(tmp_path, 0, 4)

    assert dirs == []
    assert files == [f] * 4


def test_choose_zero_requests_on_empty_tree_returns_empty_lists(tmp_path):
    assert choose_random_elements(tmp_path, 0, 0) == ([], [])


def test_choose_draws_from_all_nested_levels(tmp_path):
    base = _build_sample_tree(tmp_path)
    expected_dirs, expected_files = _expected_entries(base)

    random.seed(1234)
    dirs, files = choose_random_elements(base, 300, 300)

    assert set(dirs) == expected_dirs
    assert set(files) == expected_files


def test_choose_raises_when_no_dirs_and_onfail_raise(tmp_path):
    (tmp_path / "only.txt").write_text("x")
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 1, 0, onfail="raise")


def test_choose_raises_when_no_files_and_onfail_raise(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="raise")


def test_choose_ignore_returns_empty_list_for_missing_kind(tmp_path):
    f = tmp_path / "only.txt"
    f.write_text("x")

    dirs, files = choose_random_elements(tmp_path, 2, 3, onfail="ignore")

    assert dirs == []
    assert len(files) == 3
    assert all(p == f for p in files)


def test_choose_unknown_onfail_raises_when_files_missing(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError):
        choose_random_elements(tmp_path, 0, 1, onfail="bogus")


# ---------------------------------------------------------------------------
# sample_random_elements
# ---------------------------------------------------------------------------

def test_sample_returns_distinct_elements(tmp_path):
    base = _build_sample_tree(tmp_path)
    expected_dirs, expected_files = _expected_entries(base)

    dirs, files = sample_random_elements(base, 2, 2)

    assert len(dirs) == 2 and len(set(dirs)) == 2
    assert len(files) == 2 and len(set(files)) == 2
    assert set(dirs) <= expected_dirs
    assert set(files) <= expected_files


def test_sample_raises_when_too_few_dirs(tmp_path):
    base = _build_sample_tree(tmp_path)
    with pytest.raises(ValueError):
        sample_random_elements(base, 4, 0)


def test_sample_raises_when_too_few_files(tmp_path):
    base = _build_sample_tree(tmp_path)
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 4)


def test_sample_ignore_returns_all_available_elements(tmp_path):
    base = _build_sample_tree(tmp_path)
    expected_dirs, expected_files = _expected_entries(base)

    dirs, files = sample_random_elements(base, 10, 10, onfail="ignore")

    assert len(dirs) == 3 and set(dirs) == expected_dirs
    assert len(files) == 3 and set(files) == expected_files
