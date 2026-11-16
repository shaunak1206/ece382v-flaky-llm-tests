import itertools
import random
from pathlib import Path

import pytest

import randomfiletree.core as core
from randomfiletree.core import (
    choose_random_elements,
    iterative_gaussian_tree,
    iterative_tree,
    sample_random_elements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tree(root):
    """Create a fixed tree:
    root/d1/sub/, root/d2/, root/f1.txt, root/d1/f2.txt, root/d1/sub/f3.txt
    """
    d1 = root / "d1"
    (d1 / "sub").mkdir(parents=True)
    (root / "d2").mkdir()
    (root / "f1.txt").write_text("a")
    (d1 / "f2.txt").write_text("b")
    (d1 / "sub" / "f3.txt").write_text("c")
    return root


def expected_dirs(root):
    return {root / "d1", root / "d2", root / "d1" / "sub"}


def expected_files(root):
    return {root / "f1.txt", root / "d1" / "f2.txt", root / "d1" / "sub" / "f3.txt"}


def sequential_names(prefix="file"):
    counter = itertools.count()
    return lambda: "%s%d.txt" % (prefix, next(counter))


def zero(depth):
    return 0


# ---------------------------------------------------------------------------
# iterative_tree
# ---------------------------------------------------------------------------

def test_iterative_tree_creates_folders_at_root(tmp_path):
    base = tmp_path / "out"
    dirs, files = iterative_tree(
        base,
        nfolders_func=lambda depth: 2 if depth == 0 else 0,
        nfiles_func=zero,
    )
    assert len(dirs) == 2
    assert all(d.parent == base for d in dirs)
    assert files == []


def test_iterative_tree_created_dirs_exist_inside_basedir(tmp_path):
    base = tmp_path / "out"
    dirs, _ = iterative_tree(
        base,
        nfolders_func=lambda depth: 3 if depth == 0 else 0,
        nfiles_func=zero,
    )
    assert all(d.is_dir() for d in dirs)
    assert all(base in d.parents for d in dirs)


def test_iterative_tree_creates_empty_files_with_given_names(tmp_path):
    base = tmp_path / "out"
    dirs, files = iterative_tree(
        base,
        nfolders_func=zero,
        nfiles_func=lambda depth: 2 if depth == 0 else 0,
        filename=sequential_names("file"),
    )
    assert dirs == []
    assert {f.name for f in files} == {"file0.txt", "file1.txt"}
    for f in files:
        assert f.is_file()
        assert f.stat().st_size == 0


def test_iterative_tree_duplicate_names_collapse(tmp_path):
    base = tmp_path / "out"
    _, files = iterative_tree(
        base,
        nfolders_func=zero,
        nfiles_func=lambda depth: 3 if depth == 0 else 0,
        filename=lambda: "same.txt",
    )
    assert len(files) == 1
    assert files[0].name == "same.txt"
    assert files[0].exists()


def test_iterative_tree_repeat_creates_nested_folders(tmp_path):
    base = tmp_path / "out"
    dirs, _ = iterative_tree(
        base,
        nfolders_func=lambda depth: 1,
        nfiles_func=zero,
        repeat=3,
    )
    depths = [len(d.relative_to(base).parts) for d in dirs]
    assert max(depths) >= 2


def test_iterative_tree_maxdepth_limits_descent(tmp_path):
    base = tmp_path / "out"
    dirs, _ = iterative_tree(
        base,
        nfolders_func=lambda depth: 1,
        nfiles_func=zero,
        repeat=4,
        maxdepth=1,
    )
    assert len(dirs) == 4
    assert all(d.parent == base for d in dirs)


def test_iterative_tree_payload_writes_files(tmp_path):
    base = tmp_path / "out"

    def payload(directory):
        for i in range(2):
            path = directory / ("p%d.dat" % i)
            path.write_text("content")
            yield path

    _, files = iterative_tree(
        base,
        nfolders_func=zero,
        nfiles_func=lambda depth: 2 if depth == 0 else 0,
        payload=payload,
    )
    assert set(files) == {base / "p0.dat", base / "p1.dat"}
    for f in files:
        assert f.read_text() == "content"


def test_iterative_tree_payload_overrides_filename(tmp_path):
    base = tmp_path / "out"

    def payload(directory):
        path = directory / "from_payload.txt"
        path.write_text("x")
        yield path

    _, files = iterative_tree(
        base,
        nfolders_func=zero,
        nfiles_func=lambda depth: 1 if depth == 0 else 0,
        filename=lambda: "ignored.txt",
        payload=payload,
    )
    assert files == [base / "from_payload.txt"]
    assert not (base / "ignored.txt").exists()


def test_iterative_tree_creates_missing_basedir_and_accepts_str(tmp_path):
    base = tmp_path / "x" / "y"
    dirs, files = iterative_tree(
        str(base),
        nfolders_func=lambda depth: 1 if depth == 0 else 0,
        nfiles_func=zero,
    )
    assert base.is_dir()
    assert len(dirs) == 1
    assert isinstance(dirs[0], Path)
    assert dirs[0].is_dir()
    assert files == []


# ---------------------------------------------------------------------------
# iterative_gaussian_tree
# ---------------------------------------------------------------------------

def test_gaussian_passes_mean_and_sigma_to_random_gauss(monkeypatch, tmp_path):
    calls = []

    def fake_gauss(mu, sigma):
        calls.append((mu, sigma))
        return 0.0

    monkeypatch.setattr(core.random, "gauss", fake_gauss)
    iterative_gaussian_tree(
        tmp_path / "t",
        nfiles=4,
        nfolders=3,
        sigma_files=0.5,
        sigma_folders=2.5,
    )
    assert (4, 0.5) in calls
    assert (3, 2.5) in calls


def test_gaussian_min_values_clamp_negative_draws(monkeypatch, tmp_path):
    monkeypatch.setattr(core.random, "gauss", lambda mu, sigma: -5.0)
    dirs, files = iterative_gaussian_tree(
        tmp_path / "t",
        min_folders=2,
        min_files=1,
    )
    assert len(dirs) == 2
    assert len(files) == 1


def test_gaussian_truncates_draws_to_int(monkeypatch, tmp_path):
    monkeypatch.setattr(core.random, "gauss", lambda mu, sigma: 2.9)
    dirs, files = iterative_gaussian_tree(
        tmp_path / "t",
        nfiles=2,
        nfolders=2,
        filename=sequential_names("g"),
    )
    assert len(files) == 2
    assert len(dirs) == 2


# ---------------------------------------------------------------------------
# choose_random_elements
# ---------------------------------------------------------------------------

def test_choose_returns_requested_counts_from_tree(tmp_path):
    base = build_tree(tmp_path / "t")
    dirs, files = choose_random_elements(base, 5, 4)
    assert len(dirs) == 5
    assert len(files) == 4
    assert set(dirs) <= expected_dirs(base)
    assert set(files) <= expected_files(base)


def test_choose_allows_repeated_picks(tmp_path):
    base = tmp_path / "t"
    (base / "only").mkdir(parents=True)
    (base / "one.txt").write_text("x")
    dirs, files = choose_random_elements(base, 3, 3)
    assert dirs == [base / "only"] * 3
    assert files == [base / "one.txt"] * 3


def test_choose_zero_counts_on_empty_tree_returns_empty(tmp_path):
    base = tmp_path / "empty"
    base.mkdir()
    assert choose_random_elements(base, 0, 0) == ([], [])


def test_choose_raises_without_dirs_by_default(tmp_path):
    base = tmp_path / "files_only"
    base.mkdir()
    (base / "a.txt").write_text("")
    with pytest.raises(ValueError):
        choose_random_elements(base, 1, 0)


def test_choose_ignore_without_files_returns_empty_files(tmp_path):
    base = tmp_path / "dirs_only"
    (base / "d").mkdir(parents=True)
    dirs, files = choose_random_elements(base, 2, 1, onfail="ignore")
    assert dirs == [base / "d"] * 2
    assert files == []


def test_choose_unknown_onfail_without_files_raises(tmp_path):
    base = tmp_path / "dirs_only"
    (base / "d").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown"):
        choose_random_elements(base, 0, 1, onfail="bogus")


# ---------------------------------------------------------------------------
# sample_random_elements
# ---------------------------------------------------------------------------

def test_sample_returns_distinct_elements_from_tree(tmp_path):
    base = build_tree(tmp_path / "t")
    dirs, files = sample_random_elements(base, 3, 3)
    assert len(dirs) == 3 and len(set(dirs)) == 3
    assert len(files) == 3 and len(set(files)) == 3
    assert set(dirs) <= expected_dirs(base)
    assert set(files) <= expected_files(base)


def test_sample_raises_when_too_few_dirs(tmp_path):
    base = build_tree(tmp_path / "t")
    with pytest.raises(ValueError):
        sample_random_elements(base, 4, 0)


def test_sample_raises_when_too_few_files(tmp_path):
    base = build_tree(tmp_path / "t")
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 4)


def test_sample_ignore_returns_all_available(tmp_path):
    base = build_tree(tmp_path / "t")
    dirs, files = sample_random_elements(base, 10, 10, onfail="ignore")
    assert len(dirs) == 3 and set(dirs) == expected_dirs(base)
    assert len(files) == 3 and set(files) == expected_files(base)


def test_sample_unknown_onfail_raises_for_dirs_and_files(tmp_path):
    base = build_tree(tmp_path / "t")
    with pytest.raises(ValueError):
        sample_random_elements(base, 10, 0, onfail="bogus")
    with pytest.raises(ValueError):
        sample_random_elements(base, 0, 10, onfail="bogus")


def test_sample_is_reproducible_with_same_seed(tmp_path):
    base = build_tree(tmp_path / "t")
    random.seed(7)
    first = sample_random_elements(base, 2, 2)
    random.seed(7)
    second = sample_random_elements(base, 2, 2)
    assert first == second
