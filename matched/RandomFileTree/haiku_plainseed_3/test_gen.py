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


def make_namer(prefix="f"):
    """Return a callable producing unique, deterministic file names."""
    counter = itertools.count()
    return lambda: "{}{}.txt".format(prefix, next(counter))


def is_under(path, base):
    return base in Path(path).parents


@pytest.fixture
def sample_tree(tmp_path):
    """
    Layout (basedir itself is not part of the expected results):
        tree/
          a/            (dir)
            b/          (dir)
              file1.txt
            file2.txt
          c/            (dir)
    """
    root = tmp_path / "tree"
    (root / "a" / "b").mkdir(parents=True)
    (root / "c").mkdir()
    (root / "a" / "file2.txt").write_text("x")
    (root / "a" / "b" / "file1.txt").write_text("y")
    return root


@pytest.fixture
def expected_sample_tree(sample_tree):
    dirs = {sample_tree / "a", sample_tree / "a" / "b", sample_tree / "c"}
    files = {sample_tree / "a" / "file2.txt", sample_tree / "a" / "b" / "file1.txt"}
    return dirs, files


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def files_only_dir(tmp_path):
    d = tmp_path / "files_only"
    d.mkdir()
    (d / "a.txt").write_text("content")
    return d


@pytest.fixture
def dirs_only_dir(tmp_path):
    d = tmp_path / "dirs_only"
    (d / "x" / "y").mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# iterative_tree
# ---------------------------------------------------------------------------

def test_iterative_tree_creates_missing_basedir_from_str(tmp_path):
    basedir = tmp_path / "new" / "nested"
    dirs, files = iterative_tree(str(basedir), lambda d: 0, lambda d: 0)
    assert basedir.is_dir()
    assert dirs == []
    assert files == []


def test_iterative_tree_creates_empty_files_by_default(tmp_path):
    basedir = tmp_path / "t"
    dirs, files = iterative_tree(
        basedir, lambda d: 0, lambda d: 3, filename=make_namer("f")
    )
    assert dirs == []
    assert len(files) == 3
    for f in files:
        assert isinstance(f, Path)
        assert f.parent == basedir
        assert f.is_file()
        assert f.stat().st_size == 0


def test_iterative_tree_creates_requested_folders(tmp_path):
    basedir = tmp_path / "t"
    dirs, files = iterative_tree(basedir, lambda d: 2, lambda d: 0)
    assert files == []
    assert len(dirs) == 2
    for d in dirs:
        assert isinstance(d, Path)
        assert d.parent == basedir
        assert d.is_dir()


def test_iterative_tree_deduplicates_repeated_filenames(tmp_path):
    basedir = tmp_path / "t"
    dirs, files = iterative_tree(
        basedir, lambda d: 0, lambda d: 3, filename=lambda: "same.txt"
    )
    assert files == [basedir / "same.txt"]


def test_iterative_tree_maxdepth_one_prevents_descent(tmp_path):
    depths = []

    def nfolders(depth):
        depths.append(depth)
        return 1

    basedir = tmp_path / "t"
    iterative_tree(basedir, nfolders, lambda d: 0, repeat=3, maxdepth=1)
    # Only the root directory is ever visited, once per repeat.
    assert depths == [0, 0, 0]


def test_iterative_tree_payload_creates_files_and_overrides_filename(tmp_path):
    basedir = tmp_path / "t"

    def payload(directory):
        for i in itertools.count():
            p = directory / "payload_{}.bin".format(i)
            p.write_bytes(b"data")
            yield p

    def forbidden_filename():
        raise AssertionError("filename must not be used when payload is given")

    dirs, files = iterative_tree(
        basedir,
        lambda d: 0,
        lambda d: 2,
        filename=forbidden_filename,
        payload=payload,
    )
    assert dirs == []
    assert sorted(f.name for f in files) == ["payload_0.bin", "payload_1.bin"]
    for f in files:
        assert f.parent == basedir
        assert f.read_bytes() == b"data"


def test_iterative_tree_payload_called_once_per_visited_directory(tmp_path):
    basedir = tmp_path / "t"
    received = []

    def payload(directory):
        received.append(directory)
        return iter(())

    iterative_tree(basedir, lambda d: 0, lambda d: 0, payload=payload)
    assert received == [basedir]


# ---------------------------------------------------------------------------
# iterative_gaussian_tree
# ---------------------------------------------------------------------------

def test_gaussian_passes_mean_and_sigma_to_random_gauss(tmp_path, monkeypatch):
    calls = []

    def fake_gauss(mu, sigma):
        calls.append((mu, sigma))
        return mu

    monkeypatch.setattr(random, "gauss", fake_gauss)
    dirs, files = iterative_gaussian_tree(
        tmp_path / "g",
        nfiles=2,
        nfolders=1,
        sigma_files=3,
        sigma_folders=4,
        filename=make_namer(),
    )
    assert (1, 4) in calls
    assert (2, 3) in calls
    assert len(dirs) == 1
    assert len(files) == 2


def test_gaussian_clamps_negative_samples_to_minimums(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: -3.7)
    dirs, files = iterative_gaussian_tree(
        tmp_path / "g",
        min_folders=1,
        min_files=2,
        filename=make_namer(),
    )
    assert len(dirs) == 1
    assert len(files) == 2


def test_gaussian_truncates_float_samples_toward_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: 2.9)
    dirs, files = iterative_gaussian_tree(
        tmp_path / "g", filename=make_namer()
    )
    assert len(dirs) == 2
    assert len(files) == 2


def test_gaussian_real_random_creates_existing_paths(tmp_path):
    random.seed(0)
    basedir = tmp_path / "r"
    dirs, files = iterative_gaussian_tree(
        basedir,
        nfiles=3,
        nfolders=2,
        repeat=2,
        filename=make_namer(),
    )
    assert isinstance(dirs, list)
    assert isinstance(files, list)
    for d in dirs:
        assert isinstance(d, Path)
        assert d.is_dir()
        assert is_under(d, basedir)
    for f in files:
        assert isinstance(f, Path)
        assert f.is_file()
        assert is_under(f, basedir)


# ---------------------------------------------------------------------------
# choose_random_elements
# ---------------------------------------------------------------------------

def test_choose_returns_only_existing_entries(sample_tree, expected_sample_tree):
    expected_dirs, expected_files = expected_sample_tree
    dirs, files = choose_random_elements(sample_tree, 5, 5)
    assert len(dirs) == 5
    assert len(files) == 5
    assert all(isinstance(p, Path) for p in dirs + files)
    assert set(dirs) <= expected_dirs
    assert set(files) <= expected_files


def test_choose_allows_repetition_when_more_requested_than_available(tmp_path):
    root = tmp_path / "one"
    root.mkdir()
    only = root / "f.txt"
    only.write_text("x")
    dirs, files = choose_random_elements(root, 0, 4)
    assert dirs == []
    assert files == [only, only, only, only]


def test_choose_zero_counts_on_empty_tree_does_not_raise(empty_dir):
    assert choose_random_elements(empty_dir, 0, 0) == ([], [])


def test_choose_raises_when_dirs_requested_but_none_exist(files_only_dir):
    with pytest.raises(ValueError):
        choose_random_elements(files_only_dir, 1, 0)


def test_choose_ignore_returns_empty_dirs_when_none_exist(files_only_dir):
    dirs, files = choose_random_elements(files_only_dir, 2, 1, onfail="ignore")
    assert dirs == []
    assert len(files) == 1
    assert files[0] == files_only_dir / "a.txt"


def test_choose_raises_when_files_requested_but_none_exist(dirs_only_dir):
    with pytest.raises(ValueError):
        choose_random_elements(dirs_only_dir, 0, 1)


def test_choose_unknown_onfail_raises_when_no_files(dirs_only_dir):
    with pytest.raises(ValueError):
        choose_random_elements(dirs_only_dir, 0, 1, onfail="bogus")


# ---------------------------------------------------------------------------
# sample_random_elements
# ---------------------------------------------------------------------------

def test_sample_returns_distinct_existing_entries(sample_tree, expected_sample_tree):
    expected_dirs, expected_files = expected_sample_tree
    dirs, files = sample_random_elements(sample_tree, 2, 2)
    assert len(dirs) == 2 and len(set(dirs)) == 2
    assert len(files) == 2 and len(set(files)) == 2
    assert set(dirs) <= expected_dirs
    assert set(files) <= expected_files


def test_sample_full_selection_returns_everything(sample_tree, expected_sample_tree):
    expected_dirs, expected_files = expected_sample_tree
    dirs, files = sample_random_elements(sample_tree, 3, 2)
    assert set(dirs) == expected_dirs
    assert set(files) == expected_files


def test_sample_raises_when_not_enough_dirs(sample_tree):
    with pytest.raises(ValueError):
        sample_random_elements(sample_tree, 4, 0)


def test_sample_raises_when_not_enough_files(sample_tree):
    with pytest.raises(ValueError):
        sample_random_elements(sample_tree, 0, 3)


def test_sample_ignore_returns_all_available(sample_tree, expected_sample_tree):
    expected_dirs, expected_files = expected_sample_tree
    dirs, files = sample_random_elements(sample_tree, 10, 10, onfail="ignore")
    assert len(dirs) == len(expected_dirs)
    assert len(files) == len(expected_files)
    assert set(dirs) == expected_dirs
    assert set(files) == expected_files


def test_sample_raises_on_empty_tree_when_files_requested(empty_dir):
    with pytest.raises(ValueError):
        sample_random_elements(empty_dir, 0, 1)
