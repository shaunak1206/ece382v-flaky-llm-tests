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


def _counting_namer(prefix="f"):
    """Return a filename callable that yields unique, deterministic names."""
    counter = itertools.count()

    def _name():
        return "{}{}".format(prefix, next(counter))

    return _name


def _const(value):
    """Return a count function (accepting any args) that always gives value."""
    return lambda *args: value


@pytest.fixture
def sample_tree(tmp_path):
    """
    Layout (relative to root):
        d1/
        d1/d1a/
        d2/
        f1.txt
        d1/f2.txt
        d1/d1a/f3.txt
        d2/f4.txt
    """
    root = tmp_path / "root"
    (root / "d1" / "d1a").mkdir(parents=True)
    (root / "d2").mkdir()
    (root / "f1.txt").write_text("a")
    (root / "d1" / "f2.txt").write_text("b")
    (root / "d1" / "d1a" / "f3.txt").write_text("c")
    (root / "d2" / "f4.txt").write_text("d")
    return root


def _expected_dirs(root):
    return {root / "d1", root / "d1" / "d1a", root / "d2"}


def _expected_files(root):
    return {
        root / "f1.txt",
        root / "d1" / "f2.txt",
        root / "d1" / "d1a" / "f3.txt",
        root / "d2" / "f4.txt",
    }


# ---------------------------------------------------------------------------
# iterative_tree
# ---------------------------------------------------------------------------

def test_creates_missing_basedir(tmp_path):
    base = tmp_path / "new" / "nested"
    dirs, files = iterative_tree(base, _const(0), _const(0))
    assert base.is_dir()
    assert dirs == []
    assert files == []


def test_repeat_one_creates_only_top_level_entries(tmp_path):
    base = tmp_path / "base"
    dirs, files = iterative_tree(
        base, _const(2), _const(3), repeat=1, filename=_counting_namer()
    )
    assert len(dirs) == 2
    assert len(files) == 3
    assert all(p.parent == base for p in dirs + files)


def test_returned_paths_exist_on_disk(tmp_path):
    base = tmp_path / "base"
    dirs, files = iterative_tree(
        base, _const(1), _const(1), repeat=3, filename=_counting_namer()
    )
    assert dirs and files
    assert all(d.is_dir() for d in dirs)
    assert all(f.is_file() for f in files)


def test_repeat_descends_into_existing_folders(tmp_path):
    base = tmp_path / "base"
    # Pass 1 creates A in base. Pass 2 creates B in base and C inside A.
    dirs, files = iterative_tree(base, _const(1), _const(0), repeat=2)
    assert len(dirs) == 3
    assert files == []
    assert sum(1 for d in dirs if d.parent == base) == 2
    nested = [d for d in dirs if d.parent != base]
    assert len(nested) == 1
    assert nested[0].parent.parent == base


def test_maxdepth_one_never_descends(tmp_path):
    base = tmp_path / "base"
    dirs, files = iterative_tree(
        base, _const(2), _const(1), repeat=4, maxdepth=1,
        filename=_counting_namer(),
    )
    # Each of the 4 passes only visits the base directory.
    assert len(dirs) == 8
    assert len(files) == 4
    assert all(p.parent == base for p in dirs + files)


def test_maxdepth_limits_nesting(tmp_path):
    base = tmp_path / "base"
    dirs, _ = iterative_tree(base, _const(1), _const(0), repeat=5, maxdepth=2)
    # Depth as computed by iterative_tree is the number of path separators in
    # the relative path, so the deepest created folder has depth maxdepth.
    depths = [len(d.relative_to(base).parts) - 1 for d in dirs]
    assert max(depths) == 2


def test_unlimited_depth_grows_with_repeat(tmp_path):
    base = tmp_path / "base"
    dirs, _ = iterative_tree(base, _const(1), _const(0), repeat=3)
    assert max(len(d.relative_to(base).parts) for d in dirs) == 3


def test_nfolders_and_nfiles_receive_depth(tmp_path):
    folder_depths = []
    file_depths = []

    def nfolders_func(depth):
        folder_depths.append(depth)
        return 1

    def nfiles_func(depth):
        file_depths.append(depth)
        return 0

    iterative_tree(tmp_path / "base", nfolders_func, nfiles_func, repeat=3)
    assert set(folder_depths) == {0, 1}
    assert set(file_depths) == {0, 1}
    assert all(isinstance(d, int) for d in folder_depths + file_depths)


def test_custom_filename_and_duplicate_collapse(tmp_path):
    base = tmp_path / "base"
    dirs, files = iterative_tree(
        base, _const(0), _const(1), repeat=1, filename=lambda: "data.bin"
    )
    assert files == [base / "data.bin"]
    assert (base / "data.bin").is_file()

    # Same name requested several times yields one unique file in the result.
    base2 = tmp_path / "base2"
    _, files2 = iterative_tree(
        base2, _const(0), _const(3), repeat=1, filename=lambda: "same.txt"
    )
    assert len(files2) == 1
    assert (base2 / "same.txt").is_file()


def test_payload_generator_creates_files_and_overrides_filename(tmp_path):
    def payload(directory):
        for i in range(10):
            path = directory / "payload{}.txt".format(i)
            path.write_text("content")
            yield path

    def forbidden():
        raise AssertionError("filename must not be used when payload is given")

    base = tmp_path / "base"
    dirs, files = iterative_tree(
        base, _const(0), _const(2), repeat=1,
        filename=forbidden, payload=payload,
    )
    assert dirs == []
    assert len(files) == 2
    for f in files:
        assert f.parent == base
        assert f.is_file()
        assert f.read_text() == "content"


# ---------------------------------------------------------------------------
# iterative_gaussian_tree
# ---------------------------------------------------------------------------

def test_gaussian_passes_mean_and_sigma(tmp_path, monkeypatch):
    calls = []

    def fake_gauss(mu, sigma):
        calls.append((mu, sigma))
        return 2.0

    monkeypatch.setattr(random, "gauss", fake_gauss)
    base = tmp_path / "base"
    dirs, files = iterative_gaussian_tree(
        base,
        nfiles=5,
        nfolders=3,
        sigma_files=0.5,
        sigma_folders=0.25,
        repeat=1,
        filename=_counting_namer(),
    )
    assert (3, 0.25) in calls
    assert (5, 0.5) in calls
    assert len(dirs) == 2
    assert len(files) == 2


def test_gaussian_clamps_to_minimums(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: -3.0)
    base = tmp_path / "base"
    dirs, files = iterative_gaussian_tree(
        base,
        repeat=1,
        min_folders=1,
        min_files=2,
        filename=_counting_namer(),
    )
    assert len(dirs) == 1
    assert len(files) == 2


def test_gaussian_truncates_to_int(tmp_path, monkeypatch):
    # nfiles mean is 5, nfolders mean is 3 in this call.
    def fake_gauss(mu, sigma):
        return 2.9 if mu == 5 else -0.7

    monkeypatch.setattr(random, "gauss", fake_gauss)
    base = tmp_path / "base"
    dirs, files = iterative_gaussian_tree(
        base,
        nfiles=5,
        nfolders=3,
        repeat=1,
        filename=_counting_namer(),
    )
    assert dirs == []
    assert len(files) == 2


def test_gaussian_zero_sigma_is_exact(tmp_path):
    base = tmp_path / "base"
    dirs, files = iterative_gaussian_tree(
        base,
        nfiles=4,
        nfolders=0,
        sigma_files=0,
        sigma_folders=0,
        repeat=1,
        filename=_counting_namer(),
    )
    assert dirs == []
    assert len(files) == 4
    assert all(f.parent == base for f in files)


# ---------------------------------------------------------------------------
# choose_random_elements
# ---------------------------------------------------------------------------

def test_choose_returns_requested_counts_and_members(sample_tree):
    dirs, files = choose_random_elements(sample_tree, 5, 7)
    assert len(dirs) == 5
    assert len(files) == 7
    assert all(isinstance(p, Path) for p in dirs + files)
    assert set(dirs) <= _expected_dirs(sample_tree)
    assert set(files) <= _expected_files(sample_tree)


def test_choose_allows_repetition(tmp_path):
    base = tmp_path / "solo"
    (base / "only").mkdir(parents=True)
    (base / "lone.txt").write_text("x")
    dirs, files = choose_random_elements(base, 3, 4)
    assert dirs == [base / "only"] * 3
    assert files == [base / "lone.txt"] * 4


def test_choose_no_dirs_raises_by_default(tmp_path):
    base = tmp_path / "only_files"
    base.mkdir()
    (base / "a.txt").write_text("x")
    with pytest.raises(ValueError):
        choose_random_elements(base, 1, 0)


def test_choose_no_dirs_ignore_returns_empty_dirs(tmp_path):
    base = tmp_path / "only_files"
    base.mkdir()
    (base / "a.txt").write_text("x")
    dirs, files = choose_random_elements(base, 2, 1, onfail="ignore")
    assert dirs == []
    assert files == [base / "a.txt"]


def test_choose_no_files_raises_by_default(tmp_path):
    base = tmp_path / "only_dirs"
    (base / "sub").mkdir(parents=True)
    with pytest.raises(ValueError):
        choose_random_elements(base, 0, 1)


def test_choose_no_files_ignore_and_unknown_onfail(tmp_path):
    base = tmp_path / "only_dirs"
    (base / "sub").mkdir(parents=True)

    dirs, files = choose_random_elements(base, 0, 2, onfail="ignore")
    assert files == []

    with pytest.raises(ValueError, match="Unknown"):
        choose_random_elements(base, 0, 2, onfail="bogus")


# ---------------------------------------------------------------------------
# sample_random_elements
# ---------------------------------------------------------------------------

def test_sample_returns_distinct_members(sample_tree):
    dirs, files = sample_random_elements(sample_tree, 2, 3)
    assert len(dirs) == 2
    assert len(files) == 3
    assert len(set(dirs)) == 2
    assert len(set(files)) == 3
    assert set(dirs) <= _expected_dirs(sample_tree)
    assert set(files) <= _expected_files(sample_tree)


def test_sample_full_selection_returns_everything(sample_tree):
    dirs, files = sample_random_elements(sample_tree, 3, 4)
    assert set(dirs) == _expected_dirs(sample_tree)
    assert set(files) == _expected_files(sample_tree)


def test_sample_raises_when_insufficient(sample_tree):
    with pytest.raises(ValueError):
        sample_random_elements(sample_tree, 4, 0)
    with pytest.raises(ValueError):
        sample_random_elements(sample_tree, 0, 5)


def test_sample_ignore_returns_all_available(sample_tree):
    dirs, files = sample_random_elements(sample_tree, 10, 10, onfail="ignore")
    assert len(dirs) == 3
    assert len(files) == 4
    assert set(dirs) == _expected_dirs(sample_tree)
    assert set(files) == _expected_files(sample_tree)


def test_sample_unknown_onfail_raises(sample_tree):
    with pytest.raises(ValueError, match="Unknown"):
        sample_random_elements(sample_tree, 10, 0, onfail="bogus")
