import random
from pathlib import Path

import pytest

import randomfiletree.core as core


def test_iterative_tree_creates_expected_number_of_dirs_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_string", lambda min_length=5, max_length=10: "dirx")
    monkeypatch.setattr(core.random, "choice", lambda seq: seq[0])

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=lambda depth: 1 if depth == 0 else 0,
        nfiles_func=lambda depth: 2 if depth == 0 else 0,
        repeat=1,
        filename=lambda: "filex",
    )

    assert len(dirs) == 1
    assert len(files) == 2
    assert (tmp_path / "dirx").is_dir()
    assert (tmp_path / "filex").is_file()


def test_iterative_tree_respects_maxdepth(tmp_path, monkeypatch):
    # Create a stable nested tree: root -> child -> grandchild if not pruned.
    names = iter(["d1", "d2", "f1", "f2", "d3", "d4", "f3", "f4"])

    monkeypatch.setattr(core, "random_string", lambda min_length=5, max_length=10: next(names))

    def nfolders(depth):
        return 1 if depth < 2 else 0

    def nfiles(depth):
        return 1

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=1,
        maxdepth=2,
        filename=lambda: next(names),
    )

    # Root + child only, because depth 1 prunes further descent.
    assert (tmp_path / "d1").is_dir()
    assert (tmp_path / "d1" / "d2").is_dir()
    assert not (tmp_path / "d1" / "d2" / "d3").exists()
    assert len(dirs) == 2
    assert len(files) == 2


def test_iterative_tree_payload_overrides_filename(tmp_path):
    def payload(root):
        def gen():
            yield root / "payload_a.txt"
            yield root / "payload_b.txt"

        return gen()

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=lambda depth: 0,
        nfiles_func=lambda depth: 2,
        repeat=1,
        filename=lambda: "should_not_be_used.txt",
        payload=payload,
    )

    assert dirs == []
    assert files == [tmp_path / "payload_a.txt", tmp_path / "payload_b.txt"]
    assert (tmp_path / "payload_a.txt").exists()
    assert (tmp_path / "payload_b.txt").exists()
    assert not (tmp_path / "should_not_be_used.txt").exists()


def test_iterative_tree_deduplicates_returned_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_string", lambda min_length=5, max_length=10: "same_dir")
    monkeypatch.setattr(core.random, "choice", lambda seq: seq[0])

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=lambda depth: 1 if depth == 0 else 0,
        nfiles_func=lambda depth: 1,
        repeat=2,
        filename=lambda: "same_file.txt",
    )

    # Created twice across repeated walks, but returned lists are deduplicated.
    assert dirs == [tmp_path / "same_dir"]
    assert files == [tmp_path / "same_file.txt"]


def test_iterative_tree_creates_basedir_if_missing(tmp_path):
    basedir = tmp_path / "new_root"
    assert not basedir.exists()

    dirs, files = core.iterative_tree(
        basedir=basedir,
        nfolders_func=lambda depth: 0,
        nfiles_func=lambda depth: 0,
        repeat=1,
    )

    assert basedir.exists()
    assert dirs == []
    assert files == []


def test_iterative_gaussian_tree_uses_gauss_and_minimums(tmp_path, monkeypatch):
    values = iter([-2.2, 1.9, 3.7, 0.1])

    def fake_gauss(mu, sigma):
        return next(values)

    monkeypatch.setattr(core.random, "gauss", fake_gauss)
    monkeypatch.setattr(core, "random_string", lambda min_length=5, max_length=10: "gaussdir")

    dirs, files = core.iterative_gaussian_tree(
        basedir=tmp_path,
        nfiles=2,
        nfolders=1,
        repeat=1,
        sigma_folders=1,
        sigma_files=1,
        min_folders=0,
        min_files=0,
        filename=lambda: "gaussfile.txt",
    )

    # int(-2.2) == -2, int(1.9) == 1
    assert len(dirs) == 0
    assert len(files) == 1
    assert (tmp_path / "gaussfile.txt").exists()


def test_iterative_gaussian_tree_applies_minimum_clamp(tmp_path, monkeypatch):
    monkeypatch.setattr(core.random, "gauss", lambda mu, sigma: -100.0)
    monkeypatch.setattr(core, "random_string", lambda min_length=5, max_length=10: "min_dir")

    dirs, files = core.iterative_gaussian_tree(
        basedir=tmp_path,
        nfiles=0,
        nfolders=0,
        repeat=1,
        sigma_folders=1,
        sigma_files=1,
        min_folders=2,
        min_files=3,
        filename=lambda: "min_file.txt",
    )

    assert len(dirs) == 2
    assert len(files) == 3
    assert all(p.exists() for p in dirs + files)


def test_choose_random_elements_returns_duplicates_when_requested(tmp_path, monkeypatch):
    (tmp_path / "d1").mkdir()
    (tmp_path / "f1.txt").write_text("a")
    (tmp_path / "f2.txt").write_text("b")

    monkeypatch.setattr(core.random, "choice", lambda seq: seq[0])

    selected_dirs, selected_files = core.choose_random_elements(tmp_path, 3, 2)

    assert selected_dirs == [tmp_path / "d1", tmp_path / "d1", tmp_path / "d1"]
    assert selected_files == [tmp_path / "f1.txt", tmp_path / "f1.txt"]


def test_choose_random_elements_raises_when_no_dirs_and_onfail_raise(tmp_path):
    (tmp_path / "f.txt").write_text("x")

    with pytest.raises(ValueError, match="does not have subfolders"):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")


def test_choose_random_elements_raises_when_no_files_and_onfail_raise(tmp_path):
    (tmp_path / "d").mkdir()

    with pytest.raises(ValueError, match="does not contain any files"):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignore_missing_dirs_returns_empty_list(tmp_path):
    (tmp_path / "f.txt").write_text("x")

    selected_dirs, selected_files = core.choose_random_elements(
        tmp_path, n_dirs=2, n_files=1, onfail="ignore"
    )

    assert selected_dirs == []
    assert len(selected_files) == 1
    assert selected_files[0] == tmp_path / "f.txt"


def test_choose_random_elements_unknown_onfail_for_missing_files_raises(tmp_path):
    (tmp_path / "d").mkdir()

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="bad")


def test_sample_random_elements_returns_unique_items(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("a")
    f2.write_text("b")

    selected_dirs, selected_files = core.sample_random_elements(tmp_path, 2, 2)

    assert set(selected_dirs) == {d1, d2}
    assert set(selected_files) == {f1, f2}
    assert len(selected_dirs) == 2
    assert len(selected_files) == 2


def test_sample_random_elements_raises_when_not_enough_dirs(tmp_path):
    (tmp_path / "d1").mkdir()

    with pytest.raises(ValueError, match="not have enough subfolders"):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=0, onfail="raise")


def test_sample_random_elements_ignore_when_not_enough_files_returns_all_available(tmp_path):
    (tmp_path / "d1").mkdir()
    f1 = tmp_path / "f1.txt"
    f1.write_text("x")

    selected_dirs, selected_files = core.sample_random_elements(
        tmp_path, n_dirs=1, n_files=3, onfail="ignore"
    )

    assert selected_dirs == [tmp_path / "d1"]
    assert selected_files == [f1]


def test_sample_random_elements_unknown_onfail_raises(tmp_path):
    (tmp_path / "d1").mkdir()

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=0, onfail="weird")
