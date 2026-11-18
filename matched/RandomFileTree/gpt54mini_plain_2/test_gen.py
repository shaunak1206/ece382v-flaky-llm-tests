import os
from pathlib import Path

import pytest

import randomfiletree.core as core


def test_iterative_tree_creates_expected_number_of_dirs_and_files(tmp_path):
    def nfolders(depth):
        return 1 if depth == 0 else 0

    def nfiles(depth):
        return 2 if depth == 0 else 0

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=1,
        filename=lambda: "fixed_name",
    )

    assert len(dirs) == 1
    assert len(files) == 2
    assert (tmp_path / "fixed_name").exists()
    assert all(isinstance(p, Path) for p in dirs + files)


def test_iterative_tree_uses_payload_instead_of_filename(tmp_path):
    created = []

    def payload(root):
        def gen():
            p = root / "payload.txt"
            p.write_text("hello")
            created.append(p)
            yield p

        return gen()

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=lambda depth: 0,
        nfiles_func=lambda depth: 1,
        payload=payload,
        filename=lambda: pytest.fail("filename should not be called when payload is set"),
    )

    assert dirs == []
    assert files == [tmp_path / "payload.txt"]
    assert created == [tmp_path / "payload.txt"]
    assert (tmp_path / "payload.txt").read_text() == "hello"


def test_iterative_tree_respects_maxdepth(tmp_path):
    def nfolders(depth):
        return 1 if depth == 0 else 1

    def nfiles(depth):
        return 0

    dirs, _ = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=2,
        maxdepth=2,
        filename=lambda: "x",
    )

    assert len(dirs) >= 1
    assert all(len(p.relative_to(tmp_path).parts) <= 2 for p in dirs)


def test_iterative_tree_deduplicates_returned_paths(tmp_path):
    def nfolders(depth):
        return 0

    def nfiles(depth):
        return 3

    dirs, files = core.iterative_tree(
        basedir=tmp_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=2,
        filename=lambda: "same_name",
    )

    assert dirs == []
    assert files == [tmp_path / "same_name"]


def test_iterative_tree_creates_basedir(tmp_path):
    basedir = tmp_path / "new_root"
    assert not basedir.exists()

    core.iterative_tree(
        basedir=basedir,
        nfolders_func=lambda depth: 0,
        nfiles_func=lambda depth: 0,
    )

    assert basedir.exists()
    assert basedir.is_dir()


def test_iterative_gaussian_tree_delegates_to_iterative_tree(monkeypatch, tmp_path):
    captured = {}

    def fake_iterative_tree(**kwargs):
        captured.update(kwargs)
        return ["dirs"], ["files"]

    monkeypatch.setattr(core, "iterative_tree", fake_iterative_tree)

    result = core.iterative_gaussian_tree(
        basedir=tmp_path,
        nfiles=5,
        nfolders=3,
        repeat=7,
        maxdepth=2,
        sigma_folders=0.5,
        sigma_files=0.25,
        min_folders=1,
        min_files=2,
        filename=lambda: "name",
        payload=lambda root: iter(()),
    )

    assert result == (["dirs"], ["files"])
    assert captured["basedir"] == tmp_path
    assert captured["repeat"] == 7
    assert captured["maxdepth"] == 2
    assert captured["filename"]() == "name"
    assert captured["payload"] is not None


def test_iterative_gaussian_tree_applies_minimums(monkeypatch, tmp_path):
    monkeypatch.setattr(core.random, "gauss", lambda mean, sigma: -100)

    captured = {}

    def fake_iterative_tree(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(core, "iterative_tree", fake_iterative_tree)

    core.iterative_gaussian_tree(
        basedir=tmp_path,
        nfiles=2,
        nfolders=3,
        min_files=4,
        min_folders=5,
    )

    assert captured["nfiles_func"]() == 4
    assert captured["nfolders_func"]() == 5


def test_iterative_gaussian_tree_uses_gaussian_values_when_above_minimum(monkeypatch, tmp_path):
    values = [6.9, 8.2]

    def fake_gauss(mean, sigma):
        return values.pop(0)

    monkeypatch.setattr(core.random, "gauss", fake_gauss)

    captured = {}

    def fake_iterative_tree(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(core, "iterative_tree", fake_iterative_tree)

    core.iterative_gaussian_tree(
        basedir=tmp_path,
        nfiles=10,
        nfolders=20,
        min_files=1,
        min_folders=2,
    )

    assert captured["nfiles_func"]() == 6
    assert captured["nfolders_func"]() == 8


def test_choose_random_elements_picks_from_all_walked_items(tmp_path, monkeypatch):
    root = tmp_path
    (root / "a").mkdir()
    (root / "b").mkdir()
    (root / "file1.txt").write_text("1")
    (root / "a" / "file2.txt").write_text("2")

    chosen = []

    def fake_choice(seq):
        chosen.append(list(seq))
        return seq[0]

    monkeypatch.setattr(core.random, "choice", fake_choice)

    dirs, files = core.choose_random_elements(root, n_dirs=2, n_files=2)

    assert dirs == [root / "a", root / "a"]
    assert files == [root / "file1.txt", root / "file1.txt"]
    assert any(root / "a" in seq for seq in chosen)
    assert any(root / "file1.txt" in seq for seq in chosen)


def test_choose_random_elements_raises_when_no_dirs_and_requested(tmp_path):
    (tmp_path / "f.txt").write_text("x")

    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")


def test_choose_random_elements_raises_when_no_files_and_requested(tmp_path):
    (tmp_path / "sub").mkdir()

    with pytest.raises(ValueError):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignore_returns_empty_lists_when_tree_empty(tmp_path):
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=2, n_files=3, onfail="ignore")
    assert dirs == []
    assert files == []


def test_choose_random_elements_raises_on_unknown_onfail_for_files(tmp_path):
    (tmp_path / "sub").mkdir()

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="bad")


def test_sample_random_elements_returns_unique_samples(tmp_path, monkeypatch):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "f1.txt").write_text("1")
    (tmp_path / "f2.txt").write_text("2")

    def fake_sample(seq, k):
        return list(seq)[:k]

    monkeypatch.setattr(core.random, "sample", fake_sample)

    dirs, files = core.sample_random_elements(tmp_path, n_dirs=2, n_files=2)

    assert dirs == [tmp_path / "d1", tmp_path / "d2"]
    assert files == [tmp_path / "f1.txt", tmp_path / "f2.txt"]


def test_sample_random_elements_raises_when_not_enough_dirs(tmp_path):
    (tmp_path / "only").mkdir()

    with pytest.raises(ValueError):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=0, onfail="raise")


def test_sample_random_elements_ignore_returns_all_available_when_not_enough_files(tmp_path, monkeypatch):
    (tmp_path / "d1").mkdir()
    (tmp_path / "f1.txt").write_text("1")

    monkeypatch.setattr(core.random, "sample", lambda seq, k: list(seq)[:k])

    dirs, files = core.sample_random_elements(tmp_path, n_dirs=1, n_files=3, onfail="ignore")

    assert dirs == [tmp_path / "d1"]
    assert files == [tmp_path / "f1.txt"]


def test_sample_random_elements_raises_on_unknown_onfail(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "f1.txt").write_text("1")

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=0, onfail="maybe")
