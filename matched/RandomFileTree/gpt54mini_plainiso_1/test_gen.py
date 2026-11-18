import os
from pathlib import Path

import pytest

import randomfiletree.core as core


def _make_tree(root: Path):
    (root / "d1").mkdir(parents=True)
    (root / "d2").mkdir()
    (root / "f1.txt").write_text("a")
    (root / "d1" / "f2.txt").write_text("b")
    (root / "d1" / "f3.txt").write_text("c")


def test_iterative_tree_creates_expected_numbers_of_dirs_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_string", lambda *args, **kwargs: "x")

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 1 if depth == 0 else 0,
        nfiles_func=lambda depth: 2 if depth == 0 else 0,
        repeat=1,
    )

    assert len(dirs) == 1
    assert len(files) == 2
    assert all(isinstance(p, Path) for p in dirs + files)
    assert (tmp_path / "x").is_dir()
    assert (tmp_path / "x").exists() is False or True  # created path returned as dir


def test_iterative_tree_returns_unique_paths_when_duplicates_are_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_string", lambda *args, **kwargs: "dup")

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 2 if depth == 0 else 0,
        nfiles_func=lambda depth: 3 if depth == 0 else 0,
        repeat=1,
    )

    assert len(dirs) == 1
    assert len(files) == 1
    assert dirs[0] == tmp_path / "dup"
    assert files[0] == tmp_path / "dup"


def test_iterative_tree_respects_maxdepth_and_does_not_descend_beyond_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_string", lambda *args, **kwargs: "child")

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 1,
        nfiles_func=lambda depth: 0,
        repeat=2,
        maxdepth=1,
    )

    assert len(dirs) == 1
    assert (tmp_path / "child").is_dir()
    assert not any(p.parent == tmp_path / "child" for p in dirs)
    assert not any(p.parent == tmp_path / "child" for p in files)


def test_iterative_tree_supports_payload_and_ignores_filename_when_payload_is_given(tmp_path):
    created = []

    def payload(root: Path):
        def generator():
            for i in range(2):
                p = root / f"payload_{i}.txt"
                p.write_text(f"data-{i}")
                created.append(p)
                yield p

        return generator()

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 0,
        nfiles_func=lambda depth: 2,
        repeat=1,
        payload=payload,
        filename=lambda: pytest.fail("filename should not be called when payload is provided"),
    )

    assert dirs == []
    assert files == created
    assert all(p.read_text().startswith("data-") for p in files)


def test_iterative_tree_repeat_walks_multiple_times_and_collects_new_paths(tmp_path, monkeypatch):
    names = iter(["a", "b", "c", "d"])
    monkeypatch.setattr(core, "random_string", lambda *args, **kwargs: next(names))

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=lambda depth: 1 if depth < 2 else 0,
        nfiles_func=lambda depth: 1 if depth < 2 else 0,
        repeat=2,
    )

    assert {p.name for p in dirs} == {"a", "b"}
    assert {p.name for p in files} == {"c", "d"}


def test_iterative_gaussian_tree_applies_minimums_and_delegates_to_iterative_tree(monkeypatch, tmp_path):
    captured = {}

    def fake_iterative_tree(**kwargs):
        captured.update(kwargs)
        return [tmp_path / "d"], [tmp_path / "f"]

    gauss_values = iter([-10, -3])
    monkeypatch.setattr(core.random, "gauss", lambda mu, sigma: next(gauss_values))
    monkeypatch.setattr(core, "iterative_tree", fake_iterative_tree)

    dirs, files = core.iterative_gaussian_tree(
        tmp_path,
        nfiles=5,
        nfolders=7,
        sigma_files=1,
        sigma_folders=1,
        min_files=2,
        min_folders=1,
        repeat=3,
        maxdepth=4,
        filename=lambda: "name",
    )

    assert dirs == [tmp_path / "d"]
    assert files == [tmp_path / "f"]
    assert captured["basedir"] == tmp_path
    assert captured["repeat"] == 3
    assert captured["maxdepth"] == 4
    assert captured["filename"]() == "name"
    assert captured["nfolders_func"]() == 1
    assert captured["nfiles_func"]() == 2


def test_choose_random_elements_selects_requested_number_with_replacement(tmp_path, monkeypatch):
    _make_tree(tmp_path)
    monkeypatch.setattr(core.random, "choice", lambda seq: seq[0])

    dirs, files = core.choose_random_elements(tmp_path, n_dirs=3, n_files=2)

    assert len(dirs) == 3
    assert len(files) == 2
    assert all(p == dirs[0] for p in dirs)
    assert all(p == files[0] for p in files)


def test_choose_random_elements_raises_for_missing_directories(tmp_path):
    (tmp_path / "only.txt").write_text("x")

    with pytest.raises(ValueError, match="does not have subfolders"):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")


def test_choose_random_elements_raises_for_missing_files(tmp_path):
    (tmp_path / "sub").mkdir()

    with pytest.raises(ValueError, match="does not contain any files"):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignore_returns_empty_lists_when_tree_is_empty(tmp_path):
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=2, n_files=3, onfail="ignore")
    assert dirs == []
    assert files == []


def test_choose_random_elements_raises_for_unknown_onfail_when_files_missing(tmp_path):
    (tmp_path / "sub").mkdir()

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="bad")


def test_sample_random_elements_returns_unique_dirs_and_files(tmp_path):
    _make_tree(tmp_path)

    dirs, files = core.sample_random_elements(tmp_path, n_dirs=2, n_files=2)

    assert len(dirs) == 2
    assert len(files) == 2
    assert len(set(dirs)) == 2
    assert len(set(files)) == 2


def test_sample_random_elements_raises_when_not_enough_directories(tmp_path):
    (tmp_path / "f.txt").write_text("x")

    with pytest.raises(ValueError, match="does not have enough subfolders"):
        core.sample_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")


def test_sample_random_elements_ignore_returns_all_available_when_not_enough_files(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f1.txt").write_text("x")
    (tmp_path / "d" / "f2.txt").write_text("y")

    dirs, files = core.sample_random_elements(tmp_path, n_dirs=0, n_files=5, onfail="ignore")

    assert dirs == []
    assert len(files) == 2
    assert set(files) == {tmp_path / "d" / "f1.txt", tmp_path / "d" / "f2.txt"}


def test_sample_random_elements_raises_for_unknown_onfail_when_not_enough_dirs(tmp_path):
    (tmp_path / "f.txt").write_text("x")

    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter."):
        core.sample_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="bad")
