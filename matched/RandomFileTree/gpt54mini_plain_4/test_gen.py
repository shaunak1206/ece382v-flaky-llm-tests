import random
from pathlib import Path

import pytest

import randomfiletree.core as core


def test_iterative_tree_creates_files_and_directories(tmp_path):
    def nfolders_func(depth):
        return 1 if depth == 0 else 0

    def nfiles_func(depth):
        return 2 if depth == 0 else 0

    monkey_name = "childdir"
    monkey_files = ["a.txt", "b.txt"]

    def fake_random_string(*args, **kwargs):
        if not hasattr(fake_random_string, "calls"):
            fake_random_string.calls = 0
        fake_random_string.calls += 1
        return monkey_name if fake_random_string.calls == 1 else f"ignored{fake_random_string.calls}"

    def fake_filename():
        return monkey_files.pop(0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core, "random_string", fake_random_string)
    try:
        dirs, files = core.iterative_tree(
            tmp_path, nfolders_func=nfolders_func, nfiles_func=nfiles_func, repeat=1, filename=fake_filename
        )
    finally:
        monkeypatch.undo()

    assert len(dirs) == 1
    assert len(files) == 2
    assert (tmp_path / monkey_name).is_dir()
    assert (tmp_path / "a.txt").is_file()
    assert (tmp_path / "b.txt").is_file()


def test_iterative_tree_respects_maxdepth(tmp_path):
    def nfolders_func(depth):
        return 1

    def nfiles_func(depth):
        return 0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core, "random_string", lambda *a, **k: "subdir")
    try:
        dirs, files = core.iterative_tree(
            tmp_path, nfolders_func=nfolders_func, nfiles_func=nfiles_func, repeat=2, maxdepth=1
        )
    finally:
        monkeypatch.undo()

    assert len(files) == 0
    # Only one directory at the root should be created; maxdepth prevents descent into it.
    assert len(dirs) == 1
    assert (tmp_path / "subdir").is_dir()
    assert list((tmp_path / "subdir").iterdir()) == []


def test_iterative_tree_deduplicates_return_values(tmp_path):
    def nfolders_func(depth):
        return 2

    def nfiles_func(depth):
        return 2

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core, "random_string", lambda *a, **k: "same_name")
    monkeypatch.setattr(core, "random", random.Random(0))
    try:
        dirs, files = core.iterative_tree(
            tmp_path, nfolders_func=nfolders_func, nfiles_func=nfiles_func, repeat=1
        )
    finally:
        monkeypatch.undo()

    assert len(dirs) == 1
    assert len(files) == 1
    assert dirs[0] == tmp_path / "same_name"
    assert files[0] == tmp_path / "same_name"


def test_iterative_tree_with_payload_uses_generator_and_ignores_filename(tmp_path):
    def nfolders_func(depth):
        return 0

    def nfiles_func(depth):
        return 2

    def payload(dir_path):
        def gen():
            yield dir_path / "payload_1.txt"
            yield dir_path / "payload_2.txt"
        return gen()

    called = {"filename": 0}

    def fake_filename():
        called["filename"] += 1
        return "should_not_be_used"

    dirs, files = core.iterative_tree(
        tmp_path,
        nfolders_func=nfolders_func,
        nfiles_func=nfiles_func,
        repeat=1,
        filename=fake_filename,
        payload=payload,
    )

    assert dirs == []
    assert files == [tmp_path / "payload_1.txt", tmp_path / "payload_2.txt"]
    assert called["filename"] == 0
    assert (tmp_path / "payload_1.txt").exists()
    assert (tmp_path / "payload_2.txt").exists()


def test_iterative_gaussian_tree_uses_gaussian_parameters(tmp_path):
    gauss_values = iter([2.9, 1.9])  # int() -> 2 and 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core.random, "gauss", lambda mean, sigma: next(gauss_values))
    monkeypatch.setattr(core, "random_string", lambda *a, **k: "x")
    try:
        dirs, files = core.iterative_gaussian_tree(tmp_path, nfiles=10, nfolders=10, repeat=1)
    finally:
        monkeypatch.undo()

    assert len(dirs) == 1
    assert len(files) == 2
    assert (tmp_path / "x").exists()


def test_iterative_gaussian_tree_applies_minimums(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core.random, "gauss", lambda mean, sigma: -100)
    monkeypatch.setattr(core, "random_string", lambda *a, **k: "minname")
    try:
        dirs, files = core.iterative_gaussian_tree(
            tmp_path,
            nfiles=2,
            nfolders=1,
            repeat=1,
            min_files=3,
            min_folders=2,
        )
    finally:
        monkeypatch.undo()

    assert len(dirs) == 2
    assert len(files) == 3
    assert all(p.exists() for p in dirs)
    assert all(p.exists() for p in files)


def test_choose_random_elements_picks_requested_counts(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "f1.txt").write_text("1")
    (tmp_path / "d1" / "f2.txt").write_text("2")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core.random, "choice", lambda seq: seq[0])
    try:
        dirs, files = core.choose_random_elements(tmp_path, n_dirs=3, n_files=2)
    finally:
        monkeypatch.undo()

    assert dirs == [tmp_path / "d1", tmp_path / "d1", tmp_path / "d1"]
    assert files == [tmp_path / "f1.txt", tmp_path / "f1.txt"]


def test_choose_random_elements_raises_when_no_dirs_and_onfail_raise(tmp_path):
    (tmp_path / "f.txt").write_text("data")
    with pytest.raises(ValueError, match="does not have subfolders"):
        core.choose_random_elements(tmp_path, n_dirs=1, n_files=0, onfail="raise")


def test_choose_random_elements_raises_when_no_files_and_onfail_raise(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match="does not contain any files"):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignores_missing_dirs_and_files(tmp_path):
    dirs, files = core.choose_random_elements(tmp_path, n_dirs=2, n_files=3, onfail="ignore")
    assert dirs == []
    assert files == []


def test_choose_random_elements_unknown_onfail_raises_for_missing_files(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match="Unknown value for 'onfail' parameter"):
        core.choose_random_elements(tmp_path, n_dirs=0, n_files=1, onfail="bogus")


def test_sample_random_elements_returns_unique_elements(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "f1.txt").write_text("1")
    (tmp_path / "f2.txt").write_text("2")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core.random, "sample", lambda seq, k: list(seq)[:k])
    try:
        dirs, files = core.sample_random_elements(tmp_path, n_dirs=2, n_files=2)
    finally:
        monkeypatch.undo()

    assert dirs == [tmp_path / "d1", tmp_path / "d2"]
    assert files == [tmp_path / "f1.txt", tmp_path / "f2.txt"]


def test_sample_random_elements_raises_when_not_enough_dirs_and_onfail_raise(tmp_path):
    (tmp_path / "d1").mkdir()
    with pytest.raises(ValueError, match="does not have enough subfolders"):
        core.sample_random_elements(tmp_path, n_dirs=2, n_files=0, onfail="raise")


def test_sample_random_elements_raises_when_not_enough_files_and_onfail_raise(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "f1.txt").write_text("1")
    with pytest.raises(ValueError, match="does not contain enough files"):
        core.sample_random_elements(tmp_path, n_dirs=0, n_files=2, onfail="raise")


def test_sample_random_elements_ignore_returns_all_available_when_not_enough(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "f1.txt").write_text("1")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(core.random, "sample", lambda seq, k: list(seq)[:k])
    try:
        dirs, files = core.sample_random_elements(tmp_path, n_dirs=3, n_files=4, onfail="ignore")
    finally:
        monkeypatch.undo()

    assert dirs == [tmp_path / "d1"]
    assert files == [tmp_path / "f1.txt"]
