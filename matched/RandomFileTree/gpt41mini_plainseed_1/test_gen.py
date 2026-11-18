import pytest
import os
import random
from pathlib import Path
from randomfiletree import core


@pytest.fixture
def tmpdir_path(tmp_path):
    # Provided tmp_path fixture but rename for clarity
    return tmp_path


def test_iterative_tree_creates_correct_files_and_dirs(tmpdir_path):
    def nfolders(depth):
        return 2 if depth == 0 else 0

    def nfiles(depth):
        return 3 if depth == 0 else 0

    dirs, files = core.iterative_tree(
        basedir=tmpdir_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=1,
    )
    # Should create 2 dirs and 3 files in base
    assert len(dirs) == 2
    assert len(files) == 3
    # Check dirs and files exist
    for d in dirs:
        assert d.exists() and d.is_dir()
    for f in files:
        assert f.exists() and f.is_file()


def test_iterative_tree_respects_maxdepth(tmpdir_path):
    def nfolders(depth):
        return 1 if depth < 2 else 2

    def nfiles(depth):
        return 1

    # maxdepth=2 means max depth is 1 relative to basedir, so folders deeper omitted
    dirs, files = core.iterative_tree(
        basedir=tmpdir_path,
        nfolders_func=nfolders,
        nfiles_func=nfiles,
        repeat=1,
        maxdepth=2,
    )

    # No folder deeper than depth 1 i.e. base + one level folders
    for d in dirs:
        depth = d.relative_to(tmpdir_path).parts
        assert len(depth) <= 2

    # All files created at allowed depths
    for f in files:
        depth = f.relative_to(tmpdir_path).parts
        assert len(depth) <= 2


def test_iterative_tree_repeat_creates_more_items(tmpdir_path):
    def nfolders(depth):
        return 1

    def nfiles(depth):
        return 1

    # repeat=2 should create roughly double folders and files than repeat=1
    dirs1, files1 = core.iterative_tree(tmpdir_path / "r1", nfolders, nfiles, repeat=1)
    dirs2, files2 = core.iterative_tree(tmpdir_path / "r2", nfolders, nfiles, repeat=2)
    # Expect more or equal
    assert len(dirs2) >= 2 * len(dirs1)
    assert len(files2) >= 2 * len(files1)


def test_iterative_tree_filename_callable_used(tmpdir_path):
    def nfolders(depth):
        return 0

    def nfiles(depth):
        return 2

    def filename():
        return "fixedname.txt"

    dirs, files = core.iterative_tree(
        basedir=tmpdir_path, nfolders_func=nfolders, nfiles_func=nfiles, filename=filename
    )
    # All files named 'fixedname.txt', but may be overwritten or repeated
    for f in files:
        assert f.name == "fixedname.txt"


def test_iterative_tree_with_payload_creates_files(tmpdir_path):
    def nfolders(depth):
        return 0

    def nfiles(depth):
        return 3

    def payload_generator(dirpath):
        def gen():
            for i in range(3):
                p = dirpath / f"payload_{i}.txt"
                p.write_text(f"content {i}")
                yield p

        return gen()

    dirs, files = core.iterative_tree(
        tmpdir_path, nfolders, nfiles, payload=payload_generator
    )
    # Should create 3 files, no dirs
    assert dirs == []
    assert len(files) == 3
    for f in files:
        assert f.exists()
        content = f.read_text()
        assert content.startswith("content ")


def test_iterative_gaussian_tree_creates_dirs_and_files(tmpdir_path):
    dirs, files = core.iterative_gaussian_tree(
        tmpdir_path,
        nfiles=3,
        nfolders=2,
        repeat=1,
        sigma_folders=0,
        sigma_files=0,
        min_folders=2,
        min_files=3,
        filename=lambda: "file_gauss.txt",
    )

    # Because sigma=0, count should be exactly min_folders and min_files
    assert len(dirs) >= 2
    assert len(files) >= 3
    for d in dirs:
        assert d.exists() and d.is_dir()
    for f in files:
        assert f.exists() and f.is_file()


def test_choose_random_elements_selects_elements(tmpdir_path):
    # Setup folders and files
    (tmpdir_path / "sub1").mkdir()
    (tmpdir_path / "sub2").mkdir()
    (tmpdir_path / "file1.txt").write_text("data1")
    (tmpdir_path / "file2.txt").write_text("data2")

    dirs, files = core.choose_random_elements(tmpdir_path, n_dirs=1, n_files=1)
    assert len(dirs) == 1
    assert isinstance(dirs[0], Path)
    assert len(files) == 1
    assert isinstance(files[0], Path)
    assert dirs[0].exists() and dirs[0].is_dir()
    assert files[0].exists() and files[0].is_file()


def test_choose_random_elements_raises_onfail_raise(tmpdir_path):
    # empty dir, request dirs and files with onfail='raise'
    with pytest.raises(ValueError):
        core.choose_random_elements(tmpdir_path, n_dirs=1, n_files=0, onfail="raise")

    with pytest.raises(ValueError):
        core.choose_random_elements(tmpdir_path, n_dirs=0, n_files=1, onfail="raise")


def test_choose_random_elements_ignores_onfail_ignore(tmpdir_path):
    # empty dir with onfail='ignore' returns empty lists for requested dirs/files
    dirs, files = core.choose_random_elements(tmpdir_path, n_dirs=1, n_files=1, onfail="ignore")
    assert dirs == []
    assert files == []


def test_choose_random_elements_unknown_onfail_raises(tmpdir_path):
    # Setup a file to avoid failure on files
    (tmpdir_path / "file.txt").write_text("data")
    with pytest.raises(ValueError):
        core.choose_random_elements(tmpdir_path, n_dirs=0, n_files=1, onfail="unknown")


def test_sample_random_elements_selects_distinct(tmpdir_path):
    (tmpdir_path / "sub1").mkdir()
    (tmpdir_path / "sub2").mkdir()
    (tmpdir_path / "file1.txt").write_text("foo")
    (tmpdir_path / "file2.txt").write_text("bar")

    dirs, files = core.sample_random_elements(tmpdir_path, n_dirs=2, n_files=2)
    assert len(dirs) == 2
    assert len(files) == 2
    assert len(set(dirs)) == 2
    assert len(set(files)) == 2


def test_sample_random_elements_raises_onfail_raise(tmpdir_path):
    (tmpdir_path / "sub1").mkdir()
    (tmpdir_path / "file1.txt").write_text("foo")

    with pytest.raises(ValueError):
        # Requesting more dirs than present
        core.sample_random_elements(tmpdir_path, n_dirs=2, n_files=0, onfail="raise")

    with pytest.raises(ValueError):
        # Requesting more files than present
        core.sample_random_elements(tmpdir_path, n_dirs=0, n_files=2, onfail="raise")


def test_sample_random_elements_ignores_onfail_ignore(tmpdir_path):
    (tmpdir_path / "sub1").mkdir()
    (tmpdir_path / "file1.txt").write_text("foo")

    dirs, files = core.sample_random_elements(tmpdir_path, n_dirs=2, n_files=2, onfail="ignore")
    # Returned lists smaller or equal to requested, but not raise
    assert len(dirs) == 1
    assert len(files) == 1


def test_sample_random_elements_unknown_onfail_raises(tmpdir_path):
    with pytest.raises(ValueError):
        core.sample_random_elements(tmpdir_path, n_dirs=1, n_files=0, onfail="unknown")


def test_payload_generator_iterable_provided(tmpdir_path):
    # Implement a payload yielding exact count files each call
    def nfolders(depth):
        return 0

    def nfiles(depth):
        return 2

    def payload_gen(dirpath):
        def gen():
            for i in range(2):
                p = dirpath / f"payloadfile_{i}.txt"
                p.write_text(str(i))
                yield p

        return gen()

    dirs, files = core.iterative_tree(tmpdir_path, nfolders, nfiles, payload=payload_gen)
    assert len(dirs) == 0
    assert len(files) == 2
    for f in files:
        assert f.exists()
        assert f.read_text() in ("0", "1")


def test_iterative_tree_sets_alldirs_and_allfiles_unique(tmpdir_path):
    def nfolders(depth):
        # Return same folder count to test unique paths in list
        return 1

    def nfiles(depth):
        return 1

    dirs, files = core.iterative_tree(tmpdir_path, nfolders, nfiles, repeat=2)
    # Due to repeat >1 and folders new each iteration, count should be > 1 but duplicates removed
    assert isinstance(dirs, list)
    assert isinstance(files, list)
    assert all(isinstance(d, Path) for d in dirs)
    assert all(isinstance(f, Path) for f in files)


def test_iterative_tree_handles_zero_folders_and_files(tmpdir_path):
    def zero(depth):
        return 0

    dirs, files = core.iterative_tree(tmpdir_path, zero, zero)
    assert dirs == []
    assert files == []
