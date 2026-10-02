"""SoftFileLock specifics, UnixFileLock lock file handling and singleton locks."""
import os
import stat
import sys

import pytest

from filelock import FileLock, SoftFileLock, Timeout, UnixFileLock


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "soft.lock")


def test_soft_lock_file_exists_only_while_held(lock_path):
    lock = SoftFileLock(lock_path)
    assert not os.path.exists(lock_path)
    lock.acquire()
    assert os.path.exists(lock_path)
    lock.release()
    assert not os.path.exists(lock_path)


def test_nested_soft_lock_file_removed_only_on_final_release(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    lock.release()
    assert os.path.exists(lock_path)
    lock.release()
    assert not os.path.exists(lock_path)


@pytest.mark.skipif(sys.platform == "win32", reason="posix only")
def test_unix_lock_file_is_kept_after_release(lock_path):
    lock = UnixFileLock(lock_path)
    with lock:
        assert os.path.exists(lock_path)
    assert os.path.exists(lock_path)
    with lock:
        assert lock.is_locked


def test_stale_soft_lock_file_blocks_acquisition(lock_path):
    with open(lock_path, "w") as fh:
        fh.write("stale")
    lock = SoftFileLock(lock_path)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0.1, poll_interval=0.01)
    assert not lock.is_locked
    assert lock.lock_counter == 0
    os.remove(lock_path)
    lock.acquire(timeout=1, poll_interval=0.01)
    assert lock.is_locked
    lock.release()


def test_soft_lock_on_a_directory_raises(tmp_path):
    lock = SoftFileLock(str(tmp_path))
    with pytest.raises(IsADirectoryError):
        lock.acquire(timeout=0.1)
    assert lock.lock_counter == 0
    assert not lock.is_locked


@pytest.mark.skipif(sys.platform == "win32", reason="posix permissions")
def test_soft_lock_on_a_read_only_file_raises(lock_path):
    with open(lock_path, "w") as fh:
        fh.write("")
    os.chmod(lock_path, 0o444)
    try:
        lock = SoftFileLock(lock_path)
        with pytest.raises(PermissionError):
            lock.acquire(timeout=0.1)
        assert not lock.is_locked
    finally:
        os.chmod(lock_path, 0o644)


def test_soft_release_tolerates_an_already_deleted_lock_file(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    os.remove(lock_path)
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.skipif(sys.platform == "win32", reason="posix permissions")
def test_soft_lock_creates_the_file_with_the_requested_mode(lock_path):
    lock = SoftFileLock(lock_path, mode=0o600)
    assert lock.mode == 0o600
    with lock:
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600


def test_singleton_returns_the_same_instance(tmp_path):
    path = str(tmp_path / "single.lock")
    first = FileLock(path, is_singleton=True)
    second = FileLock(path, is_singleton=True)
    assert first is second
    assert first.is_singleton is True
    other = FileLock(str(tmp_path / "other.lock"), is_singleton=True)
    assert other is not first


def test_non_singleton_locks_are_distinct_objects(tmp_path):
    path = str(tmp_path / "plain.lock")
    assert FileLock(path) is not FileLock(path)
    assert SoftFileLock(path) is not SoftFileLock(path)


def test_singleton_with_differing_arguments_is_rejected(tmp_path):
    path = str(tmp_path / "single.lock")
    first = FileLock(path, is_singleton=True, timeout=5)
    assert FileLock(path, is_singleton=True, timeout=5) is first
    with pytest.raises(ValueError, match="differing arguments"):
        FileLock(path, is_singleton=True, timeout=10)
    with pytest.raises(ValueError):
        FileLock(path, is_singleton=True, timeout=5, thread_local=False)


def test_singleton_is_reentrant_without_passing_the_object_around(tmp_path):
    path = str(tmp_path / "reentrant.lock")
    a = SoftFileLock(path, is_singleton=True)
    b = SoftFileLock(path, is_singleton=True)
    a.acquire()
    b.acquire()
    assert a.lock_counter == 2
    b.release()
    assert a.is_locked
    a.release()
    assert not b.is_locked
