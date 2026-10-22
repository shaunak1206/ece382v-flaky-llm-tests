from __future__ import annotations

import os
import pickle
import stat
import sys
from pathlib import Path

import pytest

import filelock
from filelock import BaseFileLock, SoftFileLock, Timeout, WindowsFileLock
from filelock._util import ensure_directory_exists, raise_on_not_writable_file


@pytest.fixture
def lock_path(tmp_path: Path) -> str:
    return str(tmp_path / "soft.lock")


# SoftFileLock


def test_soft_acquire_creates_lock_file(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    try:
        assert os.path.isfile(lock_path)
        assert lock.is_locked
    finally:
        lock.release()


def test_soft_release_removes_lock_file(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.release()
    assert not os.path.exists(lock_path)
    assert not lock.is_locked


def test_soft_context_manager_removes_file_on_exit(lock_path):
    lock = SoftFileLock(lock_path)
    with lock:
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


def test_soft_second_lock_times_out(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path)
    with first:
        with pytest.raises(Timeout) as excinfo:
            second.acquire(timeout=0)
        assert excinfo.value.lock_file == lock_path
    assert not second.is_locked
    assert second.lock_counter == 0


def test_soft_existing_file_blocks_acquire(lock_path):
    Path(lock_path).write_text("")
    lock = SoftFileLock(lock_path)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0)
    assert os.path.exists(lock_path)


def test_soft_acquire_succeeds_after_stale_file_removed(lock_path):
    Path(lock_path).write_text("")
    lock = SoftFileLock(lock_path)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0)
    os.remove(lock_path)
    lock.acquire(timeout=0)
    try:
        assert lock.is_locked
    finally:
        lock.release()


def test_soft_release_tolerates_deleted_file(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    os.remove(lock_path)
    lock.release()
    assert not lock.is_locked


def test_soft_nested_acquire_and_release(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 2
    lock.release()
    assert os.path.exists(lock_path)
    lock.release()
    assert not os.path.exists(lock_path)


def test_soft_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "x" / "y" / "z.lock"
    with SoftFileLock(path):
        assert path.exists()
    assert not path.exists()
    assert path.parent.is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="posix permissions")
def test_soft_lock_mode_applied(lock_path):
    mask = os.umask(0)
    os.umask(mask)
    lock = SoftFileLock(lock_path, mode=0o666)
    assert lock.mode == 0o666
    with lock:
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o666 & ~mask


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs non-root posix")
def test_soft_readonly_existing_file_raises_permission_error(lock_path):
    Path(lock_path).write_text("")
    os.chmod(lock_path, 0o444)
    lock = SoftFileLock(lock_path)
    with pytest.raises(PermissionError):
        lock.acquire(timeout=0)
    assert lock.lock_counter == 0


@pytest.mark.skipif(sys.platform == "win32", reason="posix only")
def test_soft_directory_path_raises_is_a_directory_error(tmp_path):
    directory = tmp_path / "adir"
    directory.mkdir()
    lock = SoftFileLock(str(directory))
    with pytest.raises(IsADirectoryError):
        lock.acquire(timeout=0)


def test_soft_parent_is_a_file_raises_oserror(tmp_path):
    parent = tmp_path / "afile"
    parent.write_text("data")
    lock = SoftFileLock(str(parent / "child.lock"))
    with pytest.raises(OSError):
        lock.acquire(timeout=0)
    assert lock.lock_counter == 0


def test_soft_singleton_returns_same_instance(lock_path):
    first = SoftFileLock(lock_path, is_singleton=True)
    second = SoftFileLock(lock_path, is_singleton=True)
    assert first is second


def test_soft_lock_works_as_decorator(lock_path):
    lock = SoftFileLock(lock_path)

    @lock
    def check():
        return os.path.exists(lock_path)

    assert check() is True
    assert not os.path.exists(lock_path)


# Windows lock stub on posix


@pytest.mark.skipif(sys.platform == "win32", reason="stub only on non-windows")
def test_windows_lock_not_usable_on_posix(lock_path):
    lock = WindowsFileLock(lock_path)
    with pytest.raises(NotImplementedError):
        lock.acquire()
    assert lock.lock_counter == 0
    assert not lock.is_locked


# Timeout error


def test_timeout_is_timeout_error_subclass():
    assert issubclass(Timeout, TimeoutError)


def test_timeout_str_mentions_lock_file():
    err = Timeout("/tmp/some.lock")
    assert str(err) == "The file lock '/tmp/some.lock' could not be acquired."


def test_timeout_repr():
    assert repr(Timeout("/tmp/some.lock")) == "Timeout('/tmp/some.lock')"


def test_timeout_lock_file_property():
    assert Timeout("abc").lock_file == "abc"


def test_timeout_pickle_roundtrip():
    original = Timeout("/tmp/pickled.lock")
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, Timeout)
    assert restored.lock_file == "/tmp/pickled.lock"
    assert str(restored) == str(original)


# util helpers


def test_ensure_directory_exists_creates_nested(tmp_path):
    target = tmp_path / "one" / "two" / "file.lock"
    ensure_directory_exists(target)
    assert target.parent.is_dir()
    assert not target.exists()


def test_raise_on_not_writable_ignores_missing_file(tmp_path):
    raise_on_not_writable_file(str(tmp_path / "missing"))


def test_raise_on_not_writable_accepts_writable_file(tmp_path):
    path = tmp_path / "ok.txt"
    path.write_text("x")
    raise_on_not_writable_file(str(path))


@pytest.mark.skipif(sys.platform == "win32", reason="posix only")
def test_raise_on_not_writable_rejects_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        raise_on_not_writable_file(str(tmp_path))


# package level


def test_version_is_string_and_exported():
    assert isinstance(filelock.__version__, str)
    assert filelock.__version__.count(".") >= 1
