"""Basic acquire/release, context manager, decorator and properties."""
from __future__ import annotations

import os
import pickle
import stat
import sys
from pathlib import Path

import pytest

from filelock import (
    AcquireReturnProxy,
    BaseFileLock,
    FileLock,
    SoftFileLock,
    Timeout,
    UnixFileLock,
    __version__,
)

LOCK_CLASSES = [FileLock, SoftFileLock]


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "test.lock"


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_new_lock_is_not_locked(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_acquire_and_release(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    lock.acquire()
    assert lock.is_locked
    assert lock.lock_counter == 1
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_acquire_returns_proxy(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    proxy = lock.acquire()
    try:
        assert isinstance(proxy, AcquireReturnProxy)
        assert proxy.lock is lock
    finally:
        lock.release()


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_context_manager(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    with lock as acquired:
        assert acquired is lock
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_acquire_as_context_manager_releases(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    with lock.acquire() as inner:
        assert inner is lock
        assert lock.is_locked
        assert lock.lock_counter == 1
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_context_manager_releases_on_exception(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    with pytest.raises(RuntimeError, match="boom"):
        with lock:
            raise RuntimeError("boom")
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_decorator_holds_lock_during_call(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    seen = []

    @lock
    def work(x):
        seen.append(lock.is_locked)
        return x * 2

    assert work(21) == 42
    assert seen == [True]
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_decorator_releases_on_exception(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))

    @lock
    def fail():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        fail()
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_release_without_acquire_is_noop(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_lock_file_property_is_string(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert lock.lock_file == str(lock_path)
    assert isinstance(lock.lock_file, str)


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_lock_file_accepts_str(lock_cls, lock_path):
    lock = lock_cls(str(lock_path))
    assert lock.lock_file == os.fspath(lock_path)


def test_lock_file_is_created(lock_path):
    with FileLock(lock_path):
        assert lock_path.exists()


def test_soft_lock_file_removed_on_release(lock_path):
    lock = SoftFileLock(lock_path)
    with lock:
        assert lock_path.exists()
    assert not lock_path.exists()


def test_unix_lock_file_kept_on_release(lock_path):
    lock = UnixFileLock(lock_path)
    with lock:
        assert lock_path.exists()
    assert lock_path.exists()


def test_parent_directories_are_created(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "x.lock"
    with FileLock(nested):
        assert nested.exists()
    assert nested.parent.is_dir()


def test_soft_lock_parent_directories_are_created(tmp_path):
    nested = tmp_path / "p" / "q" / "x.lock"
    with SoftFileLock(nested):
        assert nested.exists()


def test_filelock_is_basefilelock_subclass():
    assert issubclass(FileLock, BaseFileLock)
    assert issubclass(SoftFileLock, BaseFileLock)


@pytest.mark.skipif(sys.platform == "win32", reason="unix only")
def test_filelock_alias_on_unix_is_unix_lock():
    assert FileLock is UnixFileLock


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__


def test_base_class_cannot_be_instantiated(lock_path):
    with pytest.raises(TypeError):
        BaseFileLock(lock_path)


@pytest.mark.skipif(sys.platform == "win32", reason="posix modes")
@pytest.mark.parametrize("mode", [0o600, 0o644, 0o640])
def test_mode_applied_to_lock_file(tmp_path, mode):
    path = tmp_path / "m.lock"
    lock = FileLock(path, mode=mode)
    assert lock.mode == mode
    with lock:
        assert stat.S_IMODE(path.stat().st_mode) == mode


@pytest.mark.skipif(sys.platform == "win32", reason="posix modes")
def test_soft_lock_mode_respected(tmp_path):
    path = tmp_path / "soft.lock"
    lock = SoftFileLock(path, mode=0o600)
    with lock:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_default_mode_is_0o644(lock_path):
    assert FileLock(lock_path).mode == 0o644


def test_default_timeout_is_negative(lock_path):
    assert FileLock(lock_path).timeout < 0


def test_timeout_setter_converts_to_float(lock_path):
    lock = FileLock(lock_path)
    lock.timeout = "2.5"
    assert lock.timeout == 2.5
    lock.timeout = 3
    assert isinstance(lock.timeout, float)


def test_blocking_setter(lock_path):
    lock = FileLock(lock_path)
    assert lock.blocking is True
    lock.blocking = False
    assert lock.blocking is False


def test_timeout_exception_attributes():
    exc = Timeout("/tmp/some.lock")
    assert exc.lock_file == "/tmp/some.lock"
    assert "/tmp/some.lock" in str(exc)
    assert "could not be acquired" in str(exc)
    assert repr(exc) == "Timeout('/tmp/some.lock')"


def test_timeout_is_timeouterror_subclass():
    assert issubclass(Timeout, TimeoutError)


def test_timeout_is_picklable():
    exc = pickle.loads(pickle.dumps(Timeout("/x/y.lock")))
    assert isinstance(exc, Timeout)
    assert exc.lock_file == "/x/y.lock"
