"""Reentrant locking, lock counter, timeouts, blocking and poll interval."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pytest

from filelock import FileLock, SoftFileLock, Timeout

LOCK_CLASSES = [FileLock, SoftFileLock]


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "re.lock"


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_reentrant_counter_increments(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 3
    assert lock.is_locked
    lock.release()
    assert lock.lock_counter == 2
    assert lock.is_locked
    lock.release()
    assert lock.is_locked
    lock.release()
    assert lock.lock_counter == 0
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_nested_with_blocks(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with lock:
        with lock:
            assert lock.lock_counter == 2
        assert lock.is_locked
        assert lock.lock_counter == 1
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_release_force_ignores_counter(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    lock.acquire()
    lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_extra_release_is_harmless(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    lock.release()
    lock.release()
    assert lock.lock_counter == 0
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_second_lock_object_times_out(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    with first:
        with pytest.raises(Timeout) as info:
            second.acquire(timeout=0.1, poll_interval=0.01)
        assert info.value.lock_file == str(lock_path)
    assert not second.is_locked
    assert second.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_timeout_zero_tries_once(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_blocking_false_raises_immediately(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(blocking=False)
    assert second.lock_counter == 0


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_blocking_false_succeeds_when_free(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire(blocking=False)
    try:
        assert lock.is_locked
    finally:
        lock.release()


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_blocking_constructor_argument(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path, blocking=False)
    assert second.blocking is False
    with first:
        with pytest.raises(Timeout):
            second.acquire()


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_default_timeout_from_constructor(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path, timeout=0.1)
    assert second.timeout == 0.1
    with first:
        with pytest.raises(Timeout):
            with second:
                pass


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_acquire_timeout_overrides_default(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path, timeout=100)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0.05, poll_interval=0.01)


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_timeout_waits_at_least_timeout(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    with first:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            second.acquire(timeout=0.2, poll_interval=0.02)
        elapsed = time.perf_counter() - start
    assert elapsed >= 0.2


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_acquire_after_release_by_other_lock(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    first.acquire()
    first.release()
    second.acquire(timeout=1)
    try:
        assert second.is_locked
    finally:
        second.release()


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_failed_acquire_does_not_break_holder(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0.05, poll_interval=0.01)
        assert first.is_locked
        assert first.lock_counter == 1


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_reentrant_failed_acquire_keeps_counter(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    try:
        lock.acquire(timeout=0)
        assert lock.lock_counter == 2
    finally:
        lock.release(force=True)


def test_poll_intervall_deprecated_alias(lock_path):
    first = FileLock(lock_path)
    second = FileLock(lock_path)
    with first:
        with pytest.warns(DeprecationWarning, match="poll_interval"):
            with pytest.raises(Timeout):
                second.acquire(timeout=0.05, poll_intervall=0.01)


def test_poll_interval_no_warning(lock_path):
    lock = FileLock(lock_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lock.acquire(poll_interval=0.01)
    lock.release()


def test_soft_lock_stale_file_blocks(lock_path):
    lock_path.write_text("")
    lock = SoftFileLock(lock_path)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0.05, poll_interval=0.01)
    assert not lock.is_locked


def test_soft_lock_acquires_after_stale_file_removed(lock_path):
    lock_path.write_text("")
    lock = SoftFileLock(lock_path)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0)
    lock_path.unlink()
    with lock:
        assert lock.is_locked


def test_soft_lock_read_only_existing_file_raises_permission_error(lock_path):
    lock_path.write_text("")
    lock_path.chmod(0o400)
    try:
        lock = SoftFileLock(lock_path)
        with pytest.raises(PermissionError):
            lock.acquire(timeout=0)
        assert lock.lock_counter == 0
    finally:
        lock_path.chmod(0o600)


def test_soft_lock_directory_path_raises(tmp_path):
    lock = SoftFileLock(tmp_path)
    with pytest.raises(OSError):
        lock.acquire(timeout=0)
    assert lock.lock_counter == 0
    assert not lock.is_locked
