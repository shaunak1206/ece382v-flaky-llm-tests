from __future__ import annotations

import gc
import logging
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from filelock import BaseFileLock, FileLock, Timeout, UnixFileLock

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="UnixFileLock requires fcntl")


@pytest.fixture
def lock_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.lock")


def test_acquire_sets_is_locked(lock_path):
    lock = UnixFileLock(lock_path)
    lock.acquire()
    try:
        assert lock.is_locked
        assert lock.lock_counter == 1
    finally:
        lock.release()


def test_release_unlocks(lock_path):
    lock = UnixFileLock(lock_path)
    lock.acquire()
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_lock_file_created_on_acquire(lock_path):
    lock = UnixFileLock(lock_path)
    assert not os.path.exists(lock_path)
    with lock:
        assert os.path.exists(lock_path)


def test_lock_file_persists_after_release(lock_path):
    lock = UnixFileLock(lock_path)
    with lock:
        pass
    assert os.path.exists(lock_path)


def test_context_manager_locks_and_unlocks(lock_path):
    lock = UnixFileLock(lock_path)
    with lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_context_manager_releases_on_exception(lock_path):
    lock = UnixFileLock(lock_path)
    with pytest.raises(RuntimeError, match="boom"):
        with lock:
            raise RuntimeError("boom")
    assert not lock.is_locked


def test_enter_returns_the_lock_itself(lock_path):
    lock = UnixFileLock(lock_path)
    with lock as entered:
        assert entered is lock


def test_nested_acquire_increments_counter(lock_path):
    lock = UnixFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    try:
        assert lock.lock_counter == 2
        assert lock.is_locked
    finally:
        lock.release(force=True)


def test_nested_release_keeps_lock_until_counter_zero(lock_path):
    lock = UnixFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    lock.release()
    assert lock.is_locked
    assert lock.lock_counter == 1
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_force_release_ignores_counter(lock_path):
    lock = UnixFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    lock.acquire()
    lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_release_when_not_locked_is_noop(lock_path):
    lock = UnixFileLock(lock_path)
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_second_lock_on_same_file_times_out(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)
    assert not second.is_locked


def test_timeout_exception_reports_lock_file(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    with first:
        with pytest.raises(Timeout) as excinfo:
            second.acquire(timeout=0)
    assert excinfo.value.lock_file == lock_path


def test_failed_acquire_restores_counter(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)
        assert second.lock_counter == 0


def test_blocking_false_raises_immediately_when_held(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    with first:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            second.acquire(blocking=False)
        assert time.perf_counter() - start < 5


def test_lock_can_be_acquired_after_other_releases(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    first.acquire()
    first.release()
    second.acquire(timeout=0)
    try:
        assert second.is_locked
    finally:
        second.release()


def test_timeout_waits_at_least_the_timeout(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    with first:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            second.acquire(timeout=0.1, poll_interval=0.01)
        assert time.perf_counter() - start >= 0.1


def test_default_timeout_is_used_by_acquire(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path, timeout=0)
    with first:
        with pytest.raises(Timeout):
            second.acquire()


def test_default_timeout_and_blocking_properties(lock_path):
    lock = UnixFileLock(lock_path)
    assert lock.timeout == -1
    assert lock.blocking is True
    assert lock.mode == 0o644


def test_timeout_setter_converts_to_float(lock_path):
    lock = UnixFileLock(lock_path)
    lock.timeout = "2.5"
    assert lock.timeout == 2.5
    lock.timeout = 3
    assert lock.timeout == 3.0
    assert isinstance(lock.timeout, float)


def test_blocking_setter_changes_default(lock_path):
    first = UnixFileLock(lock_path)
    second = UnixFileLock(lock_path)
    second.blocking = False
    assert second.blocking is False
    with first:
        with pytest.raises(Timeout):
            second.acquire()


def test_mode_is_applied_to_lock_file(lock_path):
    lock = UnixFileLock(lock_path, mode=0o600)
    assert lock.mode == 0o600
    with lock:
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600


def test_parent_directories_are_created(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "x.lock"
    lock = UnixFileLock(path)
    with lock:
        assert path.exists()


def test_pathlike_lock_file_is_converted_to_str(tmp_path):
    path = tmp_path / "p.lock"
    lock = UnixFileLock(path)
    assert lock.lock_file == str(path)
    assert isinstance(lock.lock_file, str)


def test_lock_works_as_decorator(lock_path):
    lock = UnixFileLock(lock_path)
    seen = []

    @lock
    def work():
        seen.append(lock.is_locked)
        return 42

    assert work() == 42
    assert seen == [True]
    assert not lock.is_locked


def test_acquire_returns_proxy_that_releases_on_exit(lock_path):
    lock = UnixFileLock(lock_path)
    with lock.acquire() as entered:
        assert entered is lock
        assert lock.is_locked
    assert not lock.is_locked


def test_deprecated_poll_intervall_warns(lock_path):
    lock = UnixFileLock(lock_path)
    with pytest.warns(DeprecationWarning, match="poll_interval"):
        lock.acquire(poll_intervall=0.01)
    lock.release()


def test_singleton_returns_same_instance(lock_path):
    first = UnixFileLock(lock_path, is_singleton=True)
    second = UnixFileLock(lock_path, is_singleton=True)
    assert first is second
    assert first.is_singleton


def test_singleton_with_differing_arguments_raises(lock_path):
    first = UnixFileLock(lock_path, is_singleton=True)
    with pytest.raises(ValueError, match="differing arguments"):
        UnixFileLock(lock_path, timeout=5, is_singleton=True)
    assert first.timeout == -1


def test_thread_local_is_default(lock_path):
    assert UnixFileLock(lock_path).is_thread_local() is True
    assert UnixFileLock(lock_path, thread_local=False).is_thread_local() is False


def test_thread_local_lock_state_not_visible_in_other_thread(lock_path):
    lock = UnixFileLock(lock_path, thread_local=True)
    results = []
    with lock:
        thread = threading.Thread(target=lambda: results.append(lock.is_locked))
        thread.start()
        thread.join()
        assert lock.is_locked
    assert results == [False]


def test_non_thread_local_lock_state_visible_in_other_thread(lock_path):
    lock = UnixFileLock(lock_path, thread_local=False)
    results = []
    with lock:
        thread = threading.Thread(target=lambda: results.append(lock.is_locked))
        thread.start()
        thread.join()
    assert results == [True]


def test_waiting_thread_acquires_after_release(lock_path):
    first = UnixFileLock(lock_path)
    acquired = threading.Event()

    def worker():
        other = UnixFileLock(lock_path)
        other.acquire(timeout=30, poll_interval=0.01)
        acquired.set()
        other.release()

    first.acquire()
    thread = threading.Thread(target=worker)
    thread.start()
    assert not acquired.wait(0.1)
    first.release()
    assert acquired.wait(30)
    thread.join()


def test_deleting_lock_releases_it(lock_path):
    first = UnixFileLock(lock_path)
    first.acquire()
    del first
    gc.collect()
    second = UnixFileLock(lock_path)
    second.acquire(timeout=0)
    try:
        assert second.is_locked
    finally:
        second.release()


def test_debug_logging_messages(lock_path, caplog):
    lock = UnixFileLock(lock_path)
    with caplog.at_level(logging.DEBUG, logger="filelock"):
        with lock:
            pass
    text = caplog.text
    assert "Attempting to acquire lock" in text
    assert "acquired on" in text
    assert "released on" in text


def test_filelock_alias_is_unix_lock_when_fcntl_available():
    assert issubclass(FileLock, BaseFileLock)
    assert FileLock is UnixFileLock
