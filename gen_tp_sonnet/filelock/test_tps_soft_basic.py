import threading
import time

import pytest

import filelock
from filelock import SoftFileLock, Timeout


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "test.lock")


def test_new_lock_is_not_locked(lock_path):
    lock = SoftFileLock(lock_path)
    assert lock.is_locked is False
    assert lock.lock_counter == 0


def test_acquire_sets_locked(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    try:
        assert lock.is_locked
        assert lock.lock_counter == 1
    finally:
        lock.release()


def test_release_clears_locked(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_release_without_acquire_is_noop(lock_path):
    lock = SoftFileLock(lock_path)
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_context_manager_acquires_and_releases(lock_path):
    lock = SoftFileLock(lock_path)
    with lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_context_manager_enter_returns_lock(lock_path):
    lock = SoftFileLock(lock_path)
    with lock as entered:
        assert entered is lock


def test_context_manager_releases_on_exception(lock_path):
    lock = SoftFileLock(lock_path)
    with pytest.raises(RuntimeError):
        with lock:
            assert lock.is_locked
            raise RuntimeError("boom")
    assert not lock.is_locked


def test_acquire_returns_proxy_usable_as_context_manager(lock_path):
    lock = SoftFileLock(lock_path)
    with lock.acquire() as entered:
        assert entered is lock
        assert lock.is_locked
    assert not lock.is_locked


def test_reentrant_acquire_increments_counter(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    try:
        assert lock.lock_counter == 2
        assert lock.is_locked
    finally:
        lock.release(force=True)


def test_nested_release_keeps_lock_until_counter_zero(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    lock.release()
    assert lock.is_locked
    assert lock.lock_counter == 1
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_force_release_ignores_counter(lock_path):
    lock = SoftFileLock(lock_path)
    lock.acquire()
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 3
    lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_nested_context_managers(lock_path):
    lock = SoftFileLock(lock_path)
    with lock:
        with lock:
            assert lock.lock_counter == 2
        assert lock.is_locked
        assert lock.lock_counter == 1
    assert not lock.is_locked


def test_lock_can_be_reacquired_after_release(lock_path):
    lock = SoftFileLock(lock_path)
    for _ in range(3):
        lock.acquire()
        assert lock.is_locked
        lock.release()
        assert not lock.is_locked


def test_second_lock_object_times_out_when_held(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)
    assert not second.is_locked


def test_second_lock_can_acquire_after_first_released(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path)
    with first:
        pass
    with second:
        assert second.is_locked


def test_acquire_timeout_waits_roughly_timeout(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path)
    with first:
        start = time.monotonic()
        with pytest.raises(Timeout):
            second.acquire(timeout=0.2, poll_interval=0.01)
        elapsed = time.monotonic() - start
    assert elapsed >= 0.15


def test_acquire_non_blocking_raises_when_held(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(blocking=False)


def test_thread_waits_until_lock_is_released(lock_path):
    holder = SoftFileLock(lock_path)
    result = {}

    def worker():
        other = SoftFileLock(lock_path)
        other.acquire(timeout=5, poll_interval=0.01)
        result["acquired"] = other.is_locked
        other.release()

    holder.acquire()
    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.1)
    holder.release()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert result == {"acquired": True}


def test_lock_works_as_decorator(lock_path):
    lock = SoftFileLock(lock_path)
    seen = []

    @lock
    def func():
        seen.append(lock.is_locked)
        return 42

    assert func() == 42
    assert seen == [True]
    assert not lock.is_locked


def test_decorator_releases_when_function_raises(lock_path):
    lock = SoftFileLock(lock_path)

    @lock
    def func():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        func()
    assert not lock.is_locked


def test_package_exports_core_names():
    assert filelock.SoftFileLock is SoftFileLock
    assert filelock.Timeout is Timeout
    assert issubclass(SoftFileLock, filelock.BaseFileLock)
