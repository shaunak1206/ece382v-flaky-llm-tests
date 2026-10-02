"""Timeout, blocking and polling behaviour."""
import pickle
import threading
import time

import pytest

from filelock import FileLock, SoftFileLock, Timeout


@pytest.fixture(params=[FileLock, SoftFileLock], ids=["FileLock", "SoftFileLock"])
def lock_cls(request):
    return request.param


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "timeout.lock")


@pytest.fixture
def held(lock_cls, lock_path):
    """A lock that is currently held by another lock object."""
    holder = lock_cls(lock_path)
    holder.acquire()
    yield holder
    holder.release(force=True)


def test_timeout_exception_attributes():
    exc = Timeout("some/file.lock")
    assert isinstance(exc, TimeoutError)
    assert exc.lock_file == "some/file.lock"
    assert "some/file.lock" in str(exc)
    assert "could not be acquired" in str(exc)
    assert repr(exc) == "Timeout('some/file.lock')"


def test_timeout_exception_can_be_pickled():
    restored = pickle.loads(pickle.dumps(Timeout("a.lock")))
    assert isinstance(restored, Timeout)
    assert restored.lock_file == "a.lock"
    assert str(restored) == str(Timeout("a.lock"))


def test_acquire_raises_timeout_after_waiting(lock_cls, lock_path, held):
    other = lock_cls(lock_path)
    start = time.perf_counter()
    with pytest.raises(Timeout) as info:
        other.acquire(timeout=0.2, poll_interval=0.01)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.2
    assert elapsed < 10
    assert info.value.lock_file == lock_path


def test_default_timeout_from_constructor_is_used(lock_cls, lock_path, held):
    other = lock_cls(lock_path, timeout=0.1)
    with pytest.raises(Timeout):
        other.acquire(poll_interval=0.01)


def test_explicit_timeout_overrides_the_default(lock_cls, lock_path, held):
    other = lock_cls(lock_path, timeout=3600)
    start = time.perf_counter()
    with pytest.raises(Timeout):
        other.acquire(timeout=0.1, poll_interval=0.01)
    assert time.perf_counter() - start < 30


def test_zero_timeout_makes_a_single_attempt(lock_cls, lock_path, held):
    other = lock_cls(lock_path)
    with pytest.raises(Timeout):
        other.acquire(timeout=0)
    held.release()
    other.acquire(timeout=0)
    assert other.is_locked
    other.release()


def test_failed_acquire_leaves_counter_untouched(lock_cls, lock_path, held):
    other = lock_cls(lock_path)
    with pytest.raises(Timeout):
        other.acquire(timeout=0.05, poll_interval=0.01)
    assert other.lock_counter == 0
    assert not other.is_locked
    assert held.is_locked
    assert held.lock_counter == 1


def test_timeout_inside_a_with_statement(lock_cls, lock_path, held):
    other = lock_cls(lock_path, timeout=0.1)
    with pytest.raises(Timeout):
        with other:
            pytest.fail("should not get the lock")


def test_non_blocking_acquire_fails_fast(lock_cls, lock_path, held):
    other = lock_cls(lock_path)
    start = time.perf_counter()
    with pytest.raises(Timeout):
        other.acquire(blocking=False)
    assert time.perf_counter() - start < 1.0
    assert not other.is_locked


def test_non_blocking_acquire_succeeds_on_a_free_lock(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire(blocking=False)
    assert lock.is_locked
    lock.release()


def test_blocking_constructor_argument_and_override(lock_cls, lock_path, held):
    other = lock_cls(lock_path, blocking=False)
    assert other.blocking is False
    with pytest.raises(Timeout):
        other.acquire()
    start = time.perf_counter()
    with pytest.raises(Timeout):
        other.acquire(blocking=True, timeout=0.15, poll_interval=0.01)
    assert time.perf_counter() - start >= 0.15
    other.blocking = True
    assert other.blocking is True


def test_blocked_acquire_succeeds_once_the_holder_releases(lock_cls, lock_path):
    acquired = threading.Event()
    go = threading.Event()

    def holder():
        lock = lock_cls(lock_path)
        with lock:
            acquired.set()
            go.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert acquired.wait(10)
        timer = threading.Timer(0.2, go.set)
        timer.start()
        waiter = lock_cls(lock_path)
        waiter.acquire(poll_interval=0.01)
        assert waiter.is_locked
        waiter.release()
        timer.join()
    finally:
        go.set()
        thread.join(10)


def test_deprecated_poll_intervall_keyword_warns(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.warns(DeprecationWarning):
        lock.acquire(poll_intervall=0.01)
    assert lock.is_locked
    lock.release()
