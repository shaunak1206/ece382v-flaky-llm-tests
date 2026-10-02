"""Basic acquire/release behaviour of FileLock and SoftFileLock."""
import gc
import os
import stat
import sys

import pytest

import filelock
from filelock import AcquireReturnProxy, BaseFileLock, FileLock, SoftFileLock, Timeout, UnixFileLock


@pytest.fixture(params=[FileLock, SoftFileLock], ids=["FileLock", "SoftFileLock"])
def lock_cls(request):
    return request.param


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "test.lock")


def test_acquire_and_release(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert not lock.is_locked
    lock.acquire()
    assert lock.is_locked
    lock.release()
    assert not lock.is_locked


def test_lock_can_be_reused_many_times(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    for _ in range(3):
        lock.acquire()
        assert lock.is_locked
        lock.release()
        assert not lock.is_locked


def test_acquire_returns_a_proxy_for_the_lock(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    proxy = lock.acquire()
    assert isinstance(proxy, AcquireReturnProxy)
    assert proxy.lock is lock
    lock.release()


def test_proxy_works_as_a_context_manager(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with lock.acquire() as held:
        assert held is lock
        assert lock.is_locked
    assert not lock.is_locked


def test_lock_as_context_manager(lock_cls, lock_path):
    with lock_cls(lock_path) as lock:
        assert isinstance(lock, lock_cls)
        assert lock.is_locked
    assert not lock.is_locked


def test_context_manager_releases_on_exception(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.raises(RuntimeError):
        with lock:
            assert lock.is_locked
            raise RuntimeError("boom")
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_lock_as_decorator(lock_cls, lock_path):
    lock = lock_cls(lock_path)

    @lock
    def work(x):
        assert lock.is_locked
        return x * 2

    assert work(21) == 42
    assert not lock.is_locked


def test_decorator_releases_when_function_raises(lock_cls, lock_path):
    lock = lock_cls(lock_path)

    @lock
    def failing():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        failing()
    assert not lock.is_locked


def test_reentrant_locking_counts_acquisitions(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert lock.lock_counter == 0
    lock.acquire()
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 3
    lock.release()
    assert lock.lock_counter == 2
    assert lock.is_locked
    lock.release()
    assert lock.is_locked
    lock.release()
    assert lock.lock_counter == 0
    assert not lock.is_locked


def test_nested_with_blocks_on_same_lock(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with lock:
        with lock:
            assert lock.lock_counter == 2
        assert lock.lock_counter == 1
        assert lock.is_locked
    assert lock.lock_counter == 0
    assert not lock.is_locked


def test_release_when_not_locked_is_a_noop(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.release()
    lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_force_release_ignores_the_counter(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 2
    lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


def test_second_lock_object_is_blocked_while_first_holds(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    first.acquire()
    with pytest.raises(Timeout):
        second.acquire(timeout=0.1, poll_interval=0.01)
    assert not second.is_locked
    first.release()
    second.acquire(timeout=1, poll_interval=0.01)
    assert second.is_locked
    second.release()


def test_nested_acquire_keeps_other_locks_out_until_fully_released(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    first.acquire()
    first.acquire()
    first.release()
    with pytest.raises(Timeout):
        second.acquire(timeout=0.1, poll_interval=0.01)
    first.release()
    second.acquire(timeout=1, poll_interval=0.01)
    assert second.is_locked
    second.release()


def test_locks_on_different_files_do_not_interfere(lock_cls, tmp_path):
    a = lock_cls(str(tmp_path / "a.lock"))
    b = lock_cls(str(tmp_path / "b.lock"))
    a.acquire()
    b.acquire(timeout=0.5, poll_interval=0.01)
    assert a.is_locked and b.is_locked
    a.release()
    assert b.is_locked
    b.release()


def test_default_properties(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert lock.lock_file == lock_path
    assert lock.timeout == -1
    assert lock.blocking is True
    assert lock.mode == 0o644
    assert lock.is_thread_local() is True
    assert lock.is_singleton is False
    assert isinstance(lock, BaseFileLock)


def test_timeout_setter_converts_to_float(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.timeout = "2.5"
    assert lock.timeout == 2.5
    lock.timeout = 3
    assert lock.timeout == 3.0
    assert isinstance(lock.timeout, float)


def test_accepts_path_like_lock_file(lock_cls, tmp_path):
    path = tmp_path / "pathlike.lock"
    lock = lock_cls(path)
    assert lock.lock_file == str(path)
    with lock:
        assert os.path.exists(path)


def test_missing_parent_directories_are_created(lock_cls, tmp_path):
    path = tmp_path / "x" / "y" / "z" / "nested.lock"
    lock = lock_cls(str(path))
    with lock:
        assert path.parent.is_dir()
        assert path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="posix permissions")
def test_unix_lock_applies_the_requested_mode(lock_path):
    lock = UnixFileLock(lock_path, mode=0o600)
    assert lock.mode == 0o600
    with lock:
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600


def test_garbage_collected_lock_is_released(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    lock.acquire()
    del lock
    gc.collect()
    other = lock_cls(lock_path)
    other.acquire(timeout=1, poll_interval=0.01)
    assert other.is_locked
    other.release()


def test_filelock_alias_is_a_lock_class(lock_path):
    assert issubclass(FileLock, BaseFileLock)
    assert issubclass(SoftFileLock, BaseFileLock)
    assert isinstance(filelock.__version__, str)
