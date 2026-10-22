import contextlib
import os
import threading
from pathlib import Path

import pytest

from filelock import BaseFileLock, SoftFileLock, Timeout


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "prop.lock")


def test_lock_file_property_returns_path_string(lock_path):
    lock = SoftFileLock(lock_path)
    assert lock.lock_file == lock_path
    assert isinstance(lock.lock_file, str)


def test_lock_file_accepts_pathlib_path(tmp_path):
    path = tmp_path / "x.lock"
    lock = SoftFileLock(path)
    assert lock.lock_file == str(path)
    assert isinstance(lock.lock_file, str)


def test_default_timeout_is_negative_one(lock_path):
    assert SoftFileLock(lock_path).timeout == -1


def test_custom_timeout_in_constructor(lock_path):
    assert SoftFileLock(lock_path, timeout=2.5).timeout == 2.5


def test_timeout_setter_updates_value(lock_path):
    lock = SoftFileLock(lock_path)
    lock.timeout = 7
    assert lock.timeout == 7


def test_timeout_setter_converts_string(lock_path):
    lock = SoftFileLock(lock_path)
    lock.timeout = "3.5"
    assert lock.timeout == 3.5


def test_default_timeout_used_when_acquire_timeout_none(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path, timeout=0)
    with first:
        with pytest.raises(Timeout):
            second.acquire()


def test_explicit_timeout_overrides_default(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path, timeout=60)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)


def test_default_blocking_is_true(lock_path):
    assert SoftFileLock(lock_path).blocking is True


def test_blocking_constructor_false(lock_path):
    assert SoftFileLock(lock_path, blocking=False).blocking is False


def test_blocking_setter(lock_path):
    lock = SoftFileLock(lock_path)
    lock.blocking = False
    assert lock.blocking is False
    lock.blocking = True
    assert lock.blocking is True


def test_non_blocking_lock_raises_timeout_when_held(lock_path):
    first = SoftFileLock(lock_path)
    second = SoftFileLock(lock_path, blocking=False)
    with first:
        with pytest.raises(Timeout):
            second.acquire()


def test_default_mode_is_0o644(lock_path):
    lock = SoftFileLock(lock_path)
    assert lock.mode == 0o644
    assert lock.mode == 420


def test_custom_mode(lock_path):
    assert SoftFileLock(lock_path, mode=0o600).mode == 0o600


def test_default_thread_local_true(lock_path):
    assert SoftFileLock(lock_path).is_thread_local() is True


def test_thread_local_false(lock_path):
    assert SoftFileLock(lock_path, thread_local=False).is_thread_local() is False


def test_default_is_singleton_false(lock_path):
    assert SoftFileLock(lock_path).is_singleton() is False


def test_singleton_returns_same_instance(lock_path):
    a = SoftFileLock(lock_path, is_singleton=True)
    b = SoftFileLock(lock_path, is_singleton=True)
    assert a is b
    assert a.is_singleton() is True


def test_thread_local_lock_state_not_visible_in_other_thread(lock_path):
    lock = SoftFileLock(lock_path, thread_local=True)
    seen = []
    with lock:
        thread = threading.Thread(target=lambda: seen.append(lock.is_locked))
        thread.start()
        thread.join(timeout=10)
    assert seen == [False]


def test_non_thread_local_lock_state_visible_in_other_thread(lock_path):
    lock = SoftFileLock(lock_path, thread_local=False)
    seen = []
    with lock:
        thread = threading.Thread(target=lambda: seen.append(lock.is_locked))
        thread.start()
        thread.join(timeout=10)
    assert seen == [True]


def test_class_hierarchy():
    assert issubclass(SoftFileLock, BaseFileLock)
    assert issubclass(BaseFileLock, contextlib.ContextDecorator)

