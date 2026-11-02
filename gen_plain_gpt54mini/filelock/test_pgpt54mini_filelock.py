import asyncio
import gc
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import filelock
from filelock import (
    AcquireReturnProxy,
    AsyncAcquireReturnProxy,
    AsyncSoftFileLock,
    BaseAsyncFileLock,
    BaseFileLock,
    FileLock,
    SoftFileLock,
    Timeout,
)
from filelock._api import FileLockContext
from filelock._util import ensure_directory_exists, raise_on_not_writable_file


class DummyLock(BaseFileLock):
    def __init__(self, *args, **kwargs):
        self.acquire_calls = 0
        self.release_calls = 0
        super().__init__(*args, **kwargs)

    def _acquire(self) -> None:
        self.acquire_calls += 1
        self._context.lock_file_fd = 10

    def _release(self) -> None:
        self.release_calls += 1
        self._context.lock_file_fd = None


class DummyAsyncLock(BaseAsyncFileLock):
    def __init__(self, *args, **kwargs):
        self.acquire_calls = 0
        self.release_calls = 0
        super().__init__(*args, **kwargs)

    def _acquire(self) -> None:
        self.acquire_calls += 1
        self._context.lock_file_fd = 11

    def _release(self) -> None:
        self.release_calls += 1
        self._context.lock_file_fd = None


@pytest.fixture()
def tmp_lock_path(tmp_path):
    return tmp_path / "test.lock"


def test_version_and_all_exports():
    assert isinstance(filelock.__version__, str)
    assert filelock.__version__ == filelock.version
    assert "FileLock" in filelock.__all__
    assert "Timeout" in filelock.__all__


def test_timeout_str_repr_and_property():
    exc = Timeout("/tmp/x.lock")
    assert str(exc) == "The file lock '/tmp/x.lock' could not be acquired."
    assert repr(exc) == "Timeout('/tmp/x.lock')"
    assert exc.lock_file == "/tmp/x.lock"


def test_timeout_pickle_roundtrip():
    exc = Timeout("/tmp/x.lock")
    restored = pickle.loads(pickle.dumps(exc))
    assert isinstance(restored, Timeout)
    assert restored.lock_file == "/tmp/x.lock"


def test_acquire_return_proxy_enters_and_exits(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock._context.lock_file_fd = 1
    proxy = AcquireReturnProxy(lock)
    assert proxy.__enter__() is lock
    proxy.__exit__(None, None, None)
    assert lock.release_calls == 1


def test_async_acquire_return_proxy_enters_and_exits(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path)
    lock._context.lock_file_fd = 1
    proxy = AsyncAcquireReturnProxy(lock)

    async def run():
        assert await proxy.__aenter__() is lock
        await proxy.__aexit__(None, None, None)

    asyncio.run(run())
    assert lock.release_calls == 1


def test_baselock_properties_and_setters(tmp_lock_path):
    lock = DummyLock(tmp_lock_path, timeout=1.5, mode=0o600, blocking=False)
    assert lock.lock_file == str(tmp_lock_path)
    assert lock.timeout == 1.5
    assert lock.mode == 0o600
    assert lock.blocking is False
    assert lock.is_locked is False
    assert lock.lock_counter == 0
    assert lock.is_thread_local() is True
    assert lock.is_singleton is False

    lock.timeout = "2.25"
    lock.blocking = True
    assert lock.timeout == 2.25
    assert lock.blocking is True


def test_baselock_context_uses_threadlocal_by_default(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    assert isinstance(lock._context, FileLockContext)
    assert lock._context.lock_file == str(tmp_lock_path)


def test_baselock_context_can_be_non_thread_local(tmp_lock_path):
    lock = DummyLock(tmp_lock_path, thread_local=False)
    assert not lock.is_thread_local()
    assert isinstance(lock._context, FileLockContext)
    assert type(lock._context) is FileLockContext


def test_baselock_enter_and_exit_release(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    with lock as entered:
        assert entered is lock
        assert lock.is_locked
    assert lock.release_calls == 1


def test_baselock_release_without_lock_is_noop(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock.release()
    assert lock.release_calls == 0


def test_baselock_release_force_releases_when_locked(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock._context.lock_file_fd = 1
    lock._context.lock_counter = 3
    lock.release(force=True)
    assert lock.release_calls == 1
    assert lock.lock_counter == 0
    assert lock.is_locked is False


def test_baselock_double_acquire_increments_counter(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock.acquire()
    lock.acquire()
    assert lock.acquire_calls == 1
    assert lock.lock_counter == 2
    lock.release()
    assert lock.lock_counter == 1
    assert lock.is_locked


def test_baselock_release_only_finally_unlocks_nested_lock(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock.acquire()
    lock.acquire()
    lock.release()
    assert lock.release_calls == 0
    lock.release()
    assert lock.release_calls == 1
    assert lock.lock_counter == 0


def test_baselock_acquire_returns_proxy_and_sets_counter(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    proxy = lock.acquire()
    assert isinstance(proxy, AcquireReturnProxy)
    assert lock.lock_counter == 1
    assert lock.is_locked


def test_baselock_acquire_nonblocking_timeout(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock._acquire = lambda: None  # never acquires
    with pytest.raises(Timeout):
        lock.acquire(blocking=False)
    assert lock.lock_counter == 0


def test_baselock_acquire_uses_default_timeout_when_none(tmp_lock_path, monkeypatch):
    lock = DummyLock(tmp_lock_path, timeout=0.1)
    times = iter([0.0, 0.2])
    monkeypatch.setattr("filelock._api.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("filelock._api.time.sleep", lambda x: None)
    lock._acquire = lambda: None
    with pytest.raises(Timeout):
        lock.acquire(timeout=None)
    assert lock.lock_counter == 0


def test_baselock_acquire_poll_intervall_deprecation(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    with pytest.warns(DeprecationWarning):
        proxy = lock.acquire(poll_intervall=0.01)
    assert isinstance(proxy, AcquireReturnProxy)


def test_baselock_acquire_poll_intervall_overrides_poll_interval(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    recorded = {}

    def fake_sleep(value):
        recorded["sleep"] = value
        raise RuntimeError("stop")

    lock._acquire = lambda: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("filelock._api.time.sleep", fake_sleep)
    try:
        with pytest.raises(RuntimeError):
            lock.acquire(poll_interval=0.2, poll_intervall=0.05)
    finally:
        monkeypatch.undo()
    assert recorded["sleep"] == 0.05


def test_baselock_acquire_with_exception_decrements_counter(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)

    def boom():
        raise ValueError("x")

    lock._acquire = boom
    with pytest.raises(ValueError):
        lock.acquire()
    assert lock.lock_counter == 0


def test_baselock_del_force_release(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock._context.lock_file_fd = 1
    lock._context.lock_counter = 2
    lock.__del__()
    assert lock.release_calls == 1
    assert lock.lock_counter == 0


def test_singleton_same_args_returns_same_instance(tmp_lock_path):
    a = DummyLock(tmp_lock_path, is_singleton=True)
    b = DummyLock(tmp_lock_path, is_singleton=True)
    assert a is b


def test_singleton_differing_timeout_raises(tmp_lock_path):
    a = DummyLock(tmp_lock_path, is_singleton=True, timeout=1)
    with pytest.raises(ValueError):
        DummyLock(tmp_lock_path, is_singleton=True, timeout=2)
    assert a.is_singleton


def test_singleton_differing_blocking_raises(tmp_lock_path):
    DummyLock(tmp_lock_path, is_singleton=True, blocking=False)
    with pytest.raises(ValueError):
        DummyLock(tmp_lock_path, is_singleton=True, blocking=True)


def test_singleton_differing_thread_local_raises(tmp_lock_path):
    DummyLock(tmp_lock_path, is_singleton=True, thread_local=False)
    with pytest.raises(ValueError):
        DummyLock(tmp_lock_path, is_singleton=True, thread_local=True)


def test_singleton_differing_mode_raises(tmp_lock_path):
    DummyLock(tmp_lock_path, is_singleton=True, mode=0o600)
    with pytest.raises(ValueError):
        DummyLock(tmp_lock_path, is_singleton=True, mode=0o644)


def test_util_ensure_directory_exists_creates_parent(tmp_path):
    target = tmp_path / "a" / "b" / "file.lock"
    ensure_directory_exists(target)
    assert (tmp_path / "a" / "b").is_dir()


def test_util_raise_on_not_writable_file_missing_file_no_error(tmp_path):
    raise_on_not_writable_file(str(tmp_path / "missing.lock"))


def test_util_raise_on_not_writable_file_writable_file_no_error(tmp_path):
    f = tmp_path / "x.lock"
    f.write_text("x")
    raise_on_not_writable_file(str(f))


def test_util_raise_on_not_writable_file_directory_raises(monkeypatch, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    monkeypatch.setattr(os, "stat", lambda _: SimpleNamespace(st_mtime=1, st_mode=0o040755))
    if sys.platform == "win32":
        with pytest.raises(PermissionError):
            raise_on_not_writable_file(str(d))
    else:
        with pytest.raises(IsADirectoryError):
            raise_on_not_writable_file(str(d))


def test_util_raise_on_not_writable_file_not_writable_raises(monkeypatch, tmp_path):
    f = tmp_path / "x.lock"
    f.write_text("x")
    monkeypatch.setattr(os, "stat", lambda _: SimpleNamespace(st_mtime=1, st_mode=0o100444))
    with pytest.raises(PermissionError):
        raise_on_not_writable_file(str(f))


def test_softfilelock_acquire_and_release(tmp_lock_path):
    lock = SoftFileLock(tmp_lock_path)
    proxy = lock.acquire()
    assert isinstance(proxy, AcquireReturnProxy)
    assert lock.is_locked
    assert Path(tmp_lock_path).exists()
    lock.release()
    assert not Path(tmp_lock_path).exists()
    assert lock.is_locked is False


def test_softfilelock_reentrant_acquire_and_release(tmp_lock_path):
    lock = SoftFileLock(tmp_lock_path)
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 2
    lock.release()
    assert lock.lock_counter == 1
    assert Path(tmp_lock_path).exists()
    lock.release()
    assert lock.lock_counter == 0
    assert not Path(tmp_lock_path).exists()


def test_softfilelock_context_manager(tmp_lock_path):
    with SoftFileLock(tmp_lock_path) as lock:
        assert lock.is_locked
        assert Path(tmp_lock_path).exists()
    assert not Path(tmp_lock_path).exists()


def test_softfilelock_release_force(tmp_lock_path):
    lock = SoftFileLock(tmp_lock_path)
    lock.acquire()
    lock.acquire()
    lock.release(force=True)
    assert lock.lock_counter == 0
    assert not Path(tmp_lock_path).exists()


def test_softfilelock_acquire_nonblocking_when_exists(tmp_lock_path):
    tmp_lock_path.write_text("held")
    lock = SoftFileLock(tmp_lock_path)
    with pytest.raises(Timeout):
        lock.acquire(blocking=False)
    assert lock.lock_counter == 0


def test_softfilelock_acquire_timeout_when_exists(monkeypatch, tmp_lock_path):
    tmp_lock_path.write_text("held")
    lock = SoftFileLock(tmp_lock_path)
    monkeypatch.setattr("filelock._api.time.perf_counter", lambda: 0.0)
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        raise Timeout(str(tmp_lock_path))

    monkeypatch.setattr("filelock._api.time.sleep", fake_sleep)
    with pytest.raises(Timeout):
        lock.acquire(timeout=0)
    assert calls["n"] == 1


def test_softfilelock_is_locked_reflects_fd(tmp_lock_path):
    lock = SoftFileLock(tmp_lock_path)
    assert not lock.is_locked
    lock._context.lock_file_fd = 3
    assert lock.is_locked


def test_async_lock_properties(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, timeout=3.0, blocking=False, run_in_executor=False)
    assert lock.timeout == 3.0
    assert lock.blocking is False
    assert lock.run_in_executor is False
    assert lock.loop is None
    assert lock.is_thread_local() is False


@pytest.mark.asyncio
async def test_async_acquire_and_release(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    proxy = await lock.acquire()
    assert isinstance(proxy, AsyncAcquireReturnProxy)
    assert lock.is_locked
    await lock.release()
    assert lock.release_calls == 1
    assert lock.is_locked is False


@pytest.mark.asyncio
async def test_async_aenter_aexit(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    async with lock as entered:
        assert entered is lock
        assert lock.is_locked
    assert lock.release_calls == 1


@pytest.mark.asyncio
async def test_async_acquire_nonblocking_timeout(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    lock._acquire = lambda: None
    with pytest.raises(Timeout):
        await lock.acquire(blocking=False)
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_async_acquire_timeout(monkeypatch, tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    lock._acquire = lambda: None
    times = iter([0.0, 0.2])
    monkeypatch.setattr("filelock.asyncio.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("filelock.asyncio.asyncio.sleep", lambda _: asyncio.sleep(0))
    with pytest.raises(Timeout):
        await lock.acquire(timeout=0.1, poll_interval=0)
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_async_acquire_polls_until_lock(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    state = {"n": 0}

    def fake_acquire():
        state["n"] += 1
        if state["n"] >= 2:
            lock._context.lock_file_fd = 1

    lock._acquire = fake_acquire
    proxy = await lock.acquire(timeout=1, poll_interval=0)
    assert isinstance(proxy, AsyncAcquireReturnProxy)
    assert state["n"] == 2


@pytest.mark.asyncio
async def test_async_release_noop_when_unlocked(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    await lock.release()
    assert lock.release_calls == 0


@pytest.mark.asyncio
async def test_async_release_force(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    await lock.acquire()
    await lock.acquire()
    await lock.release(force=True)
    assert lock.lock_counter == 0
    assert lock.release_calls == 1


def test_async_enter_is_not_allowed(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    with pytest.raises(NotImplementedError):
        lock.__enter__()


def test_async_thread_local_and_run_in_executor_incompatible(tmp_lock_path):
    with pytest.raises(ValueError):
        DummyAsyncLock(tmp_lock_path, thread_local=True, run_in_executor=True)


def test_async_thread_local_false_allowed(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, thread_local=False, run_in_executor=False)
    assert not lock.is_thread_local()


@pytest.mark.asyncio
async def test_async_run_internal_method_sync_method_inline(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    called = {"x": 0}

    def fn():
        called["x"] += 1

    await lock._run_internal_method(fn)
    assert called["x"] == 1


def test_async_filelock_alias_exists():
    assert filelock.AsyncFileLock is not None
    assert filelock.FileLock is not None


def test_filelock_alias_matches_platform_choice():
    if sys.platform == "win32":
        assert filelock.FileLock in {filelock.WindowsFileLock, filelock.SoftFileLock}
    else:
        assert filelock.FileLock in {filelock.UnixFileLock, filelock.SoftFileLock}


def test_async_softfilelock_inheritance(tmp_lock_path):
    lock = AsyncSoftFileLock(tmp_lock_path)
    assert isinstance(lock, BaseAsyncFileLock)
    assert isinstance(lock, SoftFileLock)


def test_lock_file_pathlike_support(tmp_path):
    p = tmp_path / "lockfile"
    lock = DummyLock(p)
    assert lock.lock_file == str(p)


def test_lock_counter_never_negative_after_failed_acquire(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock._acquire = lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    with pytest.raises(RuntimeError):
        lock.acquire()
    assert lock.lock_counter == 0


def test_release_reduces_counter_only_when_locked(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock.release()
    assert lock.lock_counter == 0
    lock._context.lock_file_fd = 1
    lock._context.lock_counter = 1
    lock.release()
    assert lock.release_calls == 1


def test_del_on_unlocked_object_no_error(tmp_lock_path):
    lock = DummyLock(tmp_lock_path)
    lock.__del__()
    assert lock.release_calls == 0


@pytest.mark.asyncio
async def test_async_del_on_unlocked_object_no_error(tmp_lock_path):
    lock = DummyAsyncLock(tmp_lock_path, run_in_executor=False)
    lock.__del__()
    await asyncio.sleep(0)


def test_module_exports_contain_expected_symbols():
    expected = {
        "FileLock",
        "AsyncFileLock",
        "SoftFileLock",
        "UnixFileLock",
        "WindowsFileLock",
        "Timeout",
    }
    assert expected.issubset(set(filelock.__all__))
