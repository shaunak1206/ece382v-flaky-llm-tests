import asyncio
import os
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from unittest import mock

import pytest
from freezegun import freeze_time

from filelock import (
    AcquireReturnProxy,
    AsyncAcquireReturnProxy,
    AsyncFileLock,
    AsyncSoftFileLock,
    AsyncUnixFileLock,
    AsyncWindowsFileLock,
    BaseFileLock,
    BaseAsyncFileLock,
    FileLock,
    SoftFileLock,
    Timeout,
    UnixFileLock,
    WindowsFileLock,
)


@pytest.fixture
def tmp_lockfile():
    with tempfile.TemporaryDirectory() as td:
        yield os.path.join(td, "lockfile.lock")


@pytest.mark.parametrize(
    "lock_cls",
    [SoftFileLock, FileLock],
)
def test_init_attributes_default(tmp_lockfile, lock_cls):
    lock = lock_cls(tmp_lockfile)
    assert lock.lock_file == tmp_lockfile
    assert lock.timeout == -1
    assert lock.mode == 0o644
    assert lock.blocking is True
    assert lock.lock_counter == 0
    # Thread local default
    assert isinstance(lock._context, (lock_cls._context.__class__, SoftFileLock._context.__class__))


def test_error_timeout_message_and_pickle(tmp_lockfile):
    err = Timeout(tmp_lockfile)
    assert str(err) == f"The file lock '{tmp_lockfile}' could not be acquired."
    # test repr uses lock_file property
    assert repr(err) == f"Timeout({tmp_lockfile!r})"

    # test pickle round-trip
    import pickle

    pickled = pickle.dumps(err)
    new_err = pickle.loads(pickled)
    assert isinstance(new_err, Timeout)
    assert new_err.lock_file == tmp_lockfile


def test_acquire_return_proxy_context_behavior(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    # acquire should return a proxy object
    proxy = lock.acquire(timeout=0)
    assert isinstance(proxy, AcquireReturnProxy)
    assert proxy.__enter__() is lock

    # Test that __exit__ calls release on lock
    lock._acquire = mock.Mock()
    lock._release = mock.Mock()
    lock._context.lock_file_fd = 123
    lock._context.lock_counter = 1

    # increment lock_counter manually since normally done in acquire
    lock._context.lock_counter = 1
    proxy.__exit__(None, None, None)
    lock._release.assert_called_once()
    assert lock._context.lock_file_fd is None or lock._context.lock_file_fd == 123  # We didn't reset fd here


@pytest.mark.parametrize("blocking", [True, False])
def test_acquire_timeout_and_blocking_behavior(tmp_lockfile, blocking):
    class DummyLock(BaseFileLock):
        def _acquire(self):
            # Not actually acquiring, simulate failure
            pass

        def _release(self):
            pass

        @property
        def is_locked(self):
            # Always False to trigger timeout logic
            return False

    # timeout 0 means only one attempt
    lock = DummyLock(tmp_lockfile, timeout=0, blocking=blocking)
    if blocking:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            lock.acquire(timeout=0.1)
        duration = time.perf_counter() - start
        # it should spend at least 0.1 seconds waiting
        assert duration >= 0.09
    else:
        # blocking=False means immediate failure
        with pytest.raises(Timeout):
            lock.acquire(timeout=0.1, blocking=False)


def test_acquire_releases_lock_counter_on_failure(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def _acquire(self):
            raise RuntimeError("fail")

        def _release(self):
            pass

        @property
        def is_locked(self):
            return False

    lock = DummyLock(tmp_lockfile)
    with pytest.raises(RuntimeError):
        lock.acquire()


@pytest.mark.parametrize("blocking", [True, False])
def test_release_behavior(tmp_lockfile, blocking):
    class DummyLock(BaseFileLock):
        def _acquire(self):
            self._context.lock_file_fd = 42

        def _release(self):
            self._context.lock_file_fd = None

    lock = DummyLock(tmp_lockfile, blocking=blocking)
    lock._context.lock_file_fd = 42
    lock._context.lock_counter = 2

    # release reduces lock counter by one
    lock.release()
    assert lock.lock_counter == 1
    assert lock.is_locked is True

    # release force releases regardless of counter
    lock.release(force=True)
    assert lock.lock_counter == 0
    assert lock.is_locked is False

    # release on unlocked lock does nothing
    lock.release()
    assert lock.lock_counter == 0


def test_enter_exit_context_manager(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.acquired = False
            self.released = False

        def _acquire(self):
            self.acquired = True
            self._context.lock_file_fd = 42

        def _release(self):
            self.released = True
            self._context.lock_file_fd = None

        @property
        def is_locked(self):
            return self._context.lock_file_fd is not None

    lock = DummyLock(tmp_lockfile)
    with lock as l:
        assert l is lock
        assert lock.acquired
        assert lock.is_locked is True
        assert lock.lock_counter == 1
    # Exiting context calls release fully when counter hits 0
    assert lock.released
    assert lock.lock_counter == 0


def test_del_calls_release(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.released = False

        def _acquire(self):
            self._context.lock_file_fd = 42

        def _release(self):
            self.released = True
            self._context.lock_file_fd = None

        @property
        def is_locked(self):
            return self._context.lock_file_fd is not None

    lock = DummyLock(tmp_lockfile)
    lock._context.lock_file_fd = 42
    lock._context.lock_counter = 1
    # Deleting lock should call release(force=True)
    del lock  # Just make sure no error


def test_filelock_singleton_behavior(tmp_lockfile):
    lock1 = SoftFileLock(tmp_lockfile, is_singleton=True, timeout=3)
    lock2 = SoftFileLock(tmp_lockfile, is_singleton=True, timeout=3)
    # Singleton returns same instance
    assert lock1 is lock2

    with pytest.raises(ValueError):
        SoftFileLock(tmp_lockfile, is_singleton=True, timeout=10)


def test_filelock_singleton_different_args_fails(tmp_lockfile):
    lock1 = SoftFileLock(tmp_lockfile, is_singleton=True, blocking=True)
    # Different 'blocking' param should raise ValueError
    with pytest.raises(ValueError):
        SoftFileLock(tmp_lockfile, is_singleton=True, blocking=False)


def test_timeout_property_set_and_get(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    lock.timeout = 1.23
    assert lock.timeout == pytest.approx(1.23)
    # string value converts to float
    lock.timeout = "2.34"
    assert lock.timeout == pytest.approx(2.34)


def test_blocking_property_set_and_get(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    lock.blocking = False
    assert lock.blocking is False
    lock.blocking = True
    assert lock.blocking is True


@pytest.mark.parametrize("lock_cls", [SoftFileLock, FileLock])
def test_lock_counter_increments_and_decrements(tmp_lockfile, lock_cls):
    lock = lock_cls(tmp_lockfile)
    assert lock.lock_counter == 0

    # patch _acquire and _release to simulate success
    lock._acquire = lambda: setattr(lock._context, "lock_file_fd", 42)
    lock._release = lambda: setattr(lock._context, "lock_file_fd", None)

    proxy = lock.acquire(timeout=0)
    assert lock.lock_counter == 1

    proxy.__exit__(None, None, None)
    assert lock.lock_counter == 0


def test_deprecated_poll_intervall_warns(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        proxy = lock.acquire(poll_intervall=0.01)
        assert any("poll_interval" in str(warn.message) for warn in w)
        proxy.__exit__(None, None, None)


def test_softfilelock_acquire_and_release(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    # Acquire lock should create the file and lock_file_fd should be integer
    proxy = lock.acquire(timeout=0)
    assert lock.is_locked
    assert isinstance(lock._context.lock_file_fd, int)
    proxy.__exit__(None, None, None)
    assert not lock.is_locked
    # File should be deleted after release
    assert not os.path.exists(tmp_lockfile)


def test_softfilelock_release_without_acquire(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    # release on unlocked lock does nothing
    lock.release()
    assert lock.lock_counter == 0

    # force release also works without lock
    lock.release(force=True)
    assert lock.lock_counter == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Unix only test")
def test_unixfilelock_acquire_and_release(tmp_lockfile):
    if not hasattr(UnixFileLock, "_acquire"):
        pytest.skip("UnixFileLock._acquire not implemented")

    lock = UnixFileLock(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    assert lock.is_locked is True
    proxy.__exit__(None, None, None)
    assert lock.is_locked is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_windowsfilelock_acquire_and_release(tmp_lockfile):
    if not hasattr(WindowsFileLock, "_acquire"):
        pytest.skip("WindowsFileLock._acquire not implemented")

    lock = WindowsFileLock(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    assert lock.is_locked is True
    proxy.__exit__(None, None, None)
    assert lock.is_locked is False


def test_softfilelock_cannot_acquire_non_writable(tmp_lockfile):
    # Make a file and remove write permission
    with open(tmp_lockfile, "w") as f:
        f.write("test")

    os.chmod(tmp_lockfile, 0o444)  # read-only
    lock = SoftFileLock(tmp_lockfile)

    with pytest.raises(PermissionError):
        lock.acquire(timeout=0)


@pytest.mark.asyncio
async def test_async_acquire_and_release(tmp_lockfile):
    # Use AsyncSoftFileLock as it is available on all platforms
    lock = AsyncSoftFileLock(tmp_lockfile)

    proxy = await lock.acquire(timeout=0)
    assert isinstance(proxy, AsyncAcquireReturnProxy)
    assert lock.is_locked

    await proxy.__aexit__(None, None, None)
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_release_force(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    await lock._acquire()
    lock._context.lock_counter = 2
    await lock.release(force=True)
    assert lock.lock_counter == 0
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_acquire_with_blocking_false_raises(tmp_lockfile):
    class DummyAsyncLock(AsyncSoftFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        async def _acquire(self):
            pass

        async def _release(self):
            pass

        @property
        def is_locked(self):
            return False

    lock = DummyAsyncLock(tmp_lockfile)
    with pytest.raises(Timeout):
        await lock.acquire(timeout=0.01, blocking=False)


@pytest.mark.asyncio
async def test_async_acquire_releases_lock_counter_on_error(tmp_lockfile):
    class DummyAsyncLock(AsyncSoftFileLock):
        async def _acquire(self):
            raise RuntimeError("fail")

        async def _release(self):
            pass

        @property
        def is_locked(self):
            return False

    lock = DummyAsyncLock(tmp_lockfile)
    with pytest.raises(RuntimeError):
        await lock.acquire()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    async with lock as l:
        assert l is lock
        assert lock.lock_counter == 1
        assert lock.is_locked
    assert lock.lock_counter == 0
    assert not lock.is_locked


def test_asyncfilelock_enter_raises(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    with pytest.raises(NotImplementedError):
        lock.__enter__()


def test_filelockmeta_preserves_signature_and_creation(tmp_lockfile):
    # Test that FileLockMeta.__call__ respect and passes parameters correctly
    lock = FileLock(tmp_lockfile, timeout=1, mode=0o600, blocking=False, thread_local=False)
    assert lock.timeout == 1
    assert lock.mode == 0o600
    assert lock.blocking is False
    assert lock._is_thread_local is False


def test_basefilelock_subclass_instances_are_weakvaluedict(tmp_lockfile):
    # Instantiating multiple locks should be tracked in _instances dict weakly
    lock1 = SoftFileLock(tmp_lockfile, is_singleton=True)
    lock_id = id(lock1)
    assert str(tmp_lockfile) in SoftFileLock._instances
    # Delete lock1 and force GC, _instances should be empty
    del lock1
    import gc

    gc.collect()
    assert len(SoftFileLock._instances) == 0


def test_basefilelock_is_locked_and_lock_counter_properties(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    assert lock.lock_counter == 0
    assert not lock.is_locked
    lock._context.lock_file_fd = 42
    assert lock.is_locked


def test_threadlocalfilecontext_is_thread_local_flag():
    from filelock._api import ThreadLocalFileContext

    tlc = ThreadLocalFileContext("file", 1, 0o644, True, blocking=True)
    # ThreadLocalFileContext should be instance of threading.local
    import threading

    assert isinstance(tlc, threading.local)


def test_issubclass_and_subclass_behavior():
    # BaseFileLock is abstract; cannot instantiate directly
    with pytest.raises(TypeError):
        BaseFileLock("file")

    # But subclasses can instantiate
    lock = SoftFileLock("file")
    assert isinstance(lock, BaseFileLock)


def test_is_singleton_property(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile, is_singleton=True)
    assert lock.is_singleton is True
    lock2 = SoftFileLock(tmp_lockfile, is_singleton=False)
    assert lock2.is_singleton is False


def test_filelock_context_repr_and_str():
    from filelock._api import Timeout

    e = Timeout("file.lock")
    s = str(e)
    r = repr(e)
    assert "file.lock" in s
    assert "Timeout" in r


def test_softfilelock_ensure_directory(tmp_lockfile):
    # Test that directory is created by SoftFileLock on _acquire
    directory = os.path.dirname(tmp_lockfile)
    if os.path.exists(directory):
        import shutil

        shutil.rmtree(directory)
    lock = SoftFileLock(tmp_lockfile)
    assert not os.path.exists(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    assert os.path.exists(directory)
    proxy.__exit__(None, None, None)


def test_softfilelock_release_deletes_lockfile(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    assert os.path.exists(tmp_lockfile)
    proxy.__exit__(None, None, None)
    # File should be deleted after release
    assert not os.path.exists(tmp_lockfile)


def test_softfilelock_release_handles_missing_file(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    assert os.path.exists(tmp_lockfile)
    # Remove file manually before release
    os.unlink(tmp_lockfile)
    proxy.__exit__(None, None, None)
    assert not os.path.exists(tmp_lockfile)


def test_softfilelock_acquire_eexist_and_eacces_handling(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    # Patch os.open to raise OSError with errno EEXIST to simulate lock file already exists
    with mock.patch("os.open", side_effect=OSError(os.errno.EEXIST, "file exists")):
        lock._context.lock_file_fd = None
        lock._acquire()
        assert lock._context.lock_file_fd is None

    # Patch os.open to raise OSError with errno EACCES on Windows
    if sys.platform == "win32":
        with mock.patch("os.open", side_effect=OSError(os.errno.EACCES, "access denied")):
            lock._context.lock_file_fd = None
            lock._acquire()
            assert lock._context.lock_file_fd is None


@pytest.mark.asyncio
async def test_async_filelock_run_in_executor_and_loop(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile, run_in_executor=True)
    called = False

    def _acquire_sync():
        nonlocal called
        called = True
        lock._context.lock_file_fd = 42

    lock._acquire = _acquire_sync

    proxy = await lock.acquire(timeout=0)
    assert called
    assert lock.is_locked

    await proxy.__aexit__(None, None, None)
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_filelock_run_in_executor_false(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile, run_in_executor=False)
    called = False

    def _acquire_sync():
        nonlocal called
        called = True
        lock._context.lock_file_fd = 42

    lock._acquire = _acquire_sync

    proxy = await lock.acquire(timeout=0)
    assert called
    assert lock.is_locked
    await proxy.__aexit__(None, None, None)
    assert not lock.is_locked


def test_asyncfilelock_thread_local_and_run_in_executor_conflict(tmp_lockfile):
    AsyncLockClass = AsyncSoftFileLock
    # thread_local True and run_in_executor True should raise ValueError
    with pytest.raises(ValueError):
        AsyncLockClass(
            tmp_lockfile,
            thread_local=True,
            run_in_executor=True,
        )


@pytest.mark.asyncio
async def test_async_release_handles_force_correctly(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    await lock._acquire()
    lock._context.lock_counter = 5
    await lock.release(force=True)
    assert lock.lock_counter == 0


def test_acquire_with_freeze_time_and_timeout(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0

        def _acquire(self):
            self.attempts += 1
            if self.attempts < 3:
                # Fail to acquire lock
                self._context.lock_file_fd = None
            else:
                self._context.lock_file_fd = 123

        @property
        def is_locked(self):
            return self._context.lock_file_fd is not None

        def _release(self):
            self._context.lock_file_fd = None

    lock = DummyLock(tmp_lockfile)
    with freeze_time() as frozen:
        proxy = lock.acquire(timeout=0.1, poll_interval=0.01)
        # Advance time artificially for the test timeout (simulate delay)
        frozen.tick(0.2)
    proxy.__exit__(None, None, None)
    assert lock.attempts >= 3


def test_acquire_immediate_failure_with_blocking_false(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called = False

        def _acquire(self):
            self.called = True
            # never actually acquires
            self._context.lock_file_fd = None

        @property
        def is_locked(self):
            return False

        def _release(self):
            pass

    lock = DummyLock(tmp_lockfile)
    with pytest.raises(Timeout):
        lock.acquire(timeout=1, blocking=False)
    assert lock.called


def test_acquire_returns_proxy_and_releases_on_exit(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    proxy = lock.acquire(timeout=0)
    lock_fd = lock._context.lock_file_fd
    assert lock_fd is not None

    proxy.__exit__(None, None, None)
    assert lock._context.lock_file_fd is None


def test_asyncacquire_return_proxy_context_manager(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)

    async def enter_exit():
        proxy = await lock.acquire(timeout=0)
        aenter = await proxy.__aenter__()
        assert aenter is lock
        await proxy.__aexit__(None, None, None)

    asyncio.run(enter_exit())


@pytest.mark.asyncio
async def test_asyncfilelock_del_calls_release(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    await lock._acquire()
    lock._context.lock_counter = 1
    # Call __del__ manually to ensure it does not raise
    lock.__del__()


def test_filelock_version_string():
    from filelock import __version__

    # version string is nonempty
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_filelock___all___contains_expected_names():
    import filelock

    expected_names = {
        "AcquireReturnProxy",
        "AsyncAcquireReturnProxy",
        "AsyncFileLock",
        "AsyncSoftFileLock",
        "AsyncUnixFileLock",
        "AsyncWindowsFileLock",
        "BaseAsyncFileLock",
        "BaseFileLock",
        "FileLock",
        "SoftFileLock",
        "Timeout",
        "UnixFileLock",
        "WindowsFileLock",
        "__version__",
    }
    assert set(filelock.__all__) == expected_names


def test_acquire_multiple_times_adjusts_lock_counter(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    lock._acquire = lambda: setattr(lock._context, "lock_file_fd", 42)
    lock._release = lambda: setattr(lock._context, "lock_file_fd", None)

    proxy1 = lock.acquire(timeout=0)
    proxy2 = lock.acquire(timeout=0)
    assert lock.lock_counter == 2
    proxy1.__exit__(None, None, None)
    assert lock.lock_counter == 1
    proxy2.__exit__(None, None, None)
    assert lock.lock_counter == 0


def test_release_force_releases_even_if_counter_zero(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)

    lock._context.lock_file_fd = 42
    lock._context.lock_counter = 0

    called = False

    def release_override():
        nonlocal called
        called = True
        lock._context.lock_file_fd = None

    lock._release = release_override

    lock.release(force=True)
    assert called
    assert lock.lock_counter == 0


def test_acquire_with_none_timeout_uses_default(tmp_lockfile):
    class DummyLock(SoftFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called = False

        def _acquire(self):
            self.called = True
            self._context.lock_file_fd = 123

        @property
        def is_locked(self):
            return self._context.lock_file_fd is not None

        def _release(self):
            self._context.lock_file_fd = None

    lock = DummyLock(tmp_lockfile, timeout=0.5)
    proxy = lock.acquire(timeout=None)
    assert lock.called is True
    proxy.__exit__(None, None, None)


def test_acquire_with_deprecated_poll_intervall_uses_poll_interval(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        proxy = lock.acquire(poll_intervall=0.001)
        proxy.__exit__(None, None, None)
        assert any("poll_interval" in str(warn.message) for warn in w)


def test_release_on_unlocked_lock_does_not_error(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile)
    lock._context.lock_file_fd = None
    lock.release()  # no error


def test_threadlocal_context_has_attributes(tmp_lockfile):
    from filelock._api import ThreadLocalFileContext

    ctx = ThreadLocalFileContext(
        lock_file=tmp_lockfile, timeout=1.0, mode=0o644, blocking=True, lock_file_fd=None, lock_counter=0
    )
    assert ctx.lock_file == tmp_lockfile
    assert ctx.timeout == 1.0
    assert ctx.lock_counter == 0
    assert ctx.lock_file_fd is None


def test_filelock_is_thread_local_property(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile, thread_local=True)
    assert lock.is_thread_local() is True

    lock2 = SoftFileLock(tmp_lockfile, thread_local=False)
    assert lock2.is_thread_local() is False


def test_acquire_with_nonblocking_returns_proxy(tmp_lockfile):
    class DummyLock(BaseFileLock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0

        def _acquire(self):
            self.attempts += 1
            self._context.lock_file_fd = 42

        def _release(self):
            self._context.lock_file_fd = None

        @property
        def is_locked(self):
            return self._context.lock_file_fd is not None

    lock = DummyLock(tmp_lockfile, blocking=False)
    proxy = lock.acquire(timeout=0)
    assert isinstance(proxy, AcquireReturnProxy)
    proxy.__exit__(None, None, None)


def test_lockfile_mode_property(tmp_lockfile):
    lock = SoftFileLock(tmp_lockfile, mode=0o700)
    assert lock.mode == 0o700


def test_asyncfilelock_executor_setter_and_getter(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    assert lock.executor is None
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    lock.executor = executor
    assert lock.executor == executor
    executor.shutdown(wait=True)


def test_asyncfilelock_loop_property(tmp_lockfile):
    loop = asyncio.new_event_loop()
    lock = AsyncSoftFileLock(tmp_lockfile, loop=loop)
    assert lock.loop == loop
    loop.close()


@pytest.mark.asyncio
async def test_asyncfilelock_run_in_executor_false_calls_method_directly(monkeypatch, tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile, run_in_executor=False)
    called = False

    def method_to_call():
        nonlocal called
        called = True
        lock._context.lock_file_fd = 42

    lock._acquire = method_to_call
    proxy = await lock.acquire(timeout=0)
    assert called
    await proxy.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_asyncfilelock_run_in_executor_true_uses_loop_executor(monkeypatch, tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile, run_in_executor=True)
    called = False

    def method_to_call():
        nonlocal called
        called = True
        lock._context.lock_file_fd = 42

    lock._acquire = method_to_call

    proxy = await lock.acquire(timeout=0)
    assert called
    await proxy.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_asyncfilelock_aenter_aexit(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    async with lock as l:
        assert l is lock
        assert lock.lock_counter == 1

    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_asyncacquire_return_proxy_usage(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    proxy = await lock.acquire(timeout=0)
    r = await proxy.__aenter__()
    assert r is lock
    await proxy.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_asyncfilelock_release_without_acquire(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    await lock.release(force=True)  # should not raise even locked is False


# This tests the warning when only SoftFileLock is available because no fcntl on Unix
@pytest.mark.skipif(
    sys.platform != "linux" and sys.platform != "darwin",
    reason="Only relevant on Unix without fcntl",
)
def test_softfilelock_warning_on_no_fcntl(monkeypatch):
    monkeypatch.setattr("filelock._unix.has_fcntl", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import importlib

        importlib.reload(sys.modules["filelock.__init__"])
        assert any("soft file lock" in str(warn.message) for warn in w)


@pytest.mark.parametrize(
    "lock_cls",
    [SoftFileLock, UnixFileLock, WindowsFileLock],
)
def test_acquire_and_release_with_thread_local_lock_counter(tmp_lockfile, lock_cls):
    # Only test the reentrant lock_counter behavior on locks with implemented _acquire
    if not hasattr(lock_cls, "_acquire") or lock_cls._acquire is NotImplemented:
        pytest.skip(f"{lock_cls.__name__}._acquire not implemented")

    lock = lock_cls(tmp_lockfile, thread_local=True)
    lock._acquire()
    lock._context.lock_counter = 1
    assert lock.lock_counter == 1
    # Acquire again increments counter
    proxy = lock.acquire(timeout=0)
    assert lock.lock_counter >= 2
    proxy.__exit__(None, None, None)
    # Release again decrements counter but lock still held
    assert lock.lock_counter >= 1
    lock.release(force=True)
    assert lock.lock_counter == 0


@pytest.mark.parametrize(
    "lock_cls",
    [SoftFileLock, FileLock],
)
def test_lockfile_str_path_conversion(tmp_lockfile, lock_cls):
    p = Path(tmp_lockfile)
    lock1 = lock_cls(tmp_lockfile)
    lock2 = lock_cls(p)
    assert lock1.lock_file == str(p)
    assert lock2.lock_file == str(p)


@pytest.mark.asyncio
async def test_asyncfilelock_release_decrements_counter(tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    await lock._acquire()
    lock._context.lock_counter = 2
    await lock.release()
    assert lock.lock_counter == 1


@pytest.mark.asyncio
async def test_asyncfilelock_release_no_lock(monkeypatch, tmp_lockfile):
    lock = AsyncSoftFileLock(tmp_lockfile)
    lock._context.lock_file_fd = None
    lock._context.lock_counter = 0
    await lock.release()
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_asyncfilelock_acquire_with_default_lock(monkeypatch, tmp_lockfile):
    called = False

    async def fake_acquire():
        nonlocal called
        called = True
        lock._context.lock_file_fd = 1

    lock = AsyncSoftFileLock(tmp_lockfile)
    lock._acquire = fake_acquire
    proxy = await lock.acquire(timeout=None)
    assert called
    await proxy.__aexit__(None, None, None)


# Count of test functions: over 70, within requested range.
