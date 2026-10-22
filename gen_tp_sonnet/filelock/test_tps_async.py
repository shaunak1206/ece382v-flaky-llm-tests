import asyncio

import pytest

from filelock import SoftFileLock, Timeout
from filelock.asyncio import (
    AsyncAcquireReturnProxy,
    AsyncSoftFileLock,
    BaseAsyncFileLock,
)


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "async.lock")


def test_async_soft_lock_class_hierarchy():
    assert issubclass(AsyncSoftFileLock, SoftFileLock)
    assert issubclass(AsyncSoftFileLock, BaseAsyncFileLock)


def test_async_lock_defaults(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    assert lock.lock_file == lock_path
    assert lock.timeout == -1
    assert lock.blocking is True
    assert lock.is_locked is False
    assert lock.lock_counter == 0


def test_async_default_thread_local_false(lock_path):
    assert AsyncSoftFileLock(lock_path).is_thread_local() is False


def test_async_default_run_in_executor_true(lock_path):
    assert AsyncSoftFileLock(lock_path).run_in_executor is True


def test_async_run_in_executor_false(lock_path):
    assert AsyncSoftFileLock(lock_path, run_in_executor=False).run_in_executor is False


def test_async_default_loop_and_executor_none(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    assert lock.loop is None
    assert lock.executor is None


def test_async_executor_setter(lock_path):
    from concurrent import futures

    lock = AsyncSoftFileLock(lock_path)
    with futures.ThreadPoolExecutor(max_workers=1) as pool:
        lock.executor = pool
        assert lock.executor is pool
        lock.executor = None
    assert lock.executor is None


@pytest.mark.asyncio
async def test_async_loop_property_reflects_constructor(lock_path):
    loop = asyncio.get_running_loop()
    lock = AsyncSoftFileLock(lock_path, loop=loop)
    assert lock.loop is loop


@pytest.mark.asyncio
async def test_async_with_acquires_and_releases(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    async with lock:
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_with_releases_on_exception(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    with pytest.raises(RuntimeError):
        async with lock:
            raise RuntimeError("boom")
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_acquire_and_release(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    await lock.acquire()
    assert lock.is_locked
    assert lock.lock_counter == 1
    await lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_async_acquire_returns_proxy(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    proxy = await lock.acquire()
    try:
        assert isinstance(proxy, AsyncAcquireReturnProxy)
    finally:
        await lock.release()


@pytest.mark.asyncio
async def test_async_reentrant_counter(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    await lock.acquire()
    await lock.acquire()
    assert lock.lock_counter == 2
    await lock.release()
    assert lock.is_locked
    await lock.release()
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_force_release(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    await lock.acquire()
    await lock.acquire()
    await lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_async_release_without_acquire_is_noop(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    await lock.release()
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_second_lock_times_out_when_held(lock_path):
    first = AsyncSoftFileLock(lock_path)
    second = AsyncSoftFileLock(lock_path)
    async with first:
        with pytest.raises(Timeout) as excinfo:
            await second.acquire(timeout=0)
    assert excinfo.value.lock_file == lock_path
    assert not second.is_locked


@pytest.mark.asyncio
async def test_async_non_blocking_raises_when_held(lock_path):
    first = AsyncSoftFileLock(lock_path)
    second = AsyncSoftFileLock(lock_path)
    async with first:
        with pytest.raises(Timeout):
            await second.acquire(blocking=False)


@pytest.mark.asyncio
async def test_async_waiter_gets_lock_after_release(lock_path):
    first = AsyncSoftFileLock(lock_path)
    second = AsyncSoftFileLock(lock_path)
    await first.acquire()

    async def waiter():
        await second.acquire(timeout=5, poll_interval=0.01)
        return second.is_locked

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0.1)
    assert not task.done()
    await first.release()
    assert await asyncio.wait_for(task, timeout=10) is True
    await second.release()


@pytest.mark.asyncio
async def test_async_without_executor(lock_path):
    lock = AsyncSoftFileLock(lock_path, run_in_executor=False)
    async with lock:
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_proxy_context_manager(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    async with await lock.acquire() as entered:
        assert entered is lock
        assert lock.is_locked
    assert not lock.is_locked


def test_sync_with_on_async_lock_is_rejected(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    with pytest.raises(NotImplementedError):
        with lock:
            pass
    assert not lock.is_locked
