"""Tests for AsyncFileLock and AsyncSoftFileLock."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from filelock import (
    AsyncAcquireReturnProxy,
    AsyncFileLock,
    AsyncSoftFileLock,
    BaseAsyncFileLock,
    FileLock,
    SoftFileLock,
    Timeout,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=[AsyncFileLock, AsyncSoftFileLock], ids=["AsyncFileLock", "AsyncSoftFileLock"])
def lock_cls(request):
    return request.param


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "async.lock")


async def test_acquire_and_release(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert not lock.is_locked
    proxy = await lock.acquire()
    assert isinstance(proxy, AsyncAcquireReturnProxy)
    assert proxy.lock is lock
    assert lock.is_locked
    await lock.release()
    assert not lock.is_locked


async def test_async_with_acquires_and_releases(lock_cls, lock_path):
    async with lock_cls(lock_path) as lock:
        assert lock.is_locked
        assert lock.lock_counter == 1
    assert not lock.is_locked
    assert lock.lock_counter == 0


async def test_acquire_proxy_as_async_context_manager(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    async with await lock.acquire() as held:
        assert held is lock
        assert lock.is_locked
    assert not lock.is_locked


async def test_async_with_releases_on_exception(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.raises(RuntimeError):
        async with lock:
            raise RuntimeError("boom")
    assert not lock.is_locked


async def test_reentrant_locking_and_counter(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    await lock.acquire()
    await lock.acquire()
    assert lock.lock_counter == 2
    await lock.release()
    assert lock.is_locked
    assert lock.lock_counter == 1
    await lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


async def test_release_unlocked_and_force_release(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    await lock.release()
    assert not lock.is_locked
    await lock.acquire()
    await lock.acquire()
    await lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


async def test_timeout_when_another_lock_holds_the_file(lock_cls, lock_path):
    holder = lock_cls(lock_path)
    other = lock_cls(lock_path, timeout=0.1)
    await holder.acquire()
    try:
        with pytest.raises(Timeout) as info:
            await other.acquire(poll_interval=0.01)
        assert info.value.lock_file == lock_path
        assert other.lock_counter == 0
        assert not other.is_locked
        with pytest.raises(Timeout):
            await other.acquire(timeout=0.05, poll_interval=0.01)
    finally:
        await holder.release()


async def test_non_blocking_acquire_raises_immediately(lock_cls, lock_path):
    holder = lock_cls(lock_path)
    other = lock_cls(lock_path, blocking=False)
    await holder.acquire()
    try:
        assert other.blocking is False
        with pytest.raises(Timeout):
            await other.acquire()
        with pytest.raises(Timeout):
            await lock_cls(lock_path).acquire(blocking=False)
    finally:
        await holder.release()
    await other.acquire()
    assert other.is_locked
    await other.release()


async def test_waiter_acquires_after_holder_releases(lock_cls, lock_path):
    holder = lock_cls(lock_path)
    waiter = lock_cls(lock_path)
    await holder.acquire()

    async def release_later():
        await asyncio.sleep(0.15)
        await holder.release()

    task = asyncio.create_task(release_later())
    await waiter.acquire(timeout=10, poll_interval=0.01)
    assert waiter.is_locked
    await task
    assert not holder.is_locked
    await waiter.release()


async def test_sync_with_statement_is_rejected(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.raises(NotImplementedError):
        with lock:
            pass
    assert not lock.is_locked


async def test_default_properties(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert isinstance(lock, BaseAsyncFileLock)
    assert lock.lock_file == lock_path
    assert lock.timeout == -1
    assert lock.run_in_executor is True
    assert lock.executor is None
    assert lock.loop is None
    assert lock.is_thread_local() is False


async def test_thread_local_requires_running_without_executor(lock_cls, lock_path):
    with pytest.raises(ValueError, match="run_in_executor"):
        lock_cls(lock_path, thread_local=True)
    lock = lock_cls(lock_path, thread_local=True, run_in_executor=False)
    assert lock.is_thread_local() is True
    await lock.acquire()
    assert lock.is_locked
    await lock.release()
    assert not lock.is_locked


async def test_lock_without_executor(lock_cls, lock_path):
    lock = lock_cls(lock_path, run_in_executor=False)
    assert lock.run_in_executor is False
    async with lock:
        assert lock.is_locked
    assert not lock.is_locked


async def test_custom_executor_is_used(lock_cls, lock_path):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        lock = lock_cls(lock_path, executor=executor)
        assert lock.executor is executor
        async with lock:
            assert lock.is_locked
        assert not lock.is_locked
    finally:
        executor.shutdown()


async def test_concurrent_tasks_are_mutually_exclusive(lock_cls, lock_path):
    state = {"active": 0, "max": 0, "entries": 0}

    async def worker():
        lock = lock_cls(lock_path)
        for _ in range(3):
            async with await lock.acquire(timeout=30, poll_interval=0.005):
                state["active"] += 1
                state["entries"] += 1
                state["max"] = max(state["max"], state["active"])
                await asyncio.sleep(0.002)
                state["active"] -= 1

    await asyncio.gather(*(worker() for _ in range(5)))
    assert state["max"] == 1
    assert state["entries"] == 15
    assert state["active"] == 0


@pytest.mark.parametrize(
    "async_cls,sync_cls",
    [(AsyncFileLock, FileLock), (AsyncSoftFileLock, SoftFileLock)],
    ids=["hard", "soft"],
)
async def test_async_lock_excludes_a_sync_lock(async_cls, sync_cls, lock_path):
    async_lock = async_cls(lock_path)
    await async_lock.acquire()
    sync_lock = sync_cls(lock_path)
    with pytest.raises(Timeout):
        sync_lock.acquire(timeout=0.1, poll_interval=0.01)
    await async_lock.release()
    sync_lock.acquire(timeout=1, poll_interval=0.01)
    assert sync_lock.is_locked
    sync_lock.release()


async def test_async_soft_lock_file_lifecycle(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    assert not os.path.exists(lock_path)
    async with lock:
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


async def test_async_singleton_locks(lock_cls, tmp_path):
    path = str(tmp_path / "single.lock")
    a = lock_cls(path, is_singleton=True)
    b = lock_cls(path, is_singleton=True)
    assert a is b
    await a.acquire()
    await b.acquire()
    assert a.lock_counter == 2
    await b.release()
    assert a.is_locked
    await a.release()
    assert not a.is_locked
