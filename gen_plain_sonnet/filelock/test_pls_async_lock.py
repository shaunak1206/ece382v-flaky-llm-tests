from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from filelock import (
    AsyncFileLock,
    AsyncSoftFileLock,
    BaseAsyncFileLock,
    BaseFileLock,
    Timeout,
)


@pytest.fixture
def lock_path(tmp_path: Path) -> str:
    return str(tmp_path / "async.lock")


def test_async_lock_class_hierarchy():
    assert issubclass(AsyncFileLock, BaseAsyncFileLock)
    assert issubclass(AsyncFileLock, BaseFileLock)
    assert issubclass(AsyncSoftFileLock, BaseAsyncFileLock)


def test_async_defaults(lock_path):
    lock = AsyncFileLock(lock_path)
    assert lock.is_thread_local() is False
    assert lock.run_in_executor is True
    assert lock.loop is None
    assert lock.executor is None
    assert lock.timeout == -1


def test_thread_local_with_run_in_executor_is_rejected(lock_path):
    with pytest.raises(ValueError, match="run_in_executor"):
        AsyncFileLock(lock_path, thread_local=True, run_in_executor=True)


def test_thread_local_without_executor_is_allowed(lock_path):
    lock = AsyncFileLock(lock_path, thread_local=True, run_in_executor=False)
    assert lock.is_thread_local() is True
    assert lock.run_in_executor is False


def test_sync_with_statement_is_rejected(lock_path):
    lock = AsyncFileLock(lock_path)
    with pytest.raises(NotImplementedError, match="async with"):
        with lock:
            pass
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_acquire_and_release(lock_path):
    lock = AsyncFileLock(lock_path)
    await lock.acquire()
    assert lock.is_locked
    assert lock.lock_counter == 1
    await lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_async_with_locks_and_unlocks(lock_path):
    lock = AsyncFileLock(lock_path)
    async with lock as entered:
        assert entered is lock
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_async_with_releases_on_exception(lock_path):
    lock = AsyncFileLock(lock_path)
    with pytest.raises(RuntimeError, match="boom"):
        async with lock:
            raise RuntimeError("boom")
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_second_lock_times_out(lock_path):
    first = AsyncFileLock(lock_path)
    second = AsyncFileLock(lock_path)
    async with first:
        with pytest.raises(Timeout) as excinfo:
            await second.acquire(timeout=0)
        assert excinfo.value.lock_file == lock_path
    assert second.lock_counter == 0
    assert not second.is_locked


@pytest.mark.asyncio
async def test_nested_acquire_counts_and_force_release(lock_path):
    lock = AsyncFileLock(lock_path)
    await lock.acquire()
    await lock.acquire()
    assert lock.lock_counter == 2
    await lock.release()
    assert lock.is_locked
    assert lock.lock_counter == 1
    await lock.acquire()
    await lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.asyncio
async def test_acquire_returns_proxy_usable_with_async_with(lock_path):
    lock = AsyncFileLock(lock_path)
    proxy = await lock.acquire()
    async with proxy as entered:
        assert entered is lock
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_run_in_executor_false_works(lock_path):
    lock = AsyncFileLock(lock_path, run_in_executor=False)
    async with lock:
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.asyncio
async def test_custom_executor_is_used_and_exposed(lock_path):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        lock = AsyncFileLock(lock_path, executor=executor)
        assert lock.executor is executor
        async with lock:
            assert lock.is_locked
        assert not lock.is_locked
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_async_soft_lock_creates_and_removes_file(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    async with lock:
        assert Path(lock_path).exists()
    assert not Path(lock_path).exists()


@pytest.mark.asyncio
async def test_async_singleton_returns_same_instance(lock_path):
    first = AsyncFileLock(lock_path, is_singleton=True)
    second = AsyncFileLock(lock_path, is_singleton=True)
    assert first is second
    assert first.is_singleton


@pytest.mark.asyncio
async def test_waiting_task_acquires_after_release(lock_path):
    first = AsyncFileLock(lock_path)
    second = AsyncFileLock(lock_path)
    await first.acquire()
    task = asyncio.ensure_future(second.acquire(timeout=30, poll_interval=0.01))
    await asyncio.sleep(0.1)
    assert not task.done()
    await first.release()
    proxy = await asyncio.wait_for(task, timeout=30)
    assert second.is_locked
    async with proxy:
        pass
    assert not second.is_locked
