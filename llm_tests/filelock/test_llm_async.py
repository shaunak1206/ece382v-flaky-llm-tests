"""AsyncFileLock and AsyncSoftFileLock."""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from filelock import (
    AsyncAcquireReturnProxy,
    AsyncFileLock,
    AsyncSoftFileLock,
    BaseAsyncFileLock,
    Timeout,
)

ASYNC_CLASSES = [AsyncFileLock, AsyncSoftFileLock]

pytestmark = pytest.mark.asyncio


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "async.lock"


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_acquire_release(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert not lock.is_locked
    await lock.acquire()
    assert lock.is_locked
    assert lock.lock_counter == 1
    await lock.release()
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_async_context_manager(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    async with lock as acquired:
        assert acquired is lock
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_acquire_proxy_as_async_context_manager(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    proxy = await lock.acquire()
    assert isinstance(proxy, AsyncAcquireReturnProxy)
    async with proxy as inner:
        assert inner is lock
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_async_context_manager_releases_on_error(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.raises(KeyError):
        async with lock:
            raise KeyError("x")
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_reentrant_async(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    async with lock:
        async with lock:
            assert lock.lock_counter == 2
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_release_force(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    await lock.acquire()
    await lock.acquire()
    await lock.release(force=True)
    assert not lock.is_locked
    assert lock.lock_counter == 0


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_sync_with_is_not_supported(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    with pytest.raises(NotImplementedError):
        with lock:
            pass


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_timeout_when_held_by_other_lock(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    async with first:
        with pytest.raises(Timeout):
            await second.acquire(timeout=0.1, poll_interval=0.01)
    assert second.lock_counter == 0


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_blocking_false_raises(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    async with first:
        with pytest.raises(Timeout):
            await second.acquire(blocking=False)


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_default_timeout_applies(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path, timeout=0.1)
    async with first:
        with pytest.raises(Timeout):
            await second.acquire(poll_interval=0.01)


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_concurrent_tasks_serialised(lock_cls, lock_path):
    state = {"active": 0, "max": 0, "done": 0}

    async def worker():
        lock = lock_cls(lock_path)
        async with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            state["done"] += 1

    await asyncio.wait_for(asyncio.gather(*(worker() for _ in range(4))), timeout=30)
    assert state["max"] == 1
    assert state["done"] == 4


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_waiting_task_gets_lock_after_release(lock_cls, lock_path):
    first = lock_cls(lock_path)
    second = lock_cls(lock_path)
    await first.acquire()
    task = asyncio.ensure_future(second.acquire(timeout=10, poll_interval=0.01))
    await asyncio.sleep(0.1)
    assert not task.done()
    await first.release()
    await asyncio.wait_for(task, timeout=10)
    assert second.is_locked
    await second.release()


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_sync_lock_in_other_thread_contends_with_async(lock_cls, lock_path):
    from filelock import FileLock, SoftFileLock

    sync_cls = FileLock if lock_cls is AsyncFileLock else SoftFileLock
    outcome = []

    def worker():
        try:
            sync_cls(lock_path).acquire(timeout=0.1, poll_interval=0.01)
            outcome.append("acquired")
        except Timeout:
            outcome.append("timeout")

    async with lock_cls(lock_path):
        t = threading.Thread(target=worker)
        t.start()
        await asyncio.get_running_loop().run_in_executor(None, t.join, 10)
    assert outcome == ["timeout"]


@pytest.mark.parametrize("lock_cls", ASYNC_CLASSES)
async def test_run_in_executor_false(lock_cls, lock_path):
    lock = lock_cls(lock_path, run_in_executor=False)
    assert lock.run_in_executor is False
    async with lock:
        assert lock.is_locked
    assert not lock.is_locked


async def test_thread_local_with_executor_raises(lock_path):
    with pytest.raises(ValueError, match="run_in_executor"):
        AsyncFileLock(lock_path, thread_local=True, run_in_executor=True)


async def test_thread_local_without_executor_ok(lock_path):
    lock = AsyncFileLock(lock_path, thread_local=True, run_in_executor=False)
    assert lock.is_thread_local()
    async with lock:
        assert lock.is_locked


async def test_default_properties(lock_path):
    lock = AsyncFileLock(lock_path)
    assert lock.run_in_executor is True
    assert lock.executor is None
    assert lock.loop is None
    assert not lock.is_thread_local()
    assert lock.mode == 0o644
    assert isinstance(lock, BaseAsyncFileLock)


async def test_async_soft_lock_removes_file(lock_path):
    lock = AsyncSoftFileLock(lock_path)
    async with lock:
        assert lock_path.exists()
    assert not lock_path.exists()


async def test_async_lock_creates_parent_dirs(tmp_path):
    path = tmp_path / "x" / "y" / "z.lock"
    async with AsyncFileLock(path):
        assert path.exists()


async def test_explicit_loop_is_used(lock_path):
    loop = asyncio.get_running_loop()
    lock = AsyncFileLock(lock_path, loop=loop)
    assert lock.loop is loop
    async with lock:
        assert lock.is_locked


async def test_async_timeout_waits_roughly_timeout(lock_path):
    first = AsyncFileLock(lock_path)
    second = AsyncFileLock(lock_path)
    async with first:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            await second.acquire(timeout=0.2, poll_interval=0.02)
        assert time.perf_counter() - start >= 0.2
