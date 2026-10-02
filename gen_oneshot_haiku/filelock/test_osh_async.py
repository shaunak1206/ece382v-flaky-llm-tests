"""Tests for AsyncFileLock functionality."""

import asyncio
import tempfile
from pathlib import Path
import pytest
from filelock import AsyncFileLock, AsyncSoftFileLock
from filelock._error import Timeout


@pytest.mark.asyncio
class TestAsyncFileLockBasic:
    """Test basic AsyncFileLock functionality."""

    async def test_async_filelock_acquire_and_release(self):
        """Test acquiring and releasing an async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            assert not lock.is_locked
            await lock.acquire()
            assert lock.is_locked
            await lock.release()
            assert not lock.is_locked

    async def test_async_filelock_context_manager(self):
        """Test AsyncFileLock as async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            assert not lock.is_locked
            async with lock:
                assert lock.is_locked
            assert not lock.is_locked

    async def test_async_acquire_returns_proxy(self):
        """Test that acquire() returns a proxy object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            proxy = await lock.acquire()
            assert proxy is not None
            await lock.release()

    async def test_async_acquire_proxy_context_manager(self):
        """Test using acquire() return value as async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            async with await lock.acquire():
                assert lock.is_locked
            assert not lock.is_locked

    async def test_async_properties(self):
        """Test AsyncFileLock properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), timeout=5, mode=0o644)

            assert lock.lock_file == str(lock_file)
            assert lock.timeout == 5
            assert lock.mode == 0o644


@pytest.mark.asyncio
class TestAsyncReentrantLocking:
    """Test reentrant locking with async locks."""

    async def test_async_reentrant_locking(self):
        """Test acquiring async lock multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            assert lock.lock_counter == 0
            await lock.acquire()
            assert lock.lock_counter == 1
            assert lock.is_locked

            await lock.acquire()
            assert lock.lock_counter == 2
            assert lock.is_locked

            await lock.release()
            assert lock.lock_counter == 1
            assert lock.is_locked

            await lock.release()
            assert lock.lock_counter == 0
            assert not lock.is_locked

    async def test_async_reentrant_context_managers(self):
        """Test nested async context managers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            async with lock:
                assert lock.lock_counter == 1
                async with lock:
                    assert lock.lock_counter == 2
                    assert lock.is_locked
                assert lock.lock_counter == 1
                assert lock.is_locked
            assert lock.lock_counter == 0
            assert not lock.is_locked


@pytest.mark.asyncio
class TestAsyncConcurrency:
    """Test concurrent async operations."""

    async def test_async_multiple_tasks_sequential_access(self):
        """Test that multiple tasks access lock sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), timeout=5)

            results = []

            async def worker(worker_id):
                async with lock:
                    results.append(f'start_{worker_id}')
                    await asyncio.sleep(0.01)
                    results.append(f'end_{worker_id}')

            await asyncio.gather(
                worker(0),
                worker(1),
                worker(2)
            )

            assert len(results) == 6
            # Check that each worker's start and end are consecutive
            for i in range(3):
                start_idx = results.index(f'start_{i}')
                end_idx = results.index(f'end_{i}')
                assert end_idx == start_idx + 1

    async def test_async_task_waits_for_lock(self):
        """Test that a task waits for lock to be released."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), timeout=5)

            events = []

            async def task1():
                async with lock:
                    events.append('task1_acquired')
                    await asyncio.sleep(0.05)
                    events.append('task1_released')

            async def task2():
                await asyncio.sleep(0.01)
                async with lock:
                    events.append('task2_acquired')
                    events.append('task2_released')

            await asyncio.gather(task1(), task2())

            assert events.index('task1_acquired') < events.index('task2_acquired')

    async def test_async_many_tasks_competing(self):
        """Test many tasks competing for same lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), timeout=10)

            results = []

            async def worker(worker_id):
                async with lock:
                    results.append(worker_id)
                    await asyncio.sleep(0.01)

            await asyncio.gather(
                *[worker(i) for i in range(10)]
            )

            assert len(results) == 10
            assert len(set(results)) == 10


@pytest.mark.asyncio
class TestAsyncTimeout:
    """Test timeout behavior with async locks."""

    async def test_async_timeout_waiting_for_lock(self):
        """Test timeout when waiting for async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = AsyncFileLock(str(lock_file))
            await lock1.acquire()

            lock2 = AsyncFileLock(str(lock_file), timeout=0.1)

            with pytest.raises(Timeout):
                await lock2.acquire()

            await lock1.release()

    async def test_async_acquire_with_explicit_timeout(self):
        """Test acquire() with explicit timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = AsyncFileLock(str(lock_file))
            await lock1.acquire()

            lock2 = AsyncFileLock(str(lock_file), timeout=5)

            with pytest.raises(Timeout):
                await lock2.acquire(timeout=0.1)

            await lock1.release()

    async def test_async_poll_interval(self):
        """Test poll_interval with async locks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            await lock.acquire(poll_interval=0.01)
            assert lock.is_locked
            await lock.release()


@pytest.mark.asyncio
class TestAsyncBlocking:
    """Test blocking behavior with async locks."""

    async def test_async_blocking_false(self):
        """Test blocking=False with async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = AsyncFileLock(str(lock_file))
            await lock1.acquire()

            lock2 = AsyncFileLock(str(lock_file))

            with pytest.raises(Timeout):
                await lock2.acquire(blocking=False)

            await lock1.release()

    async def test_async_acquire_with_blocking_param(self):
        """Test acquire() with blocking parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = AsyncFileLock(str(lock_file), timeout=1)
            await lock1.acquire()

            lock2 = AsyncFileLock(str(lock_file), timeout=1)

            with pytest.raises(Timeout):
                await lock2.acquire(blocking=False)

            await lock1.release()


@pytest.mark.asyncio
class TestAsyncSoftFileLock:
    """Test AsyncSoftFileLock implementation."""

    async def test_async_soft_filelock_acquire_release(self):
        """Test AsyncSoftFileLock acquire and release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncSoftFileLock(str(lock_file))

            assert not lock.is_locked
            await lock.acquire()
            assert lock.is_locked
            await lock.release()
            assert not lock.is_locked

    async def test_async_soft_lock_context_manager(self):
        """Test AsyncSoftFileLock as async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncSoftFileLock(str(lock_file))

            async with lock:
                assert lock.is_locked
            assert not lock.is_locked


@pytest.mark.asyncio
class TestAsyncForceRelease:
    """Test force release with async locks."""

    async def test_async_force_release(self):
        """Test force release with async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            await lock.acquire()
            await lock.acquire()
            assert lock.lock_counter == 2

            await lock.release(force=True)
            assert lock.lock_counter == 0
            assert not lock.is_locked


@pytest.mark.asyncio
class TestAsyncWithSync:
    """Test async locks don't work with sync context managers."""

    async def test_async_lock_with_statement_raises(self):
        """Test that using async lock with 'with' raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file))

            with pytest.raises(NotImplementedError):
                with lock:
                    pass


@pytest.mark.asyncio
class TestAsyncThreadLocal:
    """Test thread_local parameter with async locks."""

    async def test_async_thread_local_true(self):
        """Test AsyncFileLock with thread_local=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            # thread_local=True is allowed but run_in_executor must be False
            lock = AsyncFileLock(str(lock_file), thread_local=True, run_in_executor=False)

            assert lock.is_thread_local()

    async def test_async_thread_local_false(self):
        """Test AsyncFileLock with thread_local=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), thread_local=False)

            assert not lock.is_thread_local()

    async def test_async_thread_local_with_run_in_executor_raises(self):
        """Test that thread_local=True with run_in_executor=True raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            with pytest.raises(ValueError):
                AsyncFileLock(
                    str(lock_file),
                    thread_local=True,
                    run_in_executor=True
                )


@pytest.mark.asyncio
class TestAsyncProperties:
    """Test AsyncFileLock property management."""

    async def test_async_timeout_property(self):
        """Test timeout property with async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_file), timeout=5)

            assert lock.timeout == 5
            # Note: Setting timeout on async lock may not be directly settable
            # but we test the property exists

    async def test_async_lock_file_path(self):
        """Test lock_file property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / 'test.lock'
            lock = AsyncFileLock(str(lock_path))

            assert lock.lock_file == str(lock_path)
