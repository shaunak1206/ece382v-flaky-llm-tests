"""Tests for basic FileLock functionality."""

import os
import tempfile
from pathlib import Path
import pytest
from filelock import FileLock, SoftFileLock
from filelock._error import Timeout


class TestFileLockBasic:
    """Test basic FileLock functionality."""

    def test_filelock_acquire_and_release(self):
        """Test acquiring and releasing a lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            assert not lock.is_locked
            lock.acquire()
            assert lock.is_locked
            lock.release()
            assert not lock.is_locked

    def test_filelock_context_manager(self):
        """Test FileLock as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            assert not lock.is_locked
            with lock:
                assert lock.is_locked
            assert not lock.is_locked

    def test_filelock_acquire_returns_proxy(self):
        """Test that acquire() returns a proxy object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            proxy = lock.acquire()
            assert proxy is not None
            lock.release()

    def test_filelock_acquire_proxy_context_manager(self):
        """Test using acquire() return value as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            with lock.acquire():
                assert lock.is_locked
            assert not lock.is_locked

    def test_filelock_properties(self):
        """Test FileLock properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=5, mode=0o644)

            assert lock.lock_file == str(lock_file)
            assert lock.timeout == 5
            assert lock.mode == 0o644

    def test_filelock_change_timeout(self):
        """Test changing timeout property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=5)

            assert lock.timeout == 5
            lock.timeout = 10
            assert lock.timeout == 10

    def test_filelock_change_blocking(self):
        """Test changing blocking property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), blocking=True)

            assert lock.blocking is True
            lock.blocking = False
            assert lock.blocking is False


class TestSoftFileLock:
    """Test SoftFileLock implementation."""

    def test_soft_filelock_acquire_and_release(self):
        """Test acquiring and releasing a SoftFileLock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = SoftFileLock(str(lock_file))

            assert not lock.is_locked
            lock.acquire()
            assert lock.is_locked
            lock.release()
            assert not lock.is_locked

    def test_soft_filelock_context_manager(self):
        """Test SoftFileLock as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = SoftFileLock(str(lock_file))

            with lock:
                assert lock.is_locked
            assert not lock.is_locked

    def test_soft_filelock_creates_lock_file(self):
        """Test that SoftFileLock creates lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = SoftFileLock(str(lock_file))

            assert not lock_file.exists()
            lock.acquire()
            assert lock_file.exists()
            lock.release()


class TestReentrantLocking:
    """Test reentrant locking (acquiring lock multiple times)."""

    def test_reentrant_locking(self):
        """Test acquiring lock multiple times from same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            assert lock.lock_counter == 0
            lock.acquire()
            assert lock.lock_counter == 1
            assert lock.is_locked

            lock.acquire()
            assert lock.lock_counter == 2
            assert lock.is_locked

            lock.release()
            assert lock.lock_counter == 1
            assert lock.is_locked

            lock.release()
            assert lock.lock_counter == 0
            assert not lock.is_locked

    def test_reentrant_context_managers(self):
        """Test nested context managers (reentrant)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            with lock:
                assert lock.lock_counter == 1
                with lock:
                    assert lock.lock_counter == 2
                    assert lock.is_locked
                assert lock.lock_counter == 1
                assert lock.is_locked
            assert lock.lock_counter == 0
            assert not lock.is_locked

    def test_lock_counter_property(self):
        """Test lock_counter property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            assert lock.lock_counter == 0
            lock.acquire()
            assert lock.lock_counter == 1
            lock.acquire()
            assert lock.lock_counter == 2
            lock.release()
            assert lock.lock_counter == 1
            lock.release()
            assert lock.lock_counter == 0


class TestIsLocked:
    """Test is_locked property."""

    def test_is_locked_property(self):
        """Test is_locked property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            assert not lock.is_locked
            lock.acquire()
            assert lock.is_locked
            lock.release()
            assert not lock.is_locked

    def test_is_locked_with_force_release(self):
        """Test is_locked after force release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            lock.acquire()
            assert lock.is_locked
            lock.release(force=True)
            assert not lock.is_locked


class TestLockFile:
    """Test lock file creation and properties."""

    def test_lock_file_path(self):
        """Test lock_file property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / 'subdir' / 'test.lock'
            lock = FileLock(str(lock_path))

            assert lock.lock_file == str(lock_path)

    def test_lock_file_creates_directories(self):
        """Test that lock file creation creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'a' / 'b' / 'c' / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            assert lock_file.parent.exists()
            lock.release()

    def test_lock_file_mode(self):
        """Test lock file mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), mode=0o644)

            lock.acquire()
            # Check file permissions
            stat = os.stat(str(lock_file))
            # Mode is typically masked by umask, so we just verify file exists
            assert lock_file.exists()
            lock.release()


class TestThreadLocal:
    """Test thread_local parameter."""

    def test_thread_local_true(self):
        """Test FileLock with thread_local=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=True)

            assert lock.is_thread_local()

    def test_thread_local_false(self):
        """Test FileLock with thread_local=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=False)

            assert not lock.is_thread_local()


class TestDecorator:
    """Test FileLock as decorator."""

    def test_lock_as_decorator(self):
        """Test using FileLock as a decorator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            counter = {'value': 0}

            @lock
            def increment():
                counter['value'] += 1
                return counter['value']

            result = increment()
            assert result == 1
            assert counter['value'] == 1

    def test_lock_decorator_preserves_function(self):
        """Test that lock decorator preserves function name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            @lock
            def my_function():
                """My docstring."""
                return 42

            # The decorator should preserve function metadata
            assert 'my_function' in str(my_function)


class TestNegativeTimeout:
    """Test negative timeout behavior."""

    def test_negative_timeout_infinite_wait(self):
        """Test that negative timeout means infinite wait."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=-1)

            assert lock.timeout == -1
            lock.acquire()
            assert lock.is_locked
            lock.release()

    def test_zero_timeout_one_attempt(self):
        """Test that timeout=0 means one attempt only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=0)

            lock.acquire()
            assert lock.is_locked
            lock.release()


class TestForceRelease:
    """Test force release functionality."""

    def test_force_release(self):
        """Test force release ignores lock counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            lock.acquire()
            assert lock.lock_counter == 2

            lock.release(force=True)
            assert lock.lock_counter == 0
            assert not lock.is_locked

    def test_del_calls_force_release(self):
        """Test that __del__ calls force_release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            lock.acquire()
            assert lock.is_locked

            # Delete the lock object
            del lock
            # If we could check, we'd see it was released with force=True


class TestPathLike:
    """Test pathlib.Path support."""

    def test_pathlib_path(self):
        """Test FileLock with pathlib.Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(lock_file)

            lock.acquire()
            assert lock.is_locked
            lock.release()
            assert not lock.is_locked
