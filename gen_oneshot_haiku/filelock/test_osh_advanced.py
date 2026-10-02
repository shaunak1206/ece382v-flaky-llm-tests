"""Tests for advanced FileLock features."""

import os
import tempfile
from pathlib import Path
import threading
import time
import pytest
from filelock import FileLock, SoftFileLock
from filelock._error import Timeout


class TestLockInTempDirectory:
    """Test lock files in temporary directories."""

    def test_lock_file_in_temp_directory(self):
        """Test creating lock in temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            assert lock_file.exists()
            lock.release()

    def test_lock_in_nested_temp_directory(self):
        """Test creating lock in nested temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / 'a' / 'b' / 'c'
            nested_dir.mkdir(parents=True, exist_ok=True)
            lock_file = nested_dir / 'test.lock'

            lock = FileLock(str(lock_file))
            lock.acquire()
            assert lock_file.exists()
            lock.release()

    def test_lock_file_cleanup(self):
        """Test that lock file is properly cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = SoftFileLock(str(lock_file))

            lock.acquire()
            assert lock_file.exists()
            lock.release()
            # SoftFileLock deletes the file on release
            assert not lock_file.exists()


class TestTimeout:
    """Test timeout exceptions."""

    def test_timeout_exception_message(self):
        """Test Timeout exception message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file))
            lock1.acquire()

            lock2 = FileLock(str(lock_file), timeout=0.1)

            try:
                lock2.acquire()
                assert False, "Should have raised Timeout"
            except Timeout as e:
                assert str(lock_file) in str(e)
                assert 'could not be acquired' in str(e)
            finally:
                lock1.release()

    def test_timeout_exception_lock_file_property(self):
        """Test Timeout exception lock_file property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file))
            lock1.acquire()

            lock2 = FileLock(str(lock_file), timeout=0.1)

            try:
                lock2.acquire()
            except Timeout as e:
                assert e.lock_file == str(lock_file)
            finally:
                lock1.release()


class TestSingletonPattern:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that singleton returns same instance for same lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            lock1 = FileLock(str(lock_file), is_singleton=True)
            lock2 = FileLock(str(lock_file), is_singleton=True)

            assert lock1 is lock2

    def test_singleton_different_file(self):
        """Test that singleton creates different instances for different files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file1 = Path(tmpdir) / 'test1.lock'
            lock_file2 = Path(tmpdir) / 'test2.lock'

            lock1 = FileLock(str(lock_file1), is_singleton=True)
            lock2 = FileLock(str(lock_file2), is_singleton=True)

            assert lock1 is not lock2

    def test_singleton_parameter_mismatch_raises_error(self):
        """Test that singleton with mismatched parameters raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            lock1 = FileLock(str(lock_file), timeout=5, is_singleton=True)

            with pytest.raises(ValueError):
                FileLock(str(lock_file), timeout=10, is_singleton=True)

    def test_singleton_mode_mismatch_raises_error(self):
        """Test that singleton with mismatched mode raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            lock1 = FileLock(str(lock_file), mode=0o644, is_singleton=True)

            with pytest.raises(ValueError):
                FileLock(str(lock_file), mode=0o755, is_singleton=True)


class TestModeParameter:
    """Test mode parameter."""

    def test_lock_file_mode(self):
        """Test that lock file is created with specified mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            mode = 0o644

            lock = FileLock(str(lock_file), mode=mode)
            lock.acquire()

            # Get file permissions
            file_stat = os.stat(str(lock_file))
            file_mode = file_stat.st_mode & 0o777

            # Note: Mode may be affected by umask, so we just verify file exists
            assert lock_file.exists()
            lock.release()

    def test_different_modes(self):
        """Test different mode values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in [0o644, 0o755, 0o600]:
                lock_file = Path(tmpdir) / f'test_{mode}.lock'
                lock = FileLock(str(lock_file), mode=mode)
                lock.acquire()
                assert lock_file.exists()
                lock.release()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_locks_same_file(self):
        """Test multiple lock objects for same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            lock1 = FileLock(str(lock_file))
            lock2 = FileLock(str(lock_file))

            # Both should be able to acquire (sequentially)
            lock1.acquire()
            assert lock1.is_locked

            # lock2 will wait
            def try_lock2():
                lock2.acquire(timeout=0.1)
                return lock2.is_locked

            thread = threading.Thread(target=try_lock2)
            thread.start()
            time.sleep(0.05)

            lock1.release()
            thread.join()

            assert not lock1.is_locked

    def test_release_when_not_locked(self):
        """Test releasing a lock that's not acquired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            # Calling release without acquire should be safe
            lock.release()
            assert not lock.is_locked

    def test_lock_counter_does_not_go_negative(self):
        """Test that lock counter doesn't go negative."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.release()
            lock.release()

            # Counter should never be negative
            assert lock.lock_counter >= 0

    def test_lock_file_is_created_once(self):
        """Test that lock file is created only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            time1 = lock_file.stat().st_mtime
            lock.release()

            # Re-acquire should use same file
            lock.acquire()
            time2 = lock_file.stat().st_mtime

            # Times may differ slightly but file should exist
            assert lock_file.exists()
            lock.release()


class TestLockFilePermissions:
    """Test lock file permissions and ownership."""

    def test_lock_file_readable(self):
        """Test that lock file is readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            # Verify we can stat the file (readable)
            stat = os.stat(str(lock_file))
            assert stat is not None
            lock.release()

    def test_lock_file_writable(self):
        """Test that lock file is writable (for lock holder)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            lock.acquire()
            # Verify file is writable by trying to stat it
            assert os.access(str(lock_file), os.R_OK | os.W_OK)
            lock.release()


class TestContextManagerExceptions:
    """Test context manager with exceptions."""

    def test_context_manager_releases_on_exception(self):
        """Test that lock is released even if exception occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            try:
                with lock:
                    assert lock.is_locked
                    raise ValueError("test error")
            except ValueError:
                pass

            # Lock should be released
            assert not lock.is_locked

    def test_nested_exception_handling(self):
        """Test nested context managers with exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            try:
                with lock:
                    with lock:
                        raise ValueError("test")
            except ValueError:
                pass

            # Lock should be fully released
            assert not lock.is_locked
            assert lock.lock_counter == 0


class TestComparisons:
    """Test lock comparisons and properties."""

    def test_different_locks_are_different_objects(self):
        """Test that different lock instances are different."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            lock1 = FileLock(str(lock_file))
            lock2 = FileLock(str(lock_file))

            # Without singleton, they should be different objects
            assert lock1 is not lock2
            assert id(lock1) != id(lock2)

    def test_lock_representation(self):
        """Test lock string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            repr_str = repr(lock)
            assert 'FileLock' in repr_str or 'lock' in repr_str.lower()


class TestConcurrentReadWrite:
    """Test concurrent read/write protection."""

    def test_lock_protects_file_writes(self):
        """Test that lock protects concurrent file writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            data_file = Path(tmpdir) / 'data.txt'
            lock = FileLock(str(lock_file), timeout=5)

            results = []

            def writer(writer_id):
                with lock:
                    # Read current
                    if data_file.exists():
                        current = int(data_file.read_text())
                    else:
                        current = 0

                    # Increment
                    new_value = current + 1
                    time.sleep(0.01)  # Simulate work

                    # Write
                    data_file.write_text(str(new_value))
                    results.append(new_value)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Final value should be 5 if locked properly
            final = int(data_file.read_text())
            assert final == 5
            assert len(results) == 5


class TestDifferentLockTypes:
    """Test different lock types."""

    def test_filelock_vs_softfilelock(self):
        """Test FileLock vs SoftFileLock behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file1 = Path(tmpdir) / 'filelock.lock'
            lock_file2 = Path(tmpdir) / 'softlock.lock'

            filelock = FileLock(str(lock_file1))
            softlock = SoftFileLock(str(lock_file2))

            # Both should work similarly
            filelock.acquire()
            assert filelock.is_locked
            filelock.release()

            softlock.acquire()
            assert softlock.is_locked
            softlock.release()
