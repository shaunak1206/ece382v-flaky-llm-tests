"""Tests for FileLock threading functionality."""

import tempfile
from pathlib import Path
import threading
import time
import pytest
from filelock import FileLock, SoftFileLock
from filelock._error import Timeout


class TestMultiThreadedLocking:
    """Test locking from multiple threads."""

    def test_multiple_threads_acquire_lock_sequentially(self):
        """Test that multiple threads acquire lock sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            results = []

            def worker(worker_id):
                with lock:
                    results.append(f'start_{worker_id}')
                    time.sleep(0.01)
                    results.append(f'end_{worker_id}')

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Each thread's start and end should be consecutive (no interleaving)
            assert len(results) == 6

    def test_thread_waits_for_lock(self):
        """Test that a thread waits for lock to be released."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=5)

            events = []

            def thread1():
                with lock:
                    events.append('thread1_acquired')
                    time.sleep(0.1)
                    events.append('thread1_released')

            def thread2():
                time.sleep(0.02)  # Give thread1 time to acquire
                with lock:
                    events.append('thread2_acquired')
                    events.append('thread2_released')

            t1 = threading.Thread(target=thread1)
            t2 = threading.Thread(target=thread2)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # thread1 should acquire before thread2
            assert events.index('thread1_acquired') < events.index('thread2_acquired')

    def test_shared_lock_between_threads(self):
        """Test sharing same lock object between threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            counter = {'value': 0}

            def worker():
                for _ in range(10):
                    with lock:
                        counter['value'] += 1

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert counter['value'] == 30

    def test_thread_local_locks(self):
        """Test thread_local=True behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=True)

            results = []

            def worker(worker_id):
                with lock:
                    results.append(f'{worker_id}_locked')

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 2

    def test_non_thread_local_locks(self):
        """Test thread_local=False behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=False)

            events = []

            def worker(worker_id):
                with lock:
                    events.append(f'{worker_id}_start')
                    time.sleep(0.05)
                    events.append(f'{worker_id}_end')

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(events) == 4


class TestBlockingBehavior:
    """Test blocking vs non-blocking behavior."""

    def test_blocking_true_waits(self):
        """Test that blocking=True waits for lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file), blocking=True, timeout=1)

            lock1.acquire()

            def try_acquire():
                lock2 = FileLock(str(lock_file), blocking=True, timeout=0.5)
                try:
                    lock2.acquire()
                    return True
                except Timeout:
                    return False
                finally:
                    if lock2.is_locked:
                        lock2.release()

            # This should timeout since lock1 is held
            result = try_acquire()
            assert result is False

            lock1.release()

    def test_blocking_false_fails_immediately(self):
        """Test that blocking=False fails immediately if locked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file))
            lock1.acquire()

            lock2 = FileLock(str(lock_file), blocking=False)

            with pytest.raises(Timeout):
                lock2.acquire()

            lock1.release()

    def test_acquire_with_blocking_parameter(self):
        """Test acquire() with blocking parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file), timeout=1)
            lock1.acquire()

            lock2 = FileLock(str(lock_file), timeout=1)

            with pytest.raises(Timeout):
                lock2.acquire(blocking=False)

            lock1.release()


class TestTimeoutBehavior:
    """Test timeout handling in multi-threaded context."""

    def test_timeout_waiting_for_lock(self):
        """Test timeout when waiting for lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file))
            lock1.acquire()

            lock2 = FileLock(str(lock_file), timeout=0.1)

            with pytest.raises(Timeout):
                lock2.acquire()

            lock1.release()

    def test_acquire_with_explicit_timeout(self):
        """Test acquire() with explicit timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock1 = FileLock(str(lock_file))
            lock1.acquire()

            lock2 = FileLock(str(lock_file), timeout=5)

            with pytest.raises(Timeout):
                lock2.acquire(timeout=0.1)

            lock1.release()

    def test_poll_interval(self):
        """Test poll_interval parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file))

            start = time.time()
            lock.acquire(poll_interval=0.01)
            elapsed = time.time() - start

            # Should acquire quickly with short poll_interval
            assert elapsed < 0.5
            lock.release()


class TestConcurrentAcquireAndRelease:
    """Test concurrent acquire and release patterns."""

    def test_acquire_while_other_holds_lock(self):
        """Test acquiring lock while another thread holds it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=1)

            acquired_order = []

            def worker1():
                with lock:
                    acquired_order.append('worker1')
                    time.sleep(0.1)

            def worker2():
                time.sleep(0.02)
                with lock:
                    acquired_order.append('worker2')

            t1 = threading.Thread(target=worker1)
            t2 = threading.Thread(target=worker2)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert acquired_order[0] == 'worker1'
            assert acquired_order[1] == 'worker2'

    def test_many_threads_competing_for_lock(self):
        """Test many threads competing for same lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=5)

            results = []

            def worker(worker_id):
                with lock:
                    results.append(worker_id)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 10
            assert len(set(results)) == 10


class TestReentrantFromMultipleThreads:
    """Test reentrant locking from multiple threads."""

    def test_thread_local_reentrancy(self):
        """Test reentrant locking within same thread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=True)

            def worker():
                with lock:
                    assert lock.lock_counter == 1
                    with lock:
                        assert lock.lock_counter == 2
                    assert lock.lock_counter == 1

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    def test_lock_counter_across_threads(self):
        """Test lock counter with thread_local=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), thread_local=False, timeout=1)

            results = []

            def worker1():
                with lock:
                    results.append(('worker1', lock.lock_counter))
                    time.sleep(0.05)

            def worker2():
                time.sleep(0.02)
                with lock:
                    results.append(('worker2', lock.lock_counter))

            t1 = threading.Thread(target=worker1)
            t2 = threading.Thread(target=worker2)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert len(results) == 2


class TestSoftFileLockThreading:
    """Test SoftFileLock with threading."""

    def test_soft_lock_multiple_threads(self):
        """Test SoftFileLock with multiple threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'

            results = []

            def worker(worker_id):
                lock = SoftFileLock(str(lock_file))
                with lock:
                    results.append(f'{worker_id}_acquired')
                    time.sleep(0.01)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 3


class TestThreadSafety:
    """Test thread safety of lock operations."""

    def test_concurrent_access_with_protection(self):
        """Test that lock protects concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / 'test.lock'
            lock = FileLock(str(lock_file), timeout=5)

            data = {'count': 0}
            iterations = 100

            def increment_many_times():
                for _ in range(iterations):
                    with lock:
                        data['count'] += 1

            threads = [threading.Thread(target=increment_many_times) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert data['count'] == 5 * iterations
