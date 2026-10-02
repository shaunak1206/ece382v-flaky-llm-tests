"""Several threads competing for the same lock file."""
import threading
import time

import pytest

from filelock import FileLock, SoftFileLock, Timeout


@pytest.fixture(params=[FileLock, SoftFileLock], ids=["FileLock", "SoftFileLock"])
def lock_cls(request):
    return request.param


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "threads.lock")


def run_in_threads(target, count, join_timeout=60):
    """Run ``target(index)`` in ``count`` threads and fail if any of them raised."""
    errors = []

    def wrapper(index):
        try:
            target(index)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(join_timeout)
    assert not any(thread.is_alive() for thread in threads), "threads did not finish"
    assert errors == []


class CriticalSection:
    """Tracks how many threads are inside the protected region at once."""

    def __init__(self):
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.entries = 0

    def enter(self):
        with self._guard:
            self.active += 1
            self.entries += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self):
        with self._guard:
            self.active -= 1


def test_threads_with_separate_lock_objects_are_mutually_exclusive(lock_cls, lock_path):
    section = CriticalSection()

    def worker(_):
        for _i in range(4):
            lock = lock_cls(lock_path)
            with lock.acquire(timeout=30, poll_interval=0.005):
                section.enter()
                time.sleep(0.002)
                section.leave()

    run_in_threads(worker, 6)
    assert section.max_active == 1
    assert section.entries == 24
    assert section.active == 0


def test_threads_sharing_one_lock_object_are_mutually_exclusive(lock_cls, lock_path):
    lock = lock_cls(lock_path)  # thread_local defaults to True
    section = CriticalSection()

    def worker(_):
        for _i in range(4):
            with lock.acquire(timeout=30, poll_interval=0.005):
                section.enter()
                time.sleep(0.002)
                section.leave()
            assert not lock.is_locked

    run_in_threads(worker, 5)
    assert section.max_active == 1
    assert section.entries == 20
    assert not lock.is_locked


def test_read_modify_write_counter_has_no_lost_updates(lock_cls, tmp_path):
    counter = tmp_path / "counter.txt"
    counter.write_text("0")
    path = str(tmp_path / "counter.lock")

    def worker(_):
        for _i in range(8):
            with lock_cls(path).acquire(timeout=30, poll_interval=0.002):
                value = int(counter.read_text())
                counter.write_text(str(value + 1))

    run_in_threads(worker, 5)
    assert int(counter.read_text()) == 40


def test_only_one_thread_wins_a_non_blocking_race(lock_cls, lock_path):
    count = 6
    start = threading.Barrier(count)
    attempted = threading.Barrier(count)
    results = []
    results_guard = threading.Lock()

    def worker(_):
        lock = lock_cls(lock_path)
        start.wait(10)
        try:
            lock.acquire(blocking=False)
            won = True
        except Timeout:
            won = False
        with results_guard:
            results.append(won)
        attempted.wait(10)
        if won:
            lock.release()

    run_in_threads(worker, count)
    assert len(results) == count
    assert results.count(True) == 1
    follow_up = lock_cls(lock_path)
    follow_up.acquire(timeout=1, poll_interval=0.01)
    assert follow_up.is_locked
    follow_up.release()


def test_every_waiting_thread_times_out_while_the_lock_is_held(lock_cls, lock_path):
    holder = lock_cls(lock_path)
    holder.acquire()
    outcomes = []
    guard = threading.Lock()

    def worker(_):
        lock = lock_cls(lock_path)
        try:
            lock.acquire(timeout=0.1, poll_interval=0.01)
            result = "acquired"
        except Timeout:
            result = "timeout"
        with guard:
            outcomes.append(result)

    try:
        run_in_threads(worker, 4)
    finally:
        holder.release()
    assert outcomes == ["timeout"] * 4


def test_thread_local_lock_state_is_not_shared_between_threads(lock_cls, lock_path):
    lock = lock_cls(lock_path, thread_local=True)
    lock.acquire()
    outcome = {}

    def other(_):
        outcome["locked"] = lock.is_locked
        outcome["counter"] = lock.lock_counter
        try:
            lock.acquire(timeout=0.1, poll_interval=0.01)
            outcome["acquired"] = True
        except Timeout:
            outcome["acquired"] = False

    try:
        run_in_threads(other, 1)
    finally:
        lock.release()
    assert outcome == {"locked": False, "counter": 0, "acquired": False}
    assert not lock.is_locked


def test_non_thread_local_lock_is_reentrant_across_threads(lock_cls, lock_path):
    lock = lock_cls(lock_path, thread_local=False)
    assert lock.is_thread_local() is False
    lock.acquire()
    outcome = {}

    def other(_):
        lock.acquire(timeout=1, poll_interval=0.01)
        outcome["counter"] = lock.lock_counter
        lock.release()

    run_in_threads(other, 1)
    assert outcome["counter"] == 2
    assert lock.is_locked
    assert lock.lock_counter == 1
    lock.release()
    assert not lock.is_locked


def test_non_thread_local_lock_can_be_released_from_another_thread(lock_cls, lock_path):
    lock = lock_cls(lock_path, thread_local=False)
    lock.acquire()
    run_in_threads(lambda _: lock.release(), 1)
    assert not lock.is_locked
    assert lock.lock_counter == 0
