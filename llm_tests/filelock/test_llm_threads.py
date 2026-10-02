"""Locking from several threads, thread_local and singleton behaviour."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from filelock import FileLock, SoftFileLock, Timeout

LOCK_CLASSES = [FileLock, SoftFileLock]


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "thr.lock"


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_threads_increment_shared_counter(lock_cls, lock_path):
    counter = {"value": 0}
    errors = []

    def worker():
        try:
            lock = lock_cls(lock_path)
            for _ in range(5):
                with lock:
                    current = counter["value"]
                    time.sleep(0.001)
                    counter["value"] = current + 1
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert counter["value"] == 20


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_mutual_exclusion_no_overlap(lock_cls, lock_path):
    active = []
    max_active = []
    guard = threading.Lock()

    def worker():
        lock = lock_cls(lock_path)
        for _ in range(3):
            with lock:
                with guard:
                    active.append(1)
                    max_active.append(len(active))
                time.sleep(0.005)
                with guard:
                    active.pop()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert max(max_active) == 1


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_shared_lock_object_default_thread_local(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    assert lock.is_thread_local()
    results = {}

    def worker():
        results["locked_in_thread"] = lock.is_locked
        results["counter_in_thread"] = lock.lock_counter

    with lock:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        assert lock.is_locked
    assert results == {"locked_in_thread": False, "counter_in_thread": 0}


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_thread_local_false_shares_state(lock_cls, lock_path):
    lock = lock_cls(lock_path, thread_local=False)
    assert not lock.is_thread_local()
    results = {}

    def worker():
        results["locked"] = lock.is_locked
        results["counter"] = lock.lock_counter

    with lock:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
    assert results == {"locked": True, "counter": 1}


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_thread_local_false_release_from_other_thread(lock_cls, lock_path):
    lock = lock_cls(lock_path, thread_local=False)
    lock.acquire()

    def worker():
        lock.release()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert not lock.is_locked


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_thread_local_lock_blocks_other_thread_until_release(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    acquired_in_thread = threading.Event()
    lock.acquire()

    def worker():
        lock.acquire(timeout=10, poll_interval=0.01)
        acquired_in_thread.set()
        lock.release()

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)
    assert not acquired_in_thread.is_set()
    lock.release()
    t.join(timeout=10)
    assert acquired_in_thread.is_set()


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_other_thread_times_out_while_held(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    outcome = []

    def worker():
        try:
            lock.acquire(timeout=0.1, poll_interval=0.01)
            outcome.append("acquired")
        except Timeout:
            outcome.append("timeout")

    with lock:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
    assert outcome == ["timeout"]


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_other_thread_blocking_false_fails_fast(lock_cls, lock_path):
    lock = lock_cls(lock_path)
    outcome = []

    def worker():
        try:
            lock.acquire(blocking=False)
            outcome.append("acquired")
        except Timeout:
            outcome.append("timeout")

    with lock:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
    assert outcome == ["timeout"]


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_ordered_handoff_between_threads(lock_cls, lock_path):
    order = []
    lock = lock_cls(lock_path)
    started = threading.Event()

    def first():
        with lock:
            started.set()
            time.sleep(0.1)
            order.append("first")

    def second():
        started.wait(10)
        with lock:
            order.append("second")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert order == ["first", "second"]


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_many_threads_all_acquire_eventually(lock_cls, lock_path):
    done = []
    guard = threading.Lock()

    def worker(i):
        lock = lock_cls(lock_path)
        with lock.acquire(timeout=20, poll_interval=0.005):
            time.sleep(0.005)
        with guard:
            done.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(done) == list(range(8))


@pytest.mark.parametrize("lock_cls", LOCK_CLASSES)
def test_exactly_one_thread_wins_nonblocking(lock_cls, lock_path):
    barrier = threading.Barrier(4)
    results = []
    guard = threading.Lock()
    release_event = threading.Event()

    def worker():
        lock = lock_cls(lock_path)
        barrier.wait(10)
        try:
            lock.acquire(blocking=False)
        except Timeout:
            with guard:
                results.append(False)
            return
        with guard:
            results.append(True)
        release_event.wait(10)
        lock.release()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    release_event.set()
    for t in threads:
        t.join(timeout=10)
    assert results.count(True) == 1
    assert results.count(False) == 3


def test_singleton_returns_same_instance(lock_path):
    a = FileLock(lock_path, is_singleton=True)
    b = FileLock(lock_path, is_singleton=True)
    assert a is b
    assert a.is_singleton


def test_non_singleton_returns_distinct_instances(lock_path):
    a = FileLock(lock_path)
    b = FileLock(lock_path)
    assert a is not b
    assert not a.is_singleton


def test_singleton_reentrant_across_objects(lock_path):
    a = FileLock(lock_path, is_singleton=True)
    with a:
        b = FileLock(lock_path, is_singleton=True)
        with b:
            assert a.lock_counter == 2
    assert not a.is_locked


def test_singleton_with_different_args_raises(lock_path):
    first = FileLock(lock_path, is_singleton=True, timeout=1)
    with pytest.raises(ValueError, match="Non-matching arguments"):
        FileLock(lock_path, is_singleton=True, timeout=2)
    assert first.timeout == 1


def test_singletons_per_class_are_separate(lock_path):
    a = FileLock(lock_path, is_singleton=True)
    b = SoftFileLock(lock_path, is_singleton=True)
    assert a is not b
