import types

import pytest

import Tracker.event as event_mod


@pytest.fixture(autouse=True)
def restore_event_state():
    original_finished_descriptor = event_mod.Event.__dict__["finished"]
    original_status_finish = event_mod.status._finish
    try:
        yield
    finally:
        # Restore the property on the class in case a decorator call overwrote it.
        setattr(event_mod.Event, "finished", original_finished_descriptor)
        event_mod.status._finish = original_status_finish
        # Remove any instance-level shadowing if created.
        event_mod.status.__dict__.pop("finished", None)


def test_status_is_event_instance():
    assert isinstance(event_mod.status, event_mod.Event)


def test_new_event_starts_not_finished():
    ev = event_mod.Event()
    assert ev.finished is False


def test_status_starts_not_finished():
    assert event_mod.status.finished is False


def test_finished_setter_accepts_true():
    ev = event_mod.Event()
    ev.finished = True
    assert ev.finished is True


def test_finished_setter_ignores_false_when_currently_false():
    ev = event_mod.Event()
    ev.finished = False
    assert ev.finished is False


def test_finished_setter_ignores_false_when_currently_true():
    ev = event_mod.Event()
    ev.finished = True
    ev.finished = False
    assert ev.finished is True


def test_multiple_instances_do_not_share_finish_state():
    ev1 = event_mod.Event()
    ev2 = event_mod.Event()
    ev1.finished = True
    assert ev1.finished is True
    assert ev2.finished is False


def test_check_decorator_returns_function_result(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    @event_mod.Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_decorator_preserves_function_metadata(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    @event_mod.Event.check
    def sample_func():
        """sample docstring"""
        return "ok"

    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "sample docstring"


def test_check_decorator_calls_sleep_with_point_one(monkeypatch):
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr(event_mod.time, "sleep", fake_sleep)

    @event_mod.Event.check
    def noop():
        return "done"

    assert noop() == "done"
    assert calls == [0.1]


def test_check_decorator_leaves_class_finished_false_after_call(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    @event_mod.Event.check
    def noop():
        return None

    noop()
    assert event_mod.Event.finished is False


def test_check_decorator_can_be_used_on_bound_method(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    class Worker:
        def __init__(self):
            self.calls = 0

        @event_mod.Event.check
        def run(self):
            self.calls += 1
            return self.calls

    w = Worker()
    assert w.run() == 1
    assert w.calls == 1


def test_check_decorator_wraps_exception_and_propagates(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    @event_mod.Event.check
    def explode():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        explode()


def test_check_decorator_on_classmethod_like_function(monkeypatch):
    monkeypatch.setattr(event_mod.time, "sleep", lambda _: None)

    class Demo:
        value = 0

        @classmethod
        @event_mod.Event.check
        def inc(cls):
            cls.value += 1
            return cls.value

    assert Demo.inc() == 1
    assert Demo.value == 1
