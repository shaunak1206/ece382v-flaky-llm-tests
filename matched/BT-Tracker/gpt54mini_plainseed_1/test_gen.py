import types

import pytest

from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_event_class():
    original_finished = Event.__dict__["finished"]
    original_status_finish = status._finish
    yield
    # Restore class-level descriptor possibly overwritten by Event.check wrapper calls
    Event.finished = original_finished
    status._finish = original_status_finish


def test_default_status_is_not_finished():
    assert status.finished is False


def test_new_event_defaults_to_not_finished():
    event = Event()
    assert event.finished is False


def test_finished_setter_accepts_true():
    event = Event()
    event.finished = True
    assert event.finished is True


def test_finished_setter_ignores_false_when_already_false():
    event = Event()
    event.finished = False
    assert event.finished is False


def test_finished_setter_does_not_reset_true_to_false():
    event = Event()
    event.finished = True
    event.finished = False
    assert event.finished is True


def test_status_singleton_is_instance_of_event():
    assert isinstance(status, Event)


def test_status_and_new_event_are_different_instances():
    assert status is not Event()


def test_check_decorator_returns_original_result(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda _: None)

    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_decorator_preserves_function_metadata(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda _: None)

    @Event.check
    def sample_function():
        """A simple docstring."""
        return "ok"

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "A simple docstring."


def test_check_decorator_calls_sleep_with_point_one(monkeypatch):
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr("Tracker.event.time.sleep", fake_sleep)

    @Event.check
    def noop():
        return "done"

    assert noop() == "done"
    assert calls == [0.1]


def test_check_decorator_overwrites_class_finished_attribute(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda _: None)

    @Event.check
    def noop():
        return None

    assert isinstance(Event.__dict__["finished"], property)
    noop()
    assert Event.finished is False
    assert Event.__dict__["finished"] is False


def test_check_decorator_leaves_instance_finished_accessible(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda _: None)

    @Event.check
    def noop():
        return None

    noop()
    assert status.finished is False


def test_check_decorator_can_wrap_bound_method(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda _: None)

    class Worker:
        def __init__(self):
            self.calls = 0

        @Event.check
        def run(self, value):
            self.calls += value
            return self.calls

    worker = Worker()
    assert worker.run(4) == 4
    assert worker.calls == 4


def test_event_finished_is_a_property_on_class_before_wrapping():
    assert isinstance(Event.__dict__["finished"], property)
    assert isinstance(Event.__dict__["finished"], property)
