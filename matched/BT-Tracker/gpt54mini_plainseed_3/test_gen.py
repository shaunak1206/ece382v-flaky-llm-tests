import importlib

import pytest


@pytest.fixture
def event_module():
    module = importlib.import_module("Tracker.event")
    return importlib.reload(module)


def test_status_is_singleton_event_instance(event_module):
    assert isinstance(event_module.status, event_module.Event)
    assert event_module.status is event_module.status


def test_status_initial_finished_is_false(event_module):
    assert event_module.status.finished is False


def test_finished_setter_sets_true(event_module):
    event_module.status.finished = True
    assert event_module.status.finished is True


def test_finished_setter_ignores_false_assignment(event_module):
    event_module.status.finished = False
    assert event_module.status.finished is False


def test_finished_setter_does_not_reset_after_being_true(event_module):
    event_module.status.finished = True
    event_module.status.finished = False
    assert event_module.status.finished is True


def test_finished_setter_ignores_non_true_truthy_values(event_module):
    event_module.status.finished = 1
    assert event_module.status.finished is False


def test_check_decorator_returns_original_result(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)

    @event_module.Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_decorator_preserves_function_metadata(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)

    @event_module.Event.check
    def sample_function():
        """Example docstring."""
        return "ok"

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "Example docstring."


def test_check_decorator_calls_sleep_with_point_one(event_module, monkeypatch):
    calls = []

    def fake_sleep(value):
        calls.append(value)

    monkeypatch.setattr(event_module.time, "sleep", fake_sleep)

    @event_module.Event.check
    def do_nothing():
        return None

    do_nothing()
    assert calls == [0.1]


def test_check_decorator_executes_function_and_returns_value(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)
    side_effects = []

    @event_module.Event.check
    def target(value):
        side_effects.append(value)
        return value * 2

    result = target(7)
    assert result == 14
    assert side_effects == [7]


def test_check_decorator_toggles_finished_state(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)

    @event_module.Event.check
    def target():
        return "done"

    assert event_module.status.finished is False
    assert target() == "done"
    assert event_module.status.finished is False


def test_check_decorator_can_be_called_multiple_times(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)
    calls = []

    @event_module.Event.check
    def target(x):
        calls.append(x)
        return x + 1

    assert target(1) == 2
    assert target(5) == 6
    assert calls == [1, 5]


def test_check_decorator_works_with_keyword_arguments(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)

    @event_module.Event.check
    def combine(*, left, right):
        return f"{left}-{right}"

    assert combine(left="a", right="b") == "a-b"


def test_check_decorator_sets_class_attribute_finished(event_module, monkeypatch):
    monkeypatch.setattr(event_module.time, "sleep", lambda *_: None)

    @event_module.Event.check
    def target():
        return "x"

    target()
    assert hasattr(event_module.Event, "finished")
    assert event_module.Event.finished is False
    assert event_module.status.finished is False
