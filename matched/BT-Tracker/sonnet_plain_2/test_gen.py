import time

import pytest

from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_event_state():
    """Event.check replaces the `finished` property on the class, so restore it."""
    original = Event.__dict__["finished"]
    original_status_finish = status._finish
    yield
    setattr(Event, "finished", original)
    status._finish = original_status_finish


@pytest.fixture
def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(time, "sleep", lambda s: calls.append(s))
    return calls


def test_new_event_not_finished():
    assert Event().finished is False


def test_internal_flag_initially_false():
    assert Event()._finish is False


def test_setting_true_marks_finished():
    e = Event()
    e.finished = True
    assert e.finished is True


def test_setting_false_on_fresh_event_stays_false():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_setting_false_does_not_reset_once_true():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


def test_truthy_non_true_values_are_ignored():
    e = Event()
    e.finished = 1
    assert e.finished is False
    e.finished = "yes"
    assert e.finished is False


def test_instances_are_independent():
    a, b = Event(), Event()
    a.finished = True
    assert a.finished is True
    assert b.finished is False


def test_module_level_status_is_event_instance():
    assert isinstance(status, Event)


def test_status_finished_can_be_set():
    status._finish = False
    status.finished = True
    assert status.finished is True


def test_check_returns_wrapped_result(sleep_calls):
    @Event.check
    def func():
        return 42

    assert func() == 42


def test_check_passes_args_and_kwargs(sleep_calls):
    @Event.check
    def func(a, b, c=0):
        return (a, b, c)

    assert func(1, 2, c=3) == (1, 2, 3)


def test_check_preserves_function_metadata(sleep_calls):
    @Event.check
    def my_function():
        """my doc"""

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "my doc"


def test_check_sleeps_a_tenth_of_a_second(sleep_calls):
    @Event.check
    def func():
        return None

    func()
    assert sleep_calls == [0.1]


def test_check_sets_class_finished_true_during_sleep(monkeypatch):
    seen = []
    monkeypatch.setattr(time, "sleep", lambda s: seen.append(Event.__dict__["finished"]))

    @Event.check
    def func():
        return None

    func()
    assert seen == [True]


def test_check_leaves_class_finished_false_afterwards(sleep_calls):
    @Event.check
    def func():
        return None

    func()
    assert Event.__dict__["finished"] is False
    assert Event().finished is False


def test_check_calls_function_exactly_once(sleep_calls):
    calls = []

    @Event.check
    def func():
        calls.append(1)

    func()
    assert calls == [1]


def test_check_propagates_exception_without_sleeping(sleep_calls):
    @Event.check
    def func():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        func()
    assert sleep_calls == []
    assert isinstance(Event.__dict__["finished"], property)


def test_check_can_wrap_multiple_functions(sleep_calls):
    @Event.check
    def one():
        return 1

    @Event.check
    def two():
        return 2

    assert (one(), two()) == (1, 2)
    assert sleep_calls == [0.1, 0.1]
