import time

import pytest

from Tracker import event as event_module
from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_event_state():
    """Event.check replaces the `finished` property on the class with a plain
    attribute; restore the original property and the singleton's state."""
    original_prop = Event.__dict__["finished"]
    original_finish = status._finish
    yield
    Event.finished = original_prop
    status.__dict__.pop("finished", None)
    status._finish = original_finish


@pytest.fixture
def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(event_module.time, "sleep", lambda s: calls.append(s))
    return calls


def test_new_event_not_finished():
    assert Event().finished is False


def test_setting_true_marks_finished():
    e = Event()
    e.finished = True
    assert e.finished is True
    assert e._finish is True


def test_setting_false_on_fresh_event_stays_false():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_setting_false_does_not_reset_finished():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


@pytest.mark.parametrize("value", [1, "yes", None, [1], 0])
def test_non_true_values_are_ignored(value):
    e = Event()
    e.finished = value
    assert e.finished is False


def test_instances_are_independent():
    a, b = Event(), Event()
    a.finished = True
    assert a.finished is True
    assert b.finished is False


def test_module_status_is_event_instance():
    assert isinstance(status, Event)


def test_status_starts_unfinished_and_can_be_set():
    status._finish = False
    assert status.finished is False
    status.finished = True
    assert status.finished is True


def test_check_returns_function_result(sleep_calls):
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


def test_check_sleeps_for_a_tenth_of_a_second(sleep_calls):
    @Event.check
    def func():
        pass

    func()
    assert sleep_calls == [0.1]


def test_check_sets_finished_true_during_sleep(monkeypatch):
    observed = []
    monkeypatch.setattr(
        event_module.time, "sleep", lambda s: observed.append(Event.finished)
    )

    @Event.check
    def func():
        pass

    func()
    assert observed == [True]


def test_check_resets_finished_to_false_afterwards(sleep_calls):
    @Event.check
    def func():
        pass

    func()
    assert Event.finished is False
    assert Event().finished is False
    assert status.finished is False


def test_check_does_not_signal_before_function_runs(sleep_calls):
    seen = []

    @Event.check
    def func():
        seen.append(Event.__dict__["finished"])

    func()
    assert isinstance(seen[0], property)


def test_check_exception_propagates_without_signal(sleep_calls):
    @Event.check
    def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        boom()
    assert sleep_calls == []
    assert isinstance(Event.__dict__["finished"], property)


def test_check_can_be_called_multiple_times(sleep_calls):
    @Event.check
    def func(x):
        return x * 2

    assert func(1) == 2
    assert func(2) == 4
    assert sleep_calls == [0.1, 0.1]


def test_check_on_subclass_sets_attribute_on_subclass_only(sleep_calls):
    class Sub(Event):
        pass

    @Sub.check
    def func():
        return "ok"

    assert func() == "ok"
    assert Sub.__dict__["finished"] is False
    assert isinstance(Event.__dict__["finished"], property)
