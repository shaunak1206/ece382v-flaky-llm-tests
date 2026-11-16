import time

import pytest

from Tracker import event as event_module
from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_event_state():
    """Event.check overwrites the `finished` property on the class, so restore it."""
    original = Event.__dict__["finished"]
    status._finish = False
    yield
    setattr(Event, "finished", original)
    status._finish = False


@pytest.fixture
def fake_sleep(monkeypatch):
    calls = []

    def _sleep(seconds):
        calls.append((seconds, Event.__dict__.get("finished")))

    monkeypatch.setattr(time, "sleep", _sleep)
    return calls


def test_new_event_not_finished():
    assert Event().finished is False


def test_internal_flag_initial_value():
    assert Event()._finish is False


def test_module_status_is_event_instance():
    assert isinstance(status, Event)


def test_module_status_initially_not_finished():
    assert status.finished is False


def test_status_is_same_object_on_reimport():
    from Tracker.event import status as status_again
    assert status_again is status
    assert event_module.status is status


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


def test_setting_truthy_non_true_value_is_ignored():
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


def test_check_returns_wrapped_result(fake_sleep):
    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_passes_args_and_kwargs(fake_sleep):
    @Event.check
    def func(*args, **kwargs):
        return args, kwargs

    assert func(1, 2, x=3) == ((1, 2), {"x": 3})


def test_check_preserves_function_metadata(fake_sleep):
    @Event.check
    def documented():
        """my doc"""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "my doc"


def test_check_sleeps_for_a_tenth_of_a_second(fake_sleep):
    @Event.check
    def noop():
        return None

    noop()
    assert len(fake_sleep) == 1
    assert fake_sleep[0][0] == 0.1


def test_check_sets_class_finished_true_during_sleep(fake_sleep):
    @Event.check
    def noop():
        return None

    noop()
    assert fake_sleep[0][1] is True


def test_check_sets_class_finished_false_afterwards(fake_sleep):
    @Event.check
    def noop():
        return None

    noop()
    assert Event.finished is False


def test_check_propagates_exception_without_sleeping(fake_sleep):
    @Event.check
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()
    assert fake_sleep == []
    assert isinstance(Event.__dict__["finished"], property)


def test_check_called_multiple_times(fake_sleep):
    @Event.check
    def ident(x):
        return x

    assert ident(1) == 1
    assert ident(2) == 2
    assert len(fake_sleep) == 2
