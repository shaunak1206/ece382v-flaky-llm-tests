import inspect

import pytest

from Tracker import event as event_module
from Tracker.event import Event, status


@pytest.fixture
def restore_event_class():
    """Event.check replaces the `finished` property on the class with a plain
    attribute, so restore the original property after each test using it."""
    original = Event.__dict__["finished"]
    yield
    setattr(Event, "finished", original)


@pytest.fixture
def no_sleep(monkeypatch):
    calls = []
    monkeypatch.setattr(event_module.time, "sleep", lambda s: calls.append(s))
    return calls


def test_new_event_not_finished():
    assert Event().finished is False


def test_new_event_internal_flag_false():
    assert Event()._finish is False


def test_set_finished_true():
    e = Event()
    e.finished = True
    assert e.finished is True
    assert e._finish is True


def test_set_false_on_fresh_event_stays_false():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_set_false_after_true_does_not_reset():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


def test_truthy_non_true_values_ignored():
    e = Event()
    e.finished = 1
    assert e.finished is False
    e.finished = "yes"
    assert e.finished is False


def test_set_true_repeatedly_is_idempotent():
    e = Event()
    e.finished = True
    e.finished = True
    assert e.finished is True


def test_instances_are_independent():
    a, b = Event(), Event()
    a.finished = True
    assert a.finished is True
    assert b.finished is False


def test_finished_is_property_on_class():
    assert isinstance(Event.__dict__["finished"], property)


def test_status_is_event_instance(monkeypatch):
    monkeypatch.setattr(status, "_finish", False)
    assert isinstance(status, Event)
    assert status.finished is False


def test_status_setter_works(monkeypatch):
    monkeypatch.setattr(status, "_finish", False)
    status.finished = True
    assert status.finished is True


def test_check_returns_function_result(no_sleep, restore_event_class):
    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_passes_args_and_kwargs(no_sleep, restore_event_class):
    @Event.check
    def f(*args, **kwargs):
        return args, kwargs

    assert f(1, 2, x=3) == ((1, 2), {"x": 3})


def test_check_preserves_metadata(no_sleep, restore_event_class):
    @Event.check
    def documented():
        """my doc"""
        return None

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "my doc"
    assert callable(documented)
    assert inspect.isfunction(documented)


def test_check_sleeps_once_for_tenth_of_second(no_sleep, restore_event_class):
    @Event.check
    def f():
        return 1

    f()
    assert no_sleep == [0.1]


def test_check_sets_class_finished_true_during_sleep(monkeypatch, restore_event_class):
    seen = []

    def fake_sleep(seconds):
        seen.append(Event.finished)

    monkeypatch.setattr(event_module.time, "sleep", fake_sleep)

    @Event.check
    def f():
        return "ok"

    assert f() == "ok"
    assert seen == [True]
    # reset to False afterwards
    assert Event.finished is False


def test_check_does_not_touch_status_instance_flag(monkeypatch, no_sleep, restore_event_class):
    monkeypatch.setattr(status, "_finish", False)

    @Event.check
    def f():
        return 1

    f()
    assert status._finish is False


def test_check_propagates_exception_without_sleeping(no_sleep, restore_event_class):
    @Event.check
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        boom()
    assert no_sleep == []
    # property on the class is untouched when the function raised
    assert isinstance(Event.__dict__["finished"], property)
