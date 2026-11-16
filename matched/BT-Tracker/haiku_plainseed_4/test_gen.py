import time

import pytest

from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_finished_property():
    # Event.check() assigns to cls.finished, which replaces the class-level
    # property with a plain value. Restore the original property after each
    # test so one test cannot affect another.
    original = Event.__dict__["finished"]
    yield
    Event.finished = original


@pytest.fixture
def fake_sleep(monkeypatch):
    calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: calls.append(seconds))
    return calls


# ---------- Event.finished property ----------

def test_new_event_is_not_finished():
    assert Event().finished is False


def test_setting_finished_true_marks_event_finished():
    e = Event()
    e.finished = True
    assert e.finished is True


def test_setting_finished_false_after_true_is_ignored():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


@pytest.mark.parametrize("value", [1, 0, "yes", None, []])
def test_setting_non_bool_values_is_ignored(value):
    e = Event()
    e.finished = value
    assert e.finished is False


def test_instances_have_independent_status():
    first = Event()
    second = Event()
    first.finished = True
    assert first.finished is True
    assert second.finished is False


def test_private_flag_defaults_to_false():
    assert Event()._finish is False


# ---------- module-level singleton `status` ----------

def test_status_is_event_instance():
    assert isinstance(status, Event)


def test_status_starts_not_finished():
    assert status.finished is False


def test_status_is_module_singleton():
    import Tracker.event as event_module
    assert event_module.status is status


# ---------- Event.check decorator ----------

def test_check_returns_wrapped_result():
    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_returns_new_callable_not_original(fake_sleep):
    def task():
        return 1

    wrapped = Event.check(task)
    assert wrapped is not task
    assert callable(wrapped)


def test_check_preserves_function_metadata(fake_sleep):
    @Event.check
    def my_task():
        """Does a task."""
        return None

    assert my_task.__name__ == "my_task"
    assert my_task.__doc__ == "Does a task."


def test_check_forwards_args_and_kwargs(fake_sleep):
    @Event.check
    def combine(a, b, *rest, scale=1, **extra):
        return (a, b, rest, scale, extra)

    result = combine(1, 2, 3, 4, scale=10, color="red")
    assert result == (1, 2, (3, 4), 10, {"color": "red"})


def test_check_calls_sleep_with_point_one(fake_sleep):
    wrapped = Event.check(lambda: None)
    wrapped()
    assert fake_sleep == [0.1]


def test_check_marks_event_finished_during_sleep(monkeypatch):
    observed = []

    def fake(seconds):
        observed.append(Event.finished)

    monkeypatch.setattr(time, "sleep", fake)
    wrapped = Event.check(lambda: "done")
    wrapped()
    assert observed == [True]


def test_check_runs_function_before_sleep(monkeypatch):
    order = []

    def fake(seconds):
        order.append("sleep")

    monkeypatch.setattr(time, "sleep", fake)

    @Event.check
    def work():
        order.append("func")

    work()
    assert order == ["func", "sleep"]


def test_check_propagates_exceptions_and_skips_sleep(fake_sleep):
    @Event.check
    def explode():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        explode()
    assert fake_sleep == []


def test_check_wrapper_can_be_called_repeatedly(fake_sleep):
    counter = {"n": 0}

    @Event.check
    def bump():
        counter["n"] += 1
        return counter["n"]

    assert bump() == 1
    assert bump() == 2
    assert bump() == 3
    assert fake_sleep == [0.1, 0.1, 0.1]


@pytest.mark.xfail(
    strict=True,
    reason="Event.check assigns cls.finished, which replaces the class-level "
           "'finished' property with a plain value instead of setting the flag.",
)
def test_check_keeps_finished_property_intact(fake_sleep):
    Event.check(lambda: None)()
    assert isinstance(Event.__dict__["finished"], property)
