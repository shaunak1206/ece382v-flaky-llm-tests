import importlib.util
import sys
import time
from pathlib import Path

import pytest


def _find_project_root():
    """Walk upward from this file until the directory containing the Tracker package is found."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "Tracker" / "event.py").is_file():
            return candidate
    return here.parent


_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Tracker.event import Event, status  # noqa: E402

# Captured at import time, before any test can run Event.check.
# Event.check assigns a plain class attribute over the property, so the
# original descriptor must be restored before each test to keep tests isolated.
_ORIGINAL_FINISHED = Event.__dict__["finished"]


@pytest.fixture(autouse=True)
def pristine_event_class():
    Event.finished = _ORIGINAL_FINISHED
    yield
    Event.finished = _ORIGINAL_FINISHED


@pytest.fixture
def sleep_calls(monkeypatch):
    """Replace time.sleep with a recorder that also captures the flag value at call time."""
    calls = []

    def fake_sleep(seconds):
        calls.append((seconds, Event.finished))

    monkeypatch.setattr(time, "sleep", fake_sleep)
    return calls


def test_new_event_is_not_finished():
    assert Event().finished is False


def test_new_event_initializes_backing_flag_to_false():
    assert Event()._finish is False


def test_setting_finished_true_marks_event_finished():
    e = Event()
    e.finished = True
    assert e.finished is True
    assert e._finish is True


def test_setting_finished_false_on_fresh_event_keeps_false():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_finished_cannot_be_reset_to_false_once_true():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


def test_setting_finished_true_twice_stays_true():
    e = Event()
    e.finished = True
    e.finished = True
    assert e.finished is True


@pytest.mark.parametrize("value", [1, 0, "yes", "", None, [True]])
def test_non_bool_true_values_are_ignored(value):
    e = Event()
    e.finished = value
    assert e.finished is False


def test_truthy_value_does_not_override_already_true_state():
    e = Event()
    e.finished = True
    e.finished = 0
    assert e.finished is True


def test_instances_have_independent_state():
    a = Event()
    b = Event()
    a.finished = True
    assert a.finished is True
    assert b.finished is False


def test_finished_is_a_property_on_the_class():
    assert isinstance(Event.__dict__["finished"], property)


def test_module_level_status_is_an_event_instance():
    assert isinstance(status, Event)


def test_module_level_status_starts_not_finished():
    assert status.finished is False


def test_check_returns_wrapped_function_result(monkeypatch, sleep_calls):
    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_passes_args_and_kwargs_through(sleep_calls):
    received = {}

    @Event.check
    def capture(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return None

    capture(1, "x", key="value")
    assert received["args"] == (1, "x")
    assert received["kwargs"] == {"key": "value"}


def test_check_preserves_wrapped_function_metadata(sleep_calls):
    @Event.check
    def documented():
        """Some docstring."""
        return 42

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Some docstring."


def test_check_sets_flag_true_during_sleep(sleep_calls):
    @Event.check
    def work():
        return "done"

    work()
    assert len(sleep_calls) == 1
    _, flag_during_sleep = sleep_calls[0]
    assert flag_during_sleep is True


def test_check_sleeps_for_point_one_seconds(sleep_calls):
    @Event.check
    def work():
        return None

    work()
    assert sleep_calls[0][0] == 0.1


def test_check_clears_flag_after_call(sleep_calls):
    @Event.check
    def work():
        return None

    work()
    assert status.finished is False
    assert Event.finished is False


def test_check_propagates_exceptions_from_wrapped_function(sleep_calls):
    @Event.check
    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing()


def test_check_wraps_function_called_multiple_times(sleep_calls):
    counter = {"n": 0}

    @Event.check
    def increment():
        counter["n"] += 1
        return counter["n"]

    assert increment() == 1
    assert increment() == 2
    assert len(sleep_calls) == 2
