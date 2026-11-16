import pytest

from Tracker import event as event_module
from Tracker.event import Event, status

_ORIGINAL_FINISHED = Event.__dict__["finished"]


@pytest.fixture(autouse=True)
def restore_event_class(monkeypatch):
    # Event.check() assigns to cls.finished, which replaces the property on the
    # class. Restore the original property around every test so tests stay isolated.
    monkeypatch.setattr(event_module.time, "sleep", lambda _seconds: None)
    setattr(Event, "finished", _ORIGINAL_FINISHED)
    yield
    setattr(Event, "finished", _ORIGINAL_FINISHED)


def test_new_event_is_not_finished():
    assert Event().finished is False


def test_setting_true_marks_event_finished():
    e = Event()
    e.finished = True
    assert e.finished is True


def test_setting_false_on_new_event_keeps_it_unfinished():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_setting_false_after_true_is_ignored():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


def test_setting_truthy_non_bool_is_ignored():
    e = Event()
    e.finished = 1
    assert e.finished is False


def test_setting_none_is_ignored():
    e = Event()
    e.finished = None
    assert e.finished is False


def test_setting_string_is_ignored():
    e = Event()
    e.finished = "yes"
    assert e.finished is False


def test_instances_have_independent_state():
    a = Event()
    b = Event()
    a.finished = True
    assert a.finished is True
    assert b.finished is False


def test_module_level_status_is_event_instance():
    assert isinstance(status, Event)


def test_check_returns_wrapped_function_result():
    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_check_passes_positional_and_keyword_arguments():
    received = {}

    @Event.check
    def capture(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return None

    capture(1, "two", key="value")
    assert received["args"] == (1, "two")
    assert received["kwargs"] == {"key": "value"}


def test_check_preserves_function_name_and_doc():
    @Event.check
    def documented():
        """Original docstring."""
        return None

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Original docstring."


def test_check_propagates_exceptions():
    @Event.check
    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing()


def test_check_sleeps_once_for_point_one_seconds(monkeypatch):
    calls = []
    monkeypatch.setattr(event_module.time, "sleep", calls.append)

    @Event.check
    def work():
        return None

    work()
    assert calls == [0.1]


def test_check_runs_function_before_sleeping(monkeypatch):
    order = []
    monkeypatch.setattr(event_module.time, "sleep", lambda _s: order.append("sleep"))

    @Event.check
    def work():
        order.append("func")

    work()
    assert order == ["func", "sleep"]


def test_check_leaves_new_event_unfinished_afterwards():
    @Event.check
    def work():
        return None

    work()
    assert Event().finished is False


@pytest.mark.xfail(
    strict=True,
    reason="check() assigns cls.finished, which replaces the property on the class",
)
def test_check_does_not_replace_finished_property():
    @Event.check
    def work():
        return None

    work()
    assert isinstance(Event.__dict__["finished"], property)


@pytest.mark.xfail(
    strict=True,
    reason="after check() replaces the property, the setter no longer ignores False",
)
def test_setter_still_ignores_false_after_check_call():
    @Event.check
    def work():
        return None

    work()
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True
