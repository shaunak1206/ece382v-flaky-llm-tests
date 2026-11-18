import pytest

from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_event_finished_descriptor():
    """
    Event.check() assigns to cls.finished, which replaces the property on the class.
    Restore the original descriptor after each test to keep tests isolated.
    """
    original = Event.__dict__["finished"]
    yield
    # Restore the property descriptor on the class.
    Event.finished = original


def test_status_is_singleton_event_instance():
    assert isinstance(status, Event)


def test_status_starts_not_finished():
    assert status._finish is False
    assert status.finished is False


def test_new_event_instances_start_not_finished():
    e1 = Event()
    e2 = Event()
    assert e1.finished is False
    assert e2.finished is False
    assert e1 is not e2


def test_finished_property_reflects_internal_state():
    e = Event()
    e._finish = True
    assert e.finished is True
    e._finish = False
    assert e.finished is False


def test_finished_setter_sets_true():
    e = Event()
    e.finished = True
    assert e._finish is True
    assert e.finished is True


def test_finished_setter_ignores_false_when_currently_false():
    e = Event()
    e.finished = False
    assert e._finish is False
    assert e.finished is False


def test_finished_setter_does_not_clear_true_value():
    e = Event()
    e.finished = True
    e.finished = False
    assert e._finish is True
    assert e.finished is True


def test_check_decorator_preserves_function_metadata():
    def sample_function():
        """sample docstring"""
        return 123

    wrapped = Event.check(sample_function)
    assert wrapped.__name__ == "sample_function"
    assert wrapped.__doc__ == "sample docstring"


def test_check_decorator_returns_original_result(monkeypatch):
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr("Tracker.event.time.sleep", fake_sleep)

    @Event.check
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert calls == [0.1]


def test_check_decorator_passes_args_and_kwargs(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda s: None)

    @Event.check
    def combine(a, b=0, *, c=0):
        return a + b + c

    assert combine(1, b=2, c=3) == 6


def test_check_decorator_sets_class_finished_attribute_temporarily(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda s: None)

    @Event.check
    def noop():
        return "done"

    assert "finished" in Event.__dict__
    assert Event.__dict__["finished"] is Event.__dict__["finished"]
    assert noop() == "done"
    assert Event.finished is False
    assert status.finished is False


def test_check_decorator_does_not_change_return_value_when_sleep_patched(monkeypatch):
    monkeypatch.setattr("Tracker.event.time.sleep", lambda s: None)

    @Event.check
    def identity(value):
        return value

    assert identity({"key": "value"}) == {"key": "value"}


def test_check_decorator_can_be_used_multiple_times(monkeypatch):
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("Tracker.event.time.sleep", fake_sleep)

    @Event.check
    def first():
        return "first"

    @Event.check
    def second():
        return "second"

    assert first() == "first"
    assert second() == "second"
    assert sleep_calls == [0.1, 0.1]
    assert Event.finished is False
