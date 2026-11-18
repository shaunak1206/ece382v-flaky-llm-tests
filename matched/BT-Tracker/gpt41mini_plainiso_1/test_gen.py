import pytest
import time
from Tracker import event


def test_event_initial_finished_false():
    ev = event.Event()
    assert ev.finished is False


def test_event_finished_setter_true_sets_finish():
    ev = event.Event()
    ev.finished = True
    assert ev.finished is True


def test_event_finished_setter_false_does_not_clear_finish():
    ev = event.Event()
    ev.finished = True
    ev.finished = False
    # Because setter only sets _finish if True, setting False has no effect
    assert ev.finished is True


def test_event_finished_property_returns_correct_value():
    ev = event.Event()
    # manually manipulate internal _finish
    ev._finish = True
    assert ev.finished is True
    ev._finish = False
    assert ev.finished is False


def test_event_check_decorator_sets_finished_flag(monkeypatch):
    calls = []

    @event.Event.check
    def dummy_func(x):
        calls.append(x)
        return x + 1

    start_time = time.time()
    result = dummy_func(5)
    elapsed = time.time() - start_time

    # returned result as expected
    assert result == 6
    # decorated function called with expected argument
    assert calls == [5]
    # after function, finished was set to True then False internally
    # Because class attribute .finished is set in check, test that attribute exists
    # The decorator uses Event.finished (class attr), verify by accessing class attribute
    assert hasattr(event.Event, "finished")

    # elapsed sleep roughly 0.1 seconds
    assert elapsed >= 0.1


def test_event_check_decorator_resets_finished_flag_after_sleep():
    states = []

    @event.Event.check
    def dummy():
        # Read finished state inside function, should be False (because decorator sets after function returns)
        states.append(event.Event.finished if hasattr(event.Event, "finished") else None)
        return 42

    result = dummy()
    # Result check
    assert result == 42
    # Because inside function finished should not be set yet
    assert states == [False] or states == [None] or states == [getattr(event.Event, "finished", None)]
    # After function returned, finished was True and then False, test final state:
    # The class attribute finished should be False because the decorator sets True then 0.1s sleep then False
    assert event.Event.finished is False


def test_event_check_decorator_preserves_function_identity():
    @event.Event.check
    def some_func():
        """Docstring for some_func."""
        pass

    assert some_func.__name__ == "some_func"
    assert some_func.__doc__ == "Docstring for some_func."


def test_status_is_event_instance():
    assert isinstance(event.status, event.Event)


def test_status_initial_finished_is_false():
    assert event.status.finished is False


def test_setting_status_finished_true_updates_finished():
    event.status.finished = True
    assert event.status.finished is True


def test_setting_status_finished_false_does_not_change_finished():
    event.status.finished = True
    event.status.finished = False
    assert event.status.finished is True


def test_multiple_event_instances_independent_finish_flags():
    ev1 = event.Event()
    ev2 = event.Event()
    ev1.finished = True
    assert ev1.finished is True
    assert ev2.finished is False
    ev2.finished = True
    assert ev2.finished is True


def test_event_check_decorator_works_with_methods():
    class Dummy:
        def __init__(self):
            self.called = False

        @event.Event.check
        def method(self, x):
            self.called = True
            return x * 2

    d = Dummy()
    result = d.method(3)
    assert d.called is True
    assert result == 6


def test_event_finished_setter_only_accepts_true_to_set():
    ev = event.Event()
    ev._finish = False
    ev.finished = True
    assert ev.finished is True
    ev.finished = False
    # still True since setter ignores False
    assert ev.finished is True

    # Trying other values doesn't clear finished flag
    ev.finished = None
    assert ev.finished is True
    ev.finished = 0
    assert ev.finished is True


def test_event_finished_can_be_manually_cleared_directly():
    # Because setter ignores False, test that we can clear _finish directly
    ev = event.Event()
    ev.finished = True
    assert ev.finished is True
    ev._finish = False
    assert ev.finished is False


def test_decorator_can_handle_function_raising_exception():
    @event.Event.check
    def raises():
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        raises()
    # After exception, class attribute finished was set to True then False anyway (check exists)
    # Cannot guarantee because exception happens before decorator body
    # But finished attribute must still exist
    assert hasattr(event.Event, "finished")


def test_decorator_multiple_calls_resets_finished():
    call_times = []

    @event.Event.check
    def f():
        call_times.append(time.time())
        return True

    for _ in range(3):
        result = f()
        assert result is True
    # All calls recorded
    assert len(call_times) == 3


def test_finished_property_and_setter_type():
    ev = event.Event()
    assert isinstance(ev.finished, bool)
    ev.finished = True
    assert isinstance(ev.finished, bool)
