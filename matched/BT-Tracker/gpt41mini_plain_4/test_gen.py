import pytest
import time
from Tracker import event


def test_event_initial_finished_is_false():
    e = event.Event()
    assert e.finished is False


def test_event_finished_setter_only_allows_true():
    e = event.Event()
    e.finished = True
    assert e.finished is True
    e.finished = False
    # Should ignore setting to False; still True
    assert e.finished is True


def test_event_finished_getter_reflects_internal_state():
    e = event.Event()
    # Directly set internal to True
    e._finish = True
    assert e.finished is True
    # Directly set internal to False
    e._finish = False
    assert e.finished is False


def test_check_decorator_sets_finished_true_then_false_after_delay(monkeypatch):
    call_order = []

    class Dummy:
        def __init__(self):
            self.finished_states = []

        @event.Event.check
        def tested_func(self):
            call_order.append("func_called")
            # Capture finished immediately after call
            self.finished_states.append(event.status.finished)
            return 123

    dummy = Dummy()

    # Patch time.sleep to capture when it's called without actual delay
    def fake_sleep(seconds):
        call_order.append(f"slept_{seconds}")

    monkeypatch.setattr(time, "sleep", fake_sleep)

    result = dummy.tested_func()
    # After function call, wrapper sets finished True,
    # sleeps, then sets finished False.

    # Because status.finished is a property on Event instance 'status',
    # but check decorator modifies class attribute finished which should be
    # investigated:
    # Actually, decorator refers to cls.finished,
    # but cls is Event and Event has no class attribute 'finished'.
    # This sets a class attribute instead, unrelated to instance _finish.
    # So status.finished (instance property) is independent.
    
    # The test is therefore to check that class attribute Event.finished toggled by decorator.
    # So we can access that:
    assert result == 123
    assert call_order == ["func_called", "slept_0.1"]
    # cls.finished is a class attribute set by decorator; should be False at end:
    assert event.Event.finished is False


def test_check_decorator_returns_original_func_result():
    @event.Event.check
    def foo(x):
        return x * 2

    val = foo(4)
    assert val == 8


def test_check_decorator_preserves_function_metadata():
    @event.Event.check
    def bar():
        """test docstring"""
        pass

    assert bar.__name__ == "bar"
    assert bar.__doc__ == "test docstring"


def test_event_finished_setter_does_not_set_when_false():
    e = event.Event()
    # Set True to change from False -> True
    e.finished = True
    assert e.finished is True
    # Try to set False, should still be True
    e.finished = False
    assert e.finished is True


def test_status_is_instance_of_event():
    assert isinstance(event.status, event.Event)


def test_finished_property_is_read_only_except_with_true_setter():
    e = event.Event()
    # try setting with some other truthy but not True values
    e.finished = 1  # int, truthy but not True
    # Should not set _finish because setter checks specifically for True
    assert e.finished is False
    e.finished = True
    assert e.finished is True


def test_check_decorator_can_wrap_function_with_args_and_kwargs():
    calls = []

    @event.Event.check
    def func(a, b=2):
        calls.append((a, b))
        return a + b

    result = func(3, b=5)
    assert result == 8
    assert calls == [(3, 5)]


def test_done_state_different_per_instance_and_class_attribute_unaffected():
    e1 = event.Event()
    e2 = event.Event()

    # Initially
    assert e1.finished is False
    assert e2.finished is False

    e1.finished = True
    assert e1.finished is True
    assert e2.finished is False

    # Decorator sets Event.finished class attr, so check with:
    event.Event.finished = False
    assert not hasattr(e1, "finished") or e1.finished is True


def test_check_decorator_changes_class_finished_attribute_only():
    # Initially class has no attribute 'finished'
    assert not hasattr(event.Event, "finished") or event.Event.finished is False

    @event.Event.check
    def f():
        return 42

    f()
    # After call, Event.finished toggled True, then False again
    assert event.Event.finished is False


def test_check_decorator_change_does_not_affect_instance_finished_property():
    e = event.Event()
    # Initially instance finished is False
    assert e.finished is False

    @event.Event.check
    def dummy():
        return True

    dummy()
    # The decorator changes class attribute finished on Event class,
    # instance property is independent
    assert e.finished is False


def test_finished_property_with_nonbool_input_has_no_effect():
    e = event.Event()
    for val in [None, 0, "", [], {}, 1, "True"]:
        e.finished = val
        # only True (bool) sets finished to True
        assert e.finished is False

    e.finished = True
    assert e.finished is True


def test_check_decorator_returns_value_and_resets_class_finished():
    state_changes = []

    original_sleep = time.sleep

    def dummy_sleep(seconds):
        state_changes.append(("before_sleep", event.Event.finished))
        original_sleep(seconds)
        state_changes.append(("after_sleep", event.Event.finished))

    # Patch sleep to dummy_sleep
    import builtins
    import sys

    # monkeypatch doesn't work as nicely here; do manual patch
    time.sleep = dummy_sleep

    @event.Event.check
    def f():
        return "done"

    ret = f()
    time.sleep = original_sleep

    assert ret == "done"
    # Before sleep, finished = True; after sleep, finished = False
    assert ("before_sleep", True) in state_changes
    assert ("after_sleep", False) in state_changes


def test_finished_property_returns_exact_bool_value():
    e = event.Event()
    assert isinstance(e.finished, bool)

    e.finished = True
    # Should be exactly True type bool, not just truthy
    assert e.finished is True


def test_event_finished_private_attribute_encapsulation():
    e = event.Event()
    # _finish should exist and be boolean
    assert hasattr(e, "_finish")
    assert isinstance(e._finish, bool)


def test_check_decorator_can_decorate_function_with_side_effects():
    side_effects = []

    @event.Event.check
    def f(x):
        side_effects.append(x)

    f(7)
    assert side_effects == [7]

