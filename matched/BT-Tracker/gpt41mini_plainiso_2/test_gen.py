import pytest
import time
from Tracker import event


def test_event_initial_finished_false():
    evt = event.Event()
    assert evt.finished is False


def test_event_finished_setter_allows_true():
    evt = event.Event()
    evt.finished = True
    assert evt.finished is True


def test_event_finished_setter_ignores_false():
    evt = event.Event()
    evt.finished = True
    evt.finished = False
    # Should remain True because setter ignores False
    assert evt.finished is True


def test_event_finished_setter_allows_true_then_ignores_false():
    evt = event.Event()
    evt.finished = True
    assert evt.finished is True
    evt.finished = False
    # Should remain True as False is ignored
    assert evt.finished is True


def test_status_singleton_event_instance():
    # status is an Event instance
    assert isinstance(event.status, event.Event)


def test_check_decorator_sets_class_finished_true_then_false(monkeypatch):
    # Patch time.sleep to avoid delay
    monkeypatch.setattr(time, "sleep", lambda x: None)

    calls = {}

    class Dummy:
        @event.Event.check
        def f(self, x):
            calls['called'] = x
            return x * 2

    d = Dummy()
    # before call, class finished attribute should not exist or be False
    assert not hasattr(event.Event, 'finished') or event.Event.finished is False
    ret = d.f(5)
    assert ret == 10
    # After decorated call wrapper finishes, the class attribute finished is reset to False,
    # but right after calling func it's True.
    # However, we cannot observe inside the sleep window.
    # So let's check that the calls dict updated correctly
    assert calls['called'] == 5
    # The class attribute finished should be False after the wrapper finishes
    assert event.Event.finished is False


def test_check_decorator_sets_class_finished_during_call(monkeypatch):
    sleep_calls = []

    def fake_sleep(t):
        sleep_calls.append(t)
        # while sleeping, check finished is True
        assert event.Event.finished is True

    monkeypatch.setattr(time, "sleep", fake_sleep)

    called_flag = {}

    class Dummy:
        @event.Event.check
        def f(self):
            called_flag['called'] = True
            return "done"

    d = Dummy()

    # Before call
    assert not hasattr(event.Event, 'finished') or event.Event.finished is False

    result = d.f()
    assert result == "done"
    assert called_flag['called'] is True
    # The fake_sleep recorded 0.1 second sleep
    assert sleep_calls == [0.1]
    # After call finished should be False
    assert event.Event.finished is False


def test_decorator_preserves_function_name_and_doc():
    def dummy_func():
        """docstring"""
        return 42

    decorated = event.Event.check(dummy_func)
    assert decorated.__name__ == "dummy_func"
    assert decorated.__doc__ == "docstring"
    assert decorated() == 42


def test_finished_property_is_read_write():
    evt = event.Event()
    # Initially false
    assert evt.finished is False
    # Write true sets finished true
    evt.finished = True
    assert evt.finished is True
    # Writing false does not change it
    evt.finished = False
    assert evt.finished is True


def test_multiple_event_instances_independent_finished():
    evt1 = event.Event()
    evt2 = event.Event()

    evt1.finished = True
    assert evt1.finished is True
    assert evt2.finished is False

    evt2.finished = True
    assert evt1.finished is True
    assert evt2.finished is True


def test_class_attribute_finished_exists_and_default_false():
    # Initially class attribute finished should not exist or be False
    val = getattr(event.Event, 'finished', None)
    assert val in (None, False)


def test_setting_event_finished_to_true_only_when_true_sets_private():
    evt = event.Event()
    # Setting to True sets _finish to True
    evt.finished = True
    assert evt._finish is True
    # Setting to False doesn't change _finish
    evt.finished = False
    assert evt._finish is True


def test_check_decorator_returns_function_result():
    @event.Event.check
    def add(a, b):
        return a + b

    result = add(10, 20)
    assert result == 30


def test_check_decorator_can_wrap_methods_with_args_and_kwargs(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class A:
        def __init__(self):
            self.called = False

        @event.Event.check
        def method(self, x, y=5):
            self.called = True
            return x + y

    a = A()
    ret = a.method(3, y=7)
    assert ret == 10
    assert a.called is True


def test_finished_property_returns_private_finish():
    e = event.Event()
    e._finish = True
    assert e.finished is True
    e._finish = False
    assert e.finished is False


def test_finished_setter_only_allows_true_values():
    e = event.Event()
    e._finish = False
    e.finished = False  # should not change
    assert e._finish is False
    e.finished = True
    assert e._finish is True


def test_decorator_does_not_affect_function_exception(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class DummyError(Exception):
        pass

    class D:
        @event.Event.check
        def raises(self):
            raise DummyError("fail")

    d = D()
    with pytest.raises(DummyError):
        d.raises()
    # After exception, finished should still be False (decorator exits early)
    # Actually, the "finished" flag is set AFTER the function executes successfully, so on exception it won't be set
    # And class attribute should be False or not set at all
    assert getattr(event.Event, 'finished', False) is False


def test_finished_setter_accepts_only_boolean_true(monkeypatch):
    e = event.Event()
    e.finished = "True"  # string True is ignored
    assert e.finished is False
    e.finished = 1      # int 1 should be ignored because not literal True boolean
    assert e.finished is False
    e.finished = True   # actual True sets flag
    assert e.finished is True


def test_check_decorator_can_be_used_on_classmethod(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class C:
        @classmethod
        @event.Event.check
        def cls_method(cls, v):
            return v * 3

    result = C.cls_method(4)
    assert result == 12
    # After call finished flag reset to False
    assert event.Event.finished is False


def test_check_decorator_can_be_used_on_staticmethod(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class C:
        @staticmethod
        @event.Event.check
        def static_method(v):
            return v + 1

    result = C.static_method(9)
    assert result == 10
    assert event.Event.finished is False
