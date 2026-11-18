import time
import pytest
from Tracker.event import Event, status

def test_event_initial_finished_false():
    e = Event()
    assert e.finished is False

def test_event_finished_setter_only_allows_true():
    e = Event()
    e.finished = True
    assert e.finished is True

    e.finished = False  # Should not change because setter only sets if True
    assert e.finished is True

    e.finished = None
    assert e.finished is True

def test_event_finished_property_read_and_write():
    e = Event()
    # Initially False
    assert e.finished is False
    # Setting True changes _finish to True
    e.finished = True
    assert e._finish is True
    # Setting False does not change _finish
    e.finished = False
    assert e._finish is True

def test_status_is_instance_of_event():
    assert isinstance(status, Event)
    assert status.finished is False

def test_check_decorator_sets_finished_true_and_false(tmp_path):
    e = Event()
    @e.check
    def f():
        return "called"

    start = time.time()
    result = f()
    duration = time.time() - start

    # Result is what the function returns
    assert result == "called"
    # After call, finished should be reset to False (because the setter only sets True, this is via class variable, see below)
    # But note in original: wrapper sets cls.finished = True, sleeps, then cls.finished = False
    # cls.finished is a class variable, but class Event does not have _finish at class level, so it triggers setter
    # The setter only sets _finish if True, so setting False does nothing.
    # Thus, finished remains True after decorator.
    # But _finish is instance attribute, cls.finished in decorator sets/gets class var "finished" attribute dynamically, not instance attribute.
    # The behavior is a bit confusing: class variable "finished" is different from instance _finish.
    # Verify this behavior:

    # Check class attribute finished (class var)
    assert hasattr(Event, "finished")
    # Event.finished is a descriptor property so Event.finished will refer to property object, so directly accessing class variable "finished" is not trivial
    # But wrapper sets cls.finished = True/False where cls is Event, so this creates an attribute Event.finished (shadowing property)
    # So test if Event.finished after f() call is False (last assignment in wrapper)
    # Actually Event.finished is property descriptor; but assigning Event.finished = True creates class var that shadows property
    # So after call, Event.finished should be False as set last in wrapper?

    # This means after call, Event.finished (class attribute) is False
    assert getattr(Event, "finished", None) is False

    # The instance attribute _finish does not change by class attribute manipulation
    # So e.finished property reads instance _finish which remains False initially

def test_check_decorator_preserves_function_metadata():
    e = Event()

    @e.check
    def sample_func(a, b=2):
        """Sample docstring."""
        return a + b

    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Sample docstring."

def test_check_decorator_calls_function_with_args_and_kwargs():
    e = Event()

    @e.check
    def func(a, b=1):
        return a * b

    assert func(3) == 3
    assert func(3, b=4) == 12

def test_check_decorator_sleep_duration():
    e = Event()

    @e.check
    def dummy():
        return True

    start = time.perf_counter()
    dummy()
    end = time.perf_counter()
    elapsed = end - start
    assert elapsed >= 0.1

def test_finished_property_setter_does_not_accept_non_boolean():
    e = Event()
    e.finished = True
    assert e.finished is True

    # Assigning non-boolean True-like values such as 1 should NOT change _finish (setter checks `new_status is True`)
    e.finished = 1
    assert e.finished is True  # remains True, no change because 1 is not True (identity)

    # Assigning string True should not change
    e.finished = "True"
    assert e.finished is True

def test_finished_getter_and_setter_do_not_affect_other_attributes():
    e = Event()
    e.finished = True
    e._finish_extra = 123
    # Property finished getter returns _finish, should not interfere with other attributes
    assert e.finished is True
    assert e._finish_extra == 123

def test_check_decorator_class_attribute_does_not_affect_instance_finished():
    e = Event()
    # Clear any class attribute 'finished' set by previous tests
    if hasattr(Event, "finished") and not isinstance(getattr(Event, "finished"), property):
        delattr(Event, "finished")

    collected = []

    @e.check
    def dummy(x):
        collected.append(x)
        return x

    val = dummy(10)
    assert val == 10
    # The class attribute finishing True is set/reset but instance _finish remains unchanged
    assert e.finished is False

def test_setting_finished_true_multiple_times_idempotent():
    e = Event()
    e.finished = True
    assert e.finished is True
    e.finished = True
    assert e.finished is True

def test_setting_finished_false_does_not_reset_true_state():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True

def test_event_class_check_returns_decorator_that_wraps_function():
    decorator = Event.check
    def foo():
        return "foo"
    wrapped = decorator(foo)
    assert callable(wrapped)
    assert wrapped() == "foo"

def test_event_check_wrapper_uses_cls_finished_attribute():
    # We want to check that cls.finished attribute on Event class toggled by the decorator.
    # Remove class attribute if exists first
    if hasattr(Event, "finished") and not isinstance(getattr(Event, "finished"), property):
        delattr(Event, "finished")

    @Event.check
    def testfunc():
        return "ok"

    assert testfunc() == "ok"
    assert getattr(Event, "finished") is False

def test_decorator_works_with_multiple_calls():
    @status.check
    def f(x):
        return x * 2

    for i in range(3):
        result = f(i)
        assert result == i * 2
    # At end, class attribute finished should be False
    assert getattr(Event, "finished") is False

def test_finished_setter_type_enforcement():
    e = Event()
    # The setter ignores non-exact True (identity)
    e.finished = True
    assert e.finished is True
    e.finished = 1     # 1 is truthy but not True identity
    assert e.finished is True
    e.finished = False
    assert e.finished is True
    e.finished = None
    assert e.finished is True

def test_finished_property_is_boolean():
    e = Event()
    assert isinstance(e.finished, bool)
    e.finished = True
    assert isinstance(e.finished, bool)

def test_check_wrapper_returns_original_function_result():
    e = Event()

    @e.check
    def add(a, b):
        return a + b

    assert add(3, 4) == 7

def test_check_wrapper_preserves_function_signature():
    import inspect

    e = Event()

    def example(x, y=1):
        return x + y

    wrapped = e.check(example)
    sig_orig = inspect.signature(example)
    sig_wrapped = inspect.signature(wrapped)
    # signatures may differ due to wrapper but parameters should be same length/names
    assert list(sig_wrapped.parameters.keys()) == list(sig_orig.parameters.keys())

def test_finished_property_is_instance_property_not_class_attribute():
    e1 = Event()
    e2 = Event()
    e1.finished = True
    # Only e1 finished changed (since it's instance attribute _finish)
    assert e1.finished is True
    assert e2.finished is False
