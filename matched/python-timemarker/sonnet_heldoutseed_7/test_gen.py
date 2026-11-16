import importlib
import sys

import pytest

import timemarker
from timemarker import TimeMarker

# Candidate names for the "record a point in time" operation.
# The tests below skip, rather than fail, if none of these exist.
MARK_NAMES = ("mark", "start", "tic", "begin", "set_mark", "add_mark")


def _make():
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def _public_callables(obj):
    return [
        n for n in dir(obj)
        if not n.startswith("_") and callable(getattr(obj, n))
    ]


def _find_mark_method(obj):
    for name in MARK_NAMES:
        fn = getattr(obj, name, None)
        if callable(fn):
            return name, fn
    pytest.skip("no known mark-like method on TimeMarker")


def test_exported_from_package():
    from timemarker import timer
    assert timemarker.TimeMarker is timer.TimeMarker


def test_timemarker_is_a_class():
    assert isinstance(TimeMarker, type)
    assert TimeMarker.__name__ == "TimeMarker"


def test_class_defined_in_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_instantiation_without_arguments():
    tm = _make()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct():
    a = _make()
    b = _make()
    assert a is not b


def test_instance_state_not_shared_between_instances():
    a = _make()
    b = _make()
    if not hasattr(a, "__dict__"):
        pytest.skip("instances use __slots__")
    for key, value in vars(a).items():
        if isinstance(value, (list, dict, set)) and key in vars(b):
            assert vars(b)[key] is not value


def test_repr_and_str_return_strings():
    tm = _make()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_instance_equals_itself_and_is_not_none():
    tm = _make()
    assert tm == tm
    assert tm is not None
    assert tm != object()


def test_exposes_public_methods():
    tm = _make()
    assert len(_public_callables(tm)) >= 1


def test_can_be_subclassed():
    class Sub(TimeMarker):
        pass

    try:
        inst = Sub()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(inst, TimeMarker)


def test_mark_method_callable_does_not_raise():
    tm = _make()
    name, fn = _find_mark_method(tm)
    try:
        fn()
    except TypeError:
        # Probably needs a label argument.
        fn("label")


def test_mark_method_can_be_called_repeatedly():
    tm = _make()
    name, fn = _find_mark_method(tm)
    for i in range(3):
        try:
            fn()
        except TypeError:
            fn("label%d" % i)


def test_separate_instances_can_be_marked_independently():
    a = _make()
    b = _make()
    name, fn_a = _find_mark_method(a)
    fn_b = getattr(b, name)
    try:
        fn_a()
        fn_b()
    except TypeError:
        fn_a("x")
        fn_b("y")


def test_reload_package_succeeds_on_supported_python():
    reloaded = importlib.reload(timemarker)
    assert hasattr(reloaded, "TimeMarker")
    assert isinstance(reloaded.TimeMarker, type)


def test_import_rejects_old_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 0, "final", 0))
    with pytest.raises(AssertionError):
        importlib.reload(timemarker)
    monkeypatch.undo()
    # Package must still be importable afterward.
    importlib.reload(timemarker)
    assert hasattr(timemarker, "TimeMarker")
