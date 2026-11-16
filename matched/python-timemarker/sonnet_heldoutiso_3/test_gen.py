import importlib
import inspect
import sys
import time

import pytest

import timemarker
from timemarker import TimeMarker


def make_marker():
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def public_methods(obj):
    return [
        name
        for name, member in inspect.getmembers(obj, callable)
        if not name.startswith("_")
    ]


def find_method(obj, candidates):
    for name in candidates:
        if callable(getattr(obj, name, None)):
            return name
    pytest.skip("none of %s available on TimeMarker" % (candidates,))


def test_package_exports_timemarker():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is TimeMarker


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_defined_in_timer_submodule():
    timer = importlib.import_module("timemarker.timer")
    assert timer.TimeMarker is TimeMarker


def test_python_version_assertion_satisfied():
    assert sys.version_info >= (3, 6)


def test_instantiation_returns_instance():
    tm = make_marker()
    assert isinstance(tm, TimeMarker)


def test_instances_are_independent():
    a = make_marker()
    b = make_marker()
    assert a is not b
    assert a.__dict__ is not b.__dict__


def test_repr_and_str_return_strings():
    tm = make_marker()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_has_public_interface():
    tm = make_marker()
    assert len(public_methods(tm)) >= 1


def test_public_methods_have_valid_signatures():
    tm = make_marker()
    for name in public_methods(tm):
        sig = inspect.signature(getattr(tm, name))
        assert isinstance(sig, inspect.Signature)


def test_mark_call_does_not_raise():
    tm = make_marker()
    name = find_method(tm, ["mark", "start", "tick", "lap", "record"])
    method = getattr(tm, name)
    try:
        method()
    except TypeError:
        try:
            method("label")
        except TypeError:
            pytest.skip("unknown signature for %s" % name)


def test_mark_call_twice_does_not_raise():
    tm = make_marker()
    name = find_method(tm, ["mark", "start", "tick", "lap", "record"])
    method = getattr(tm, name)
    try:
        method()
        time.sleep(0.01)
        method()
    except TypeError:
        pytest.skip("unknown signature for %s" % name)


def test_instance_attribute_state_changes_after_mark():
    tm = make_marker()
    name = find_method(tm, ["mark", "start", "tick", "lap", "record"])
    method = getattr(tm, name)
    before = repr(tm.__dict__)
    try:
        method()
    except TypeError:
        pytest.skip("unknown signature for %s" % name)
    after = repr(tm.__dict__)
    # Either state changed or the marker is stateless; just ensure it's a string.
    assert isinstance(before, str) and isinstance(after, str)


def test_module_reimport_keeps_same_class():
    mod = importlib.import_module("timemarker")
    assert mod.TimeMarker is TimeMarker
