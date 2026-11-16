# -*- coding: utf-8 -*-
import copy
import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _required_params(cls):
    """Return the names of required __init__ parameters (excluding self)."""
    sig = inspect.signature(cls)
    return [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _new_instance():
    """Create a TimeMarker, skipping the test if it needs constructor arguments."""
    if _required_params(TimeMarker):
        pytest.skip("TimeMarker requires constructor arguments")
    return TimeMarker()


def test_python_version_guard_passes():
    # The package asserts Python >= 3.6 on import; the test environment must satisfy it.
    assert sys.version_info >= (3, 6)


def test_package_exports_timemarker():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert isinstance(TimeMarker, type)


def test_timemarker_is_same_object_as_in_timer_module():
    timer_module = importlib.import_module("timemarker.timer")
    assert timer_module.TimeMarker is TimeMarker


def test_class_name_and_module():
    assert TimeMarker.__name__ == "TimeMarker"
    assert TimeMarker.__module__.startswith("timemarker")


def test_timemarker_is_not_an_instance_of_unrelated_types():
    assert not issubclass(TimeMarker, (int, str, list, dict))


def test_instantiation_without_arguments():
    tm = _new_instance()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct_objects():
    a = _new_instance()
    b = _new_instance()
    assert a is not b


def test_instance_attributes_are_isolated():
    a = _new_instance()
    b = _new_instance()
    a._timemarker_test_flag = "only-a"
    assert not hasattr(b, "_timemarker_test_flag")


def test_repr_returns_string():
    tm = _new_instance()
    text = repr(tm)
    assert isinstance(text, str)
    assert text


def test_str_returns_string():
    tm = _new_instance()
    assert isinstance(str(tm), str)


def test_public_callables_have_inspectable_signatures():
    for name, member in inspect.getmembers(TimeMarker):
        if name.startswith("_"):
            continue
        if callable(member):
            # Must not raise; ensures the callable is a well-formed function/method.
            inspect.signature(member)


def test_instance_is_identical_to_itself_and_default_equality():
    tm = _new_instance()
    assert tm == tm
    if TimeMarker.__eq__ is object.__eq__:
        other = _new_instance()
        assert tm != other


def test_copy_produces_timemarker_instance():
    tm = _new_instance()
    duplicate = copy.copy(tm)
    assert isinstance(duplicate, TimeMarker)
    assert duplicate is not tm


def test_mark_method_accepts_a_label_if_present():
    tm = _new_instance()
    mark = getattr(tm, "mark", None)
    if mark is None or not callable(mark):
        pytest.skip("TimeMarker has no callable 'mark' method")
    params = _required_params(mark)
    if len(params) > 1:
        pytest.skip("'mark' requires more than one argument")
    # Calling with a single label must not raise.
    if params:
        mark("first")
    else:
        mark()
