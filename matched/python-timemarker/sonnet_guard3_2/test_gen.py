import copy
import inspect
import sys
import time

import pytest

import timemarker
from timemarker import TimeMarker


def _make():
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def _public_callables(obj):
    result = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        if callable(attr):
            result.append(name)
    return result


def test_timemarker_is_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is TimeMarker


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_defined_in_timer_module():
    import timemarker.timer as timer_module

    assert timer_module.TimeMarker is TimeMarker
    assert TimeMarker.__module__ == "timemarker.timer"


def test_python_version_requirement_satisfied():
    assert sys.version_info >= (3, 6)


def test_default_instantiation_gives_instance():
    tm = _make()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct_objects():
    a = _make()
    b = _make()
    assert a is not b


def test_instance_state_is_not_shared_between_instances():
    a = _make()
    b = _make()
    if not hasattr(a, "__dict__"):
        pytest.skip("instances have no __dict__")
    a.some_new_test_attribute = 123
    assert not hasattr(b, "some_new_test_attribute")
    assert not hasattr(TimeMarker, "some_new_test_attribute")


def test_instances_have_same_public_interface():
    a = _make()
    b = _make()
    assert _public_callables(a) == _public_callables(b)


def test_has_public_methods():
    names = [
        n for n, _ in inspect.getmembers(TimeMarker, predicate=inspect.isfunction)
        if not n.startswith("_")
    ]
    assert len(names) >= 1


def test_public_method_signatures_are_inspectable():
    tm = _make()
    for name in _public_callables(tm):
        attr = getattr(tm, name)
        try:
            sig = inspect.signature(attr)
        except (TypeError, ValueError):
            continue
        assert isinstance(sig, inspect.Signature)


def test_instantiation_does_not_modify_class_namespace():
    before = sorted(vars(TimeMarker).keys())
    _make()
    _make()
    after = sorted(vars(TimeMarker).keys())
    assert before == after


def test_instantiation_with_fixed_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    if hasattr(time, "perf_counter"):
        monkeypatch.setattr(time, "perf_counter", lambda: 1000.0)
    tm = _make()
    assert isinstance(tm, TimeMarker)


def test_repr_and_str_return_strings():
    tm = _make()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_can_be_subclassed():
    class Child(TimeMarker):
        extra = 5

    try:
        child = Child()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(child, TimeMarker)
    assert child.extra == 5
    assert not hasattr(TimeMarker, "extra")


def test_copy_of_instance_is_independent():
    tm = _make()
    if not hasattr(tm, "__dict__"):
        pytest.skip("instances have no __dict__")
    try:
        clone = copy.copy(tm)
    except Exception:
        pytest.skip("instance not copyable")
    assert isinstance(clone, TimeMarker)
    clone.another_test_attribute = 1
    assert not hasattr(tm, "another_test_attribute")
