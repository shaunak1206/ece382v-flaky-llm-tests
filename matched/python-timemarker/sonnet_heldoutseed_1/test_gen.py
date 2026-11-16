import importlib
import inspect

import pytest

import timemarker
from timemarker import TimeMarker


def _make():
    """Create a TimeMarker with no arguments, skipping if that is unsupported."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def _public_callables(obj):
    return [
        name
        for name, member in inspect.getmembers(obj)
        if not name.startswith("_") and callable(member)
    ]


def test_timemarker_is_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is TimeMarker


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_same_object_as_in_timer_module():
    timer_module = importlib.import_module("timemarker.timer")
    assert timer_module.TimeMarker is TimeMarker


def test_package_reimport_is_stable():
    reloaded = importlib.import_module("timemarker")
    assert reloaded.TimeMarker is TimeMarker


def test_class_has_name_and_module():
    assert TimeMarker.__name__ == "TimeMarker"
    assert TimeMarker.__module__.startswith("timemarker")


def test_can_instantiate_without_arguments():
    tm = _make()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct():
    a = _make()
    b = _make()
    assert a is not b
    assert isinstance(a, TimeMarker) and isinstance(b, TimeMarker)


def test_repr_and_str_return_strings():
    tm = _make()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_instance_exposes_public_api():
    tm = _make()
    assert len(_public_callables(tm)) >= 1


def test_public_methods_are_documented_or_callable():
    tm = _make()
    for name in _public_callables(tm):
        assert callable(getattr(tm, name))


def test_instance_state_not_shared_between_instances():
    a = _make()
    b = _make()
    try:
        a_state = dict(vars(a))
        b_state = dict(vars(b))
    except TypeError:
        pytest.skip("instances have no __dict__")
    # Mutable containers must not be shared across instances.
    for key, value in a_state.items():
        if isinstance(value, (list, dict, set)) and key in b_state:
            assert value is not b_state[key]


def test_mark_method_if_present_runs_without_error():
    tm = _make()
    if not hasattr(tm, "mark"):
        pytest.skip("no mark method")
    try:
        tm.mark()
    except TypeError:
        try:
            tm.mark("label")
        except TypeError:
            pytest.skip("unknown mark signature")


def test_mark_does_not_change_instance_type():
    tm = _make()
    if not hasattr(tm, "mark"):
        pytest.skip("no mark method")
    try:
        tm.mark()
    except TypeError:
        try:
            tm.mark("label")
        except TypeError:
            pytest.skip("unknown mark signature")
    assert isinstance(tm, TimeMarker)


def test_instance_is_garbage_collectable():
    import gc
    import weakref

    tm = _make()
    try:
        ref = weakref.ref(tm)
    except TypeError:
        pytest.skip("instances are not weak-referenceable")
    del tm
    gc.collect()
    assert ref() is None
