import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _make_instance():
    """Build a TimeMarker using only default arguments, or skip if impossible."""
    try:
        sig = inspect.signature(TimeMarker)
    except (TypeError, ValueError):
        pytest.skip("cannot introspect TimeMarker signature")
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        pytest.skip("TimeMarker requires arguments: %s" % [p.name for p in required])
    return TimeMarker()


def test_timemarker_is_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is TimeMarker


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_defined_in_timer_module():
    timer_module = importlib.import_module("timemarker.timer")
    assert timer_module.TimeMarker is TimeMarker


def test_python_version_supported():
    assert sys.version_info >= (3, 6)


def test_default_instantiation_returns_instance():
    obj = _make_instance()
    assert isinstance(obj, TimeMarker)
    assert type(obj) is TimeMarker


def test_instances_are_distinct_objects():
    a = _make_instance()
    b = _make_instance()
    assert a is not b


def test_instance_equals_itself_and_is_hashable_or_raises_typeerror():
    obj = _make_instance()
    assert obj == obj
    try:
        hash(obj)
    except TypeError:
        pass  # unhashable is acceptable if __eq__ is overridden


def test_repr_and_str_return_strings():
    obj = _make_instance()
    assert isinstance(repr(obj), str)
    assert isinstance(str(obj), str)


def test_class_has_public_callable_attributes_or_is_callable():
    public = [
        name for name, member in inspect.getmembers(TimeMarker)
        if not name.startswith("_") and callable(member)
    ]
    # A timing helper is expected to expose at least one public method
    # or be callable itself.
    assert public or callable(getattr(TimeMarker, "__call__", None))


def test_class_can_be_subclassed():
    class Child(TimeMarker):
        pass

    try:
        sig = inspect.signature(Child)
    except (TypeError, ValueError):
        pytest.skip("cannot introspect signature")
    if any(
        p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        for p in sig.parameters.values()
    ):
        pytest.skip("requires arguments")
    child = Child()
    assert isinstance(child, TimeMarker)


def test_reload_keeps_timemarker_exported():
    reloaded = importlib.reload(timemarker)
    assert hasattr(reloaded, "TimeMarker")
    assert inspect.isclass(reloaded.TimeMarker)


def test_init_signature_is_introspectable():
    sig = inspect.signature(TimeMarker.__init__)
    assert "self" in sig.parameters


def test_instance_has_dict_or_slots():
    obj = _make_instance()
    assert hasattr(obj, "__dict__") or hasattr(type(obj), "__slots__")


def test_public_zero_arg_attributes_accessible():
    obj = _make_instance()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        # Accessing attributes (not calling) must not raise.
        getattr(obj, name)
