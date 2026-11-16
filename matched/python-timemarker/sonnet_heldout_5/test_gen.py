import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _make_instance():
    """Build a TimeMarker with no arguments, skipping if that isn't supported."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def test_timemarker_is_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")
    assert "TimeMarker" in dir(timemarker)


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_defined_in_timer_submodule():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_package_export_is_same_object_as_submodule_attribute():
    timer_module = importlib.import_module("timemarker.timer")
    assert timer_module.TimeMarker is TimeMarker


def test_python_version_requirement_satisfied():
    assert sys.version_info >= (3, 6)


def test_reload_package_keeps_timemarker_available():
    reloaded = importlib.reload(timemarker)
    assert inspect.isclass(reloaded.TimeMarker)
    assert reloaded.TimeMarker.__name__ == "TimeMarker"


def test_class_signature_is_inspectable():
    sig = inspect.signature(TimeMarker)
    assert isinstance(sig, inspect.Signature)


def test_instantiation_returns_instance():
    obj = _make_instance()
    assert isinstance(obj, TimeMarker)


def test_instances_are_distinct_objects():
    a = _make_instance()
    b = _make_instance()
    assert a is not b


def test_instance_equals_itself_and_is_hashable_or_explicitly_not():
    obj = _make_instance()
    assert obj == obj
    try:
        hash(obj)
    except TypeError:
        # Class explicitly defines __eq__ without __hash__; acceptable.
        pass


def test_repr_and_str_return_strings():
    obj = _make_instance()
    assert isinstance(repr(obj), str)
    assert isinstance(str(obj), str)


def test_class_name_attribute():
    assert TimeMarker.__name__ == "TimeMarker"


def test_class_can_be_subclassed():
    class Sub(TimeMarker):
        pass

    assert issubclass(Sub, TimeMarker)
    try:
        obj = Sub()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(obj, TimeMarker)
    assert isinstance(obj, Sub)


def test_public_attributes_accessible_on_instance():
    obj = _make_instance()
    public = [n for n in dir(TimeMarker) if not n.startswith("_")]
    for name in public:
        # Accessing a class-level public attribute through the instance must
        # not raise AttributeError (properties may raise other errors, so
        # only check the class-level descriptor presence).
        assert hasattr(type(obj), name)
        class_attr = inspect.getattr_static(type(obj), name)
        assert class_attr is not None
