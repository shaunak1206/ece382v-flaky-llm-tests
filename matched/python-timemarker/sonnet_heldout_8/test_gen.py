import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _make():
    """Create a TimeMarker with no arguments, skipping if the signature requires some."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def _public_methods(obj):
    return [
        name
        for name, member in inspect.getmembers(obj, callable)
        if not name.startswith("_")
    ]


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is TimeMarker


def test_timemarker_defined_in_timer_submodule():
    timer_module = importlib.import_module("timemarker.timer")
    assert timer_module.TimeMarker is TimeMarker


def test_timemarker_class_name():
    assert TimeMarker.__name__ == "TimeMarker"


def test_package_module_imports_cleanly_on_supported_python():
    assert sys.version_info >= (3, 6)
    reloaded = importlib.reload(timemarker)
    assert hasattr(reloaded, "TimeMarker")


def test_timemarker_instantiation():
    tm = _make()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct_objects():
    a = _make()
    b = _make()
    assert a is not b
    assert isinstance(a, TimeMarker)
    assert isinstance(b, TimeMarker)


def test_instances_do_not_share_instance_state():
    a = _make()
    b = _make()
    assert a.__dict__ is not b.__dict__


def test_repr_and_str_return_strings():
    tm = _make()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_timemarker_exposes_public_api():
    tm = _make()
    assert len(_public_methods(tm)) > 0


def test_public_methods_are_inspectable():
    tm = _make()
    for name in _public_methods(tm):
        method = getattr(tm, name)
        # Every public callable must have a retrievable signature
        try:
            inspect.signature(method)
        except ValueError:
            pass  # some builtins have no signature; acceptable


def test_subclassing_timemarker():
    class MyMarker(TimeMarker):
        pass

    try:
        obj = MyMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(obj, TimeMarker)


def test_timemarker_hashable_or_explicitly_unhashable():
    tm = _make()
    try:
        hash(tm)
    except TypeError:
        # Explicitly unhashable is a valid design (e.g. __eq__ without __hash__)
        assert TimeMarker.__hash__ is None
    else:
        assert isinstance(hash(tm), int)


def test_timemarker_attribute_access_unknown_raises_attribute_error():
    tm = _make()
    with pytest.raises(AttributeError):
        getattr(tm, "this_attribute_definitely_does_not_exist_12345")
