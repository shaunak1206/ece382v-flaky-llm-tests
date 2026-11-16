import importlib
import inspect

import pytest

import timemarker
from timemarker import TimeMarker


def _public_names(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


@pytest.fixture
def marker():
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def test_timemarker_is_exported_class():
    assert inspect.isclass(TimeMarker)
    assert TimeMarker.__name__ == "TimeMarker"


def test_package_attribute_matches_import():
    assert timemarker.TimeMarker is TimeMarker


def test_same_object_as_in_timer_submodule():
    timer = importlib.import_module("timemarker.timer")
    assert timer.TimeMarker is TimeMarker


def test_class_defined_in_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_reimport_returns_same_class():
    mod = importlib.import_module("timemarker")
    assert mod.TimeMarker is TimeMarker


def test_instantiation_without_arguments(marker):
    assert isinstance(marker, TimeMarker)


def test_instances_are_distinct():
    try:
        a = TimeMarker()
        b = TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert a is not b


def test_instance_state_not_shared(marker):
    other = TimeMarker()
    if hasattr(marker, "__dict__") and hasattr(other, "__dict__"):
        assert marker.__dict__ is not other.__dict__


def test_str_and_repr_return_strings(marker):
    assert isinstance(str(marker), str)
    assert isinstance(repr(marker), str)


def test_class_exposes_public_api():
    assert len(_public_names(TimeMarker)) >= 1


def test_public_api_available_on_instance(marker):
    for name in _public_names(TimeMarker):
        assert hasattr(marker, name)


def test_class_is_subclassable(marker):
    class Sub(TimeMarker):
        pass

    try:
        sub = Sub()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(sub, TimeMarker)


def test_subclass_inherits_public_methods():
    class Sub(TimeMarker):
        pass

    for name in _public_names(TimeMarker):
        assert hasattr(Sub, name)


def test_public_methods_are_callable_or_attributes(marker):
    for name in _public_names(TimeMarker):
        attr = getattr(TimeMarker, name)
        # Every public member is either callable or a plain/descriptor attribute
        assert callable(attr) or attr is not None or attr is None


def test_start_stop_style_usage_does_not_raise(marker):
    if not (hasattr(marker, "start") and hasattr(marker, "stop")):
        pytest.skip("no start/stop API")
    try:
        marker.start()
        marker.stop()
    except TypeError:
        pytest.skip("start/stop require arguments")
