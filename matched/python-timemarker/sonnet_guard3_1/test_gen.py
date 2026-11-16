import importlib
import sys

import pytest

import timemarker
import timemarker.timer as timer_module
from timemarker import TimeMarker


def _make():
    """Create a TimeMarker with no arguments, skipping if that is unsupported."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def test_timemarker_is_a_class():
    assert isinstance(TimeMarker, type)


def test_timemarker_reexported_from_timer_module():
    assert timemarker.TimeMarker is timer_module.TimeMarker


def test_timemarker_class_name_and_module():
    assert TimeMarker.__name__ == "TimeMarker"
    assert TimeMarker.__module__ == "timemarker.timer"


def test_package_exposes_version_info():
    assert timemarker.version_info >= (3, 6)


def test_package_has_timemarker_attribute():
    assert hasattr(timemarker, "TimeMarker")


def test_reload_succeeds_on_supported_python():
    reloaded = importlib.reload(timemarker)
    assert reloaded is timemarker
    assert timemarker.TimeMarker is timer_module.TimeMarker


def test_reload_fails_on_old_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 0, "final", 0))
    try:
        with pytest.raises(AssertionError):
            importlib.reload(timemarker)
    finally:
        monkeypatch.undo()
        importlib.reload(timemarker)
    assert timemarker.version_info >= (3, 6)
    assert timemarker.TimeMarker is timer_module.TimeMarker


def test_instance_is_instance_of_class():
    tm = _make()
    assert isinstance(tm, TimeMarker)
    assert type(tm) is TimeMarker


def test_instances_are_distinct_objects():
    a = _make()
    b = _make()
    assert a is not b


def test_instance_attributes_are_independent():
    a = _make()
    b = _make()
    a.some_new_test_attribute = 123
    assert not hasattr(b, "some_new_test_attribute")
    assert not hasattr(TimeMarker, "some_new_test_attribute")


def test_repr_and_str_return_strings():
    tm = _make()
    assert isinstance(repr(tm), str)
    assert isinstance(str(tm), str)


def test_subclass_can_be_created():
    class Sub(TimeMarker):
        pass

    try:
        sub = Sub()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(sub, TimeMarker)
    assert isinstance(sub, Sub)


def test_instance_equals_itself():
    tm = _make()
    assert tm == tm
    assert not (tm != tm)
