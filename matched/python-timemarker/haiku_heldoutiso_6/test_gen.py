# -*- coding: utf-8 -*-
import copy
import importlib
import inspect

import pytest

import timemarker
from timemarker import TimeMarker
from timemarker import timer as timer_module


@pytest.fixture
def marker():
    """Return a default-constructed TimeMarker, skipping if it requires arguments."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def test_package_exports_timemarker_class():
    assert hasattr(timemarker, "TimeMarker")
    assert isinstance(TimeMarker, type)


def test_exported_class_is_same_object_as_timer_module_class():
    assert TimeMarker is timer_module.TimeMarker


def test_class_name_is_timemarker():
    assert TimeMarker.__name__ == "TimeMarker"


def test_class_qualname_belongs_to_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_reimporting_package_returns_same_class():
    reloaded = importlib.import_module("timemarker")
    assert reloaded.TimeMarker is TimeMarker


def test_timer_module_is_importable_directly():
    module = importlib.import_module("timemarker.timer")
    assert hasattr(module, "TimeMarker")


def test_constructor_signature_is_inspectable():
    sig = inspect.signature(TimeMarker)
    assert sig is not None


def test_default_instantiation_produces_instance(marker):
    assert isinstance(marker, TimeMarker)
    assert marker is not None


def test_two_instances_are_distinct_objects():
    try:
        first = TimeMarker()
        second = TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert first is not second


def test_instance_equals_itself(marker):
    assert marker == marker


def test_instance_is_hashable(marker):
    assert isinstance(hash(marker), int)


def test_repr_returns_string(marker):
    result = repr(marker)
    assert isinstance(result, str)
    assert result


def test_str_returns_string(marker):
    result = str(marker)
    assert isinstance(result, str)


def test_class_is_subclassable():
    class SubMarker(TimeMarker):
        pass

    assert issubclass(SubMarker, TimeMarker)
    assert isinstance(SubMarker.__new__(SubMarker), TimeMarker)


def test_shallow_copy_produces_equal_type(marker):
    duplicate = copy.copy(marker)
    assert type(duplicate) is TimeMarker
    assert duplicate is not marker


def test_class_defines_public_callable_members():
    public_members = [
        name
        for name, value in vars(TimeMarker).items()
        if not name.startswith("_") and callable(value)
    ]
    assert public_members, "Expected TimeMarker to define at least one public method"
