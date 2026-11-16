# -*- coding: utf-8 -*-
import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _instantiate_or_skip():
    """Create a TimeMarker with no arguments, or skip if the constructor requires some."""
    try:
        sig = inspect.signature(TimeMarker)
    except (TypeError, ValueError):
        pytest.skip("Cannot inspect TimeMarker signature")
    for param in sig.parameters.values():
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            pytest.skip("TimeMarker requires constructor arguments")
    return TimeMarker()


def test_package_imports_without_error():
    module = importlib.import_module("timemarker")
    assert module is not None


def test_python_version_gate_is_satisfied():
    assert sys.version_info >= (3, 6)


def test_timemarker_exposed_at_package_level():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_is_same_object_as_timer_module_attribute():
    timer_module = importlib.import_module("timemarker.timer")
    assert timemarker.TimeMarker is timer_module.TimeMarker


def test_timer_submodule_is_importable():
    timer_module = importlib.import_module("timemarker.timer")
    assert hasattr(timer_module, "TimeMarker")


def test_timemarker_defined_in_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_timemarker_name_is_correct():
    assert TimeMarker.__name__ == "TimeMarker"


def test_timemarker_is_not_an_instance_of_builtin_type():
    assert not isinstance(TimeMarker, type(None))
    assert type(TimeMarker) is type


def test_timemarker_has_callable_constructor():
    assert callable(TimeMarker)


def test_timemarker_constructor_signature_is_inspectable():
    sig = inspect.signature(TimeMarker)
    assert isinstance(sig.parameters, dict) or hasattr(sig.parameters, "items")


def test_instantiation_produces_timemarker_instance():
    instance = _instantiate_or_skip()
    assert isinstance(instance, TimeMarker)


def test_instance_is_not_none():
    instance = _instantiate_or_skip()
    assert instance is not None


def test_two_instances_are_distinct_objects():
    first = _instantiate_or_skip()
    second = TimeMarker()
    assert first is not second


def test_instance_repr_is_string():
    instance = _instantiate_or_skip()
    assert isinstance(repr(instance), str)


def test_instance_str_is_string():
    instance = _instantiate_or_skip()
    assert isinstance(str(instance), str)


def test_instance_class_attribute_points_back_to_timemarker():
    instance = _instantiate_or_skip()
    assert instance.__class__ is TimeMarker


def test_timemarker_is_importable_via_from_import_in_fresh_namespace():
    namespace = {}
    exec("from timemarker import TimeMarker as T", namespace)
    assert namespace["T"] is TimeMarker
