# -*- coding: utf-8 -*-
import importlib
import inspect
from sys import version_info

import pytest

import timemarker
from timemarker import TimeMarker


def _instantiate_or_skip():
    """Create a TimeMarker with no arguments, or skip if its constructor requires them."""
    try:
        sig = inspect.signature(TimeMarker)
    except (TypeError, ValueError):
        pytest.skip("Could not inspect TimeMarker signature")
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if required:
        pytest.skip("TimeMarker requires constructor arguments")
    return TimeMarker()


def test_python_version_guard_satisfied():
    assert version_info >= (3, 6)


def test_timemarker_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_not_none():
    assert TimeMarker is not None


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)
    assert isinstance(TimeMarker, type)


def test_class_name_is_timemarker():
    assert TimeMarker.__name__ == "TimeMarker"


def test_class_module_belongs_to_package():
    assert TimeMarker.__module__.startswith("timemarker")


def test_timer_submodule_importable():
    module = importlib.import_module("timemarker.timer")
    assert hasattr(module, "TimeMarker")


def test_package_export_is_same_object_as_submodule_class():
    timer_module = importlib.import_module("timemarker.timer")
    assert timemarker.TimeMarker is timer_module.TimeMarker


def test_repeated_imports_return_same_class():
    again = importlib.import_module("timemarker")
    assert again.TimeMarker is TimeMarker


def test_package_dir_lists_timemarker():
    assert "TimeMarker" in dir(timemarker)


def test_class_defines_public_callables():
    public_callables = [
        name for name, value in vars(TimeMarker).items()
        if not name.startswith("_") and callable(value)
    ]
    assert public_callables, "TimeMarker should define at least one public method"


def test_all_defined_public_members_are_callable_or_descriptors():
    for name, value in vars(TimeMarker).items():
        if name.startswith("_"):
            continue
        assert callable(value) or isinstance(value, (property, staticmethod, classmethod)), name


def test_init_signature_is_inspectable():
    sig = inspect.signature(TimeMarker)
    assert sig is not None


def test_instance_can_be_created_without_arguments():
    instance = _instantiate_or_skip()
    assert isinstance(instance, TimeMarker)


def test_instances_are_distinct_objects():
    first = _instantiate_or_skip()
    second = TimeMarker()
    assert first is not second


def test_instance_repr_and_str_return_strings():
    instance = _instantiate_or_skip()
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)


def test_instance_is_hashable():
    instance = _instantiate_or_skip()
    assert isinstance(hash(instance), int)
    assert {instance: True}[instance] is True


def test_instance_public_callables_are_bound_methods():
    instance = _instantiate_or_skip()
    for name in dir(instance):
        if name.startswith("_"):
            continue
        attr = getattr(instance, name)
        if inspect.ismethod(attr):
            assert attr.__self__ is instance, name
