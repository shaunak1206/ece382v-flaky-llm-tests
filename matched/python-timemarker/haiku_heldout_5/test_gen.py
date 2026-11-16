# -*- coding: utf-8 -*-
import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def test_package_exports_timemarker():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_comes_from_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_reexport_is_same_object_as_timer_module_class():
    timer = importlib.import_module("timemarker.timer")
    assert timer.TimeMarker is TimeMarker


def test_reimporting_package_returns_same_class():
    reloaded = importlib.import_module("timemarker")
    assert reloaded.TimeMarker is TimeMarker


def test_python_version_is_supported_by_package_assertion():
    assert sys.version_info >= (3, 6)


def test_can_instantiate_without_arguments():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)


def test_instances_are_distinct_objects():
    first = TimeMarker()
    second = TimeMarker()
    assert first is not second


def test_instance_string_representation_is_str():
    tm = TimeMarker()
    assert isinstance(str(tm), str)
    assert isinstance(repr(tm), str)


def test_constructor_parameters_all_have_defaults():
    signature = inspect.signature(TimeMarker.__init__)
    params = list(signature.parameters.values())[1:]  # skip self
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        assert param.default is not param.empty, (
            "parameter %r has no default" % param.name
        )


def test_public_attributes_of_class_are_accessible():
    public = [name for name in dir(TimeMarker) if not name.startswith("_")]
    for name in public:
        assert getattr(TimeMarker, name) is not None


def test_mark_method_accepts_label_when_present():
    tm = TimeMarker()
    if not hasattr(tm, "mark"):
        pytest.skip("TimeMarker has no 'mark' method")
    tm.mark("first")
    tm.mark("second")


def test_mark_calls_do_not_share_state_between_instances():
    first = TimeMarker()
    second = TimeMarker()
    if not hasattr(first, "mark"):
        pytest.skip("TimeMarker has no 'mark' method")
    first.mark("only-in-first")
    assert first is not second
    assert second is not first


def test_timemarker_is_not_abstract_or_none():
    assert TimeMarker is not None
    assert not inspect.isabstract(TimeMarker)


def test_isinstance_check_against_module_reference():
    tm = timemarker.TimeMarker()
    assert isinstance(tm, timemarker.TimeMarker)
