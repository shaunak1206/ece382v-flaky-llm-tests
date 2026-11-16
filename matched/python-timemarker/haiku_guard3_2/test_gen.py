# tests/test_timemarker_init.py
#
# Note: only timemarker/__init__.py was provided, and it re-exports TimeMarker
# from timemarker/timer.py. Without that module's source, these tests cover the
# public surface exposed by the package and the class's observable properties,
# and they do not exercise TimeMarker's methods.

import importlib
import inspect
import os
import sys
import types

import pytest

import timemarker
from timemarker import TimeMarker


def test_package_exposes_timemarker_name():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_is_same_object_as_timer_module_class():
    from timemarker.timer import TimeMarker as TimerClass

    assert TimeMarker is TimerClass


def test_package_attribute_matches_imported_name():
    assert timemarker.TimeMarker is TimeMarker


def test_timer_submodule_is_bound_on_package():
    assert inspect.ismodule(timemarker.timer)
    assert timemarker.timer.__name__ == "timemarker.timer"


def test_timemarker_defined_in_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_timemarker_class_name():
    assert TimeMarker.__name__ == "TimeMarker"


def test_timemarker_source_file_is_timer_py_in_package():
    source_file = inspect.getfile(TimeMarker)
    package_dir = os.path.dirname(os.path.abspath(timemarker.__file__))
    assert os.path.basename(source_file) == "timer.py"
    assert os.path.dirname(os.path.abspath(source_file)) == package_dir


def test_timemarker_has_signature_introspectable():
    sig = inspect.signature(TimeMarker)
    assert isinstance(sig, inspect.Signature)


def test_timemarker_metaclass_is_type():
    assert type(TimeMarker) is type


def test_timemarker_can_be_subclassed():
    class SubMarker(TimeMarker):
        pass

    assert issubclass(SubMarker, TimeMarker)
    assert SubMarker.__mro__[1] is TimeMarker


def test_timemarker_listed_in_dir_of_package():
    assert "TimeMarker" in dir(timemarker)


def test_star_import_provides_timemarker():
    namespace = {}
    exec("from timemarker import *", namespace)
    assert "TimeMarker" in namespace
    assert namespace["TimeMarker"] is TimeMarker


def test_import_module_returns_cached_package():
    module = importlib.import_module("timemarker")
    assert module is timemarker
    assert sys.modules["timemarker"] is timemarker


def test_package_is_a_module_object():
    assert isinstance(timemarker, types.ModuleType)


def test_timemarker_identity_is_stable_across_lookups():
    first = timemarker.TimeMarker
    second = getattr(timemarker, "TimeMarker")
    assert first is second


def test_import_fails_on_unsupported_python_version(monkeypatch):
    # Simulate an old interpreter; the package's version assert should fire.
    # monkeypatch restores sys.modules and sys.version_info afterwards.
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "version_info", (2, 7, 18, "final", 0))

    with pytest.raises(AssertionError, match="python >= 3.6"):
        importlib.import_module("timemarker")
