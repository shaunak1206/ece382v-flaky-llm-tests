import importlib
import pathlib
import sys
import types

import pytest

import timemarker


def test_package_import_succeeds():
    assert timemarker is not None


def test_time_marker_is_exposed_on_package():
    assert hasattr(timemarker, "TimeMarker")


def test_time_marker_can_be_imported_from_package():
    from timemarker import TimeMarker

    assert TimeMarker is timemarker.TimeMarker


def test_time_marker_matches_timer_module_symbol():
    from timemarker.timer import TimeMarker as TimerTimeMarker

    assert timemarker.TimeMarker is TimerTimeMarker


def test_time_marker_is_a_class():
    assert isinstance(timemarker.TimeMarker, type)


def test_time_marker_module_is_timer():
    assert timemarker.TimeMarker.__module__ == "timemarker.timer"


def test_package_dir_contains_time_marker():
    assert "TimeMarker" in dir(timemarker)


def test_reimport_returns_same_time_marker_object():
    module1 = importlib.import_module("timemarker")
    module2 = importlib.reload(module1)

    assert module1.TimeMarker is module2.TimeMarker


def test_star_import_exports_time_marker_via_namespace():
    namespace = {}
    exec("from timemarker import *", namespace)

    assert namespace["TimeMarker"] is timemarker.TimeMarker


def test_imported_time_marker_is_callable_or_instantiable():
    # This checks the public symbol can at least be referenced like a class.
    # If the implementation requires arguments, the assertion is still valid
    # because we're only verifying the exported object exists and is a type.
    assert callable(timemarker.TimeMarker)


def test_module_file_exists():
    path = pathlib.Path(timemarker.__file__)
    assert path.name == "__init__.py"
    assert path.exists()


def test_import_source_has_version_guard():
    source = pathlib.Path(timemarker.__file__).read_text(encoding="utf-8")
    assert "assert version_info >= (3, 6)" in source


def test_version_guard_accepts_supported_version():
    source = pathlib.Path(timemarker.__file__).read_text(encoding="utf-8")
    fake_sys = types.ModuleType("sys")
    fake_sys.version_info = (3, 8, 0, "final", 0)

    old_sys = sys.modules["sys"]
    try:
        sys.modules["sys"] = fake_sys
        module = types.ModuleType("timemarker_init_test_supported")
        exec(compile(source, str(pathlib.Path(timemarker.__file__)), "exec"), module.__dict__)
    finally:
        sys.modules["sys"] = old_sys

    assert module.TimeMarker is timemarker.TimeMarker


def test_version_guard_rejects_unsupported_version():
    source = pathlib.Path(timemarker.__file__).read_text(encoding="utf-8")
    fake_sys = types.ModuleType("sys")
    fake_sys.version_info = (3, 5, 9, "final", 0)

    old_sys = sys.modules["sys"]
    try:
        sys.modules["sys"] = fake_sys
        module = types.ModuleType("timemarker_init_test_unsupported")
        with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
            exec(compile(source, str(pathlib.Path(timemarker.__file__)), "exec"), module.__dict__)
    finally:
        sys.modules["sys"] = old_sys
