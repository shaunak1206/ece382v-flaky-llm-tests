import importlib
import inspect
import pkgutil
import sys

import pytest

import timemarker


def test_package_import_succeeds():
    assert timemarker is not None


def test_time_marker_is_exposed_at_package_level():
    assert hasattr(timemarker, "TimeMarker")


def test_time_marker_can_be_imported_directly_from_package():
    from timemarker import TimeMarker

    assert TimeMarker is timemarker.TimeMarker


def test_time_marker_reexports_timer_class():
    from timemarker.timer import TimeMarker as TimerTimeMarker

    assert timemarker.TimeMarker is TimerTimeMarker


def test_time_marker_is_a_class_or_type_like_object():
    # The package re-exports a class from timemarker.timer.
    assert isinstance(timemarker.TimeMarker, type)


def test_time_marker_name_matches():
    assert timemarker.TimeMarker.__name__ == "TimeMarker"


def test_time_marker_module_matches_timer_module():
    assert timemarker.TimeMarker.__module__ == "timemarker.timer"


def test_package_has_version_info_from_sys():
    assert timemarker.version_info >= (3, 6)


def test_package_init_file_is_loaded_from_timemarker_package():
    assert timemarker.__file__.endswith("__init__.py")


def test_import_star_brings_time_marker_into_namespace():
    namespace = {}
    exec("from timemarker import *", namespace)
    assert "TimeMarker" in namespace
    assert namespace["TimeMarker"] is timemarker.TimeMarker


def test_package_does_not_define_unexpected_all():
    assert not hasattr(timemarker, "__all__")


def test_reload_keeps_time_marker_identity():
    original = timemarker.TimeMarker
    reloaded = importlib.reload(timemarker)
    assert reloaded.TimeMarker is original


def test_pkgutil_finds_timemarker_package():
    assert pkgutil.find_loader("timemarker") is not None


def test_source_contains_version_guard_message():
    source = inspect.getsource(timemarker)
    assert "Currently only python >= 3.6 is supported" in source


def test_source_reexports_timemarker_class():
    source = inspect.getsource(timemarker)
    assert "from .timer import TimeMarker" in source


def test_from_package_imported_object_is_same_as_module_attribute_after_reload():
    import timemarker as tm1

    tm2 = importlib.reload(tm1)
    from timemarker import TimeMarker

    assert TimeMarker is tm2.TimeMarker
    assert TimeMarker is tm1.TimeMarker


def test_package_namespace_includes_time_marker():
    assert "TimeMarker" in timemarker.__dict__


def test_time_marker_object_is_accessible_via_getattr():
    assert getattr(timemarker, "TimeMarker") is timemarker.TimeMarker
