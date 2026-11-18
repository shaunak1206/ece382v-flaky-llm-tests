import importlib
import sys

import pytest


def test_package_exports_timemarker():
    import timemarker

    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_direct_import_matches_package_attribute():
    import timemarker

    from timemarker import TimeMarker

    assert TimeMarker is timemarker.TimeMarker


def test_time_marker_matches_timer_module_class():
    import timemarker
    import timemarker.timer

    assert timemarker.TimeMarker is timemarker.timer.TimeMarker


def test_time_marker_name_is_preserved():
    from timemarker import TimeMarker

    assert TimeMarker.__name__ == "TimeMarker"


def test_time_marker_module_is_timer_module():
    from timemarker import TimeMarker

    assert TimeMarker.__module__ == "timemarker.timer"


def test_package_module_dict_contains_time_marker():
    import timemarker

    assert timemarker.__dict__["TimeMarker"] is timemarker.TimeMarker


def test_multiple_imports_return_same_object():
    import timemarker

    first = timemarker.TimeMarker
    second = importlib.import_module("timemarker").TimeMarker

    assert first is second


def test_reload_keeps_time_marker_identity():
    import timemarker

    before = timemarker.TimeMarker
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is before


def test_from_import_after_reload_still_works():
    import timemarker

    importlib.reload(timemarker)
    from timemarker import TimeMarker

    assert TimeMarker is timemarker.TimeMarker


def test_import_module_returns_same_reexported_class():
    module = importlib.import_module("timemarker")

    assert getattr(module, "TimeMarker") is module.TimeMarker


def test_import_fails_for_python_below_36(monkeypatch):
    import timemarker

    monkeypatch.setattr(sys, "version_info", (3, 5, 0))
    with pytest.raises(AssertionError, match="python >= 3\\.6"):
        importlib.reload(timemarker)


def test_import_succeeds_for_python_36(monkeypatch):
    import timemarker

    monkeypatch.setattr(sys, "version_info", (3, 6, 0))
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is timemarker.TimeMarker


def test_import_succeeds_for_python_newer_version(monkeypatch):
    import timemarker

    monkeypatch.setattr(sys, "version_info", (3, 8, 0))
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is timemarker.TimeMarker


def test_time_marker_access_through_package_attribute_is_stable():
    import timemarker

    first = timemarker.TimeMarker
    second = timemarker.TimeMarker

    assert first is second
