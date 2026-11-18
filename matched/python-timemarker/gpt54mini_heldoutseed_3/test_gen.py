import importlib
import sys

import pytest


def _fresh_import_timemarker():
    """Import timemarker after removing it from sys.modules."""
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)
    return importlib.import_module("timemarker")


def test_package_import_exposes_timemarker():
    tm = _fresh_import_timemarker()
    assert hasattr(tm, "TimeMarker")


def test_time_marker_is_same_object_as_timer_module_class():
    tm = _fresh_import_timemarker()
    timer_module = importlib.import_module("timemarker.timer")
    assert tm.TimeMarker is timer_module.TimeMarker


def test_from_package_import_works():
    _fresh_import_timemarker()
    from timemarker import TimeMarker

    timer_module = importlib.import_module("timemarker.timer")
    assert TimeMarker is timer_module.TimeMarker


def test_time_marker_name_is_preserved():
    tm = _fresh_import_timemarker()
    assert tm.TimeMarker.__name__ == "TimeMarker"


def test_time_marker_module_is_timer():
    tm = _fresh_import_timemarker()
    assert tm.TimeMarker.__module__ == "timemarker.timer"


def test_package_attribute_points_to_timer_class():
    tm = _fresh_import_timemarker()
    assert getattr(tm, "TimeMarker") is tm.timer.TimeMarker


def test_time_marker_is_class_like():
    tm = _fresh_import_timemarker()
    assert isinstance(tm.TimeMarker, type)


def test_time_marker_is_in_module_dict():
    tm = _fresh_import_timemarker()
    assert "TimeMarker" in tm.__dict__


def test_importing_timemarker_does_not_create_extra_public_names():
    tm = _fresh_import_timemarker()
    public_names = {name for name in tm.__dict__ if not name.startswith("_")}
    assert "TimeMarker" in public_names
    assert "version_info" in public_names or "version_info" not in public_names


def test_reloading_package_keeps_same_time_marker_object():
    tm = _fresh_import_timemarker()
    timer_module = importlib.import_module("timemarker.timer")
    before = tm.TimeMarker
    tm2 = importlib.reload(tm)
    assert tm2.TimeMarker is before
    assert tm2.TimeMarker is timer_module.TimeMarker


def test_importing_package_under_supported_version_succeeds(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 8, 0))
    tm = _fresh_import_timemarker()
    assert tm.TimeMarker is not None


def test_importing_package_under_minimum_supported_version_succeeds(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 0))
    tm = _fresh_import_timemarker()
    assert tm.TimeMarker is not None


def test_importing_package_with_version_below_requirement_raises_assertion_error(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 9))
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.import_module("timemarker")


def test_importing_package_after_failed_import_can_succeed_again(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 9))
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)
    with pytest.raises(AssertionError):
        importlib.import_module("timemarker")

    monkeypatch.setattr(sys, "version_info", (3, 8, 0))
    tm = _fresh_import_timemarker()
    assert tm.TimeMarker is not None
