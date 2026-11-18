import importlib
import sys

import pytest


def _clear_timemarker_modules():
    """Remove timemarker modules so imports re-execute package __init__."""
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)


def test_package_import_exposes_timemarker():
    import timemarker

    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_timemarker_is_imported_from_timer_module():
    import timemarker
    from timemarker.timer import TimeMarker as TimerTimeMarker

    assert timemarker.TimeMarker is TimerTimeMarker


def test_time_marker_class_name_is_preserved():
    from timemarker import TimeMarker

    assert TimeMarker.__name__ == "TimeMarker"


def test_time_marker_class_module_is_timer():
    from timemarker import TimeMarker

    assert TimeMarker.__module__ == "timemarker.timer"


def test_package_does_not_define_extra_public_symbols_for_timer_export():
    import timemarker

    assert "TimeMarker" in timemarker.__dict__
    assert timemarker.__dict__["TimeMarker"] is timemarker.TimeMarker


def test_from_import_matches_package_attribute():
    import timemarker
    from timemarker import TimeMarker

    assert TimeMarker is timemarker.TimeMarker


def test_reimport_keeps_same_time_marker_object():
    import timemarker

    first = timemarker.TimeMarker
    second = importlib.import_module("timemarker").TimeMarker

    assert first is second


def test_reload_keeps_same_time_marker_object():
    import timemarker

    first = timemarker.TimeMarker
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is first


def test_import_succeeds_for_supported_python_version(monkeypatch):
    _clear_timemarker_modules()
    import sys as _sys

    monkeypatch.setattr(_sys, "version_info", (3, 6))
    module = importlib.import_module("timemarker")

    assert hasattr(module, "TimeMarker")
    assert module.TimeMarker is not None


def test_import_succeeds_for_python_version_above_minimum(monkeypatch):
    _clear_timemarker_modules()
    import sys as _sys

    monkeypatch.setattr(_sys, "version_info", (3, 8, 0))
    module = importlib.import_module("timemarker")

    assert module.TimeMarker is not None


def test_import_fails_for_unsupported_python_version(monkeypatch):
    _clear_timemarker_modules()
    import sys as _sys

    monkeypatch.setattr(_sys, "version_info", (3, 5, 9))

    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.import_module("timemarker")


def test_import_fails_for_exactly_python_35(monkeypatch):
    _clear_timemarker_modules()
    import sys as _sys

    monkeypatch.setattr(_sys, "version_info", (3, 5))

    with pytest.raises(AssertionError):
        importlib.import_module("timemarker")
