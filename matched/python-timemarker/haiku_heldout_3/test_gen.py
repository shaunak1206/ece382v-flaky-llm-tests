import importlib
import inspect
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def _reload_package():
    """Re-execute timemarker/__init__.py with the current sys.version_info."""
    return importlib.reload(timemarker)


def test_package_imports_successfully():
    assert timemarker is not None


def test_timemarker_is_exported_from_package():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_is_callable():
    assert callable(TimeMarker)


def test_timemarker_class_name():
    assert TimeMarker.__name__ == "TimeMarker"


def test_timer_submodule_is_importable():
    timer = importlib.import_module("timemarker.timer")
    assert hasattr(timer, "TimeMarker")


def test_package_export_is_same_object_as_timer_class():
    timer = importlib.import_module("timemarker.timer")
    assert timemarker.TimeMarker is timer.TimeMarker


def test_timemarker_defined_in_timer_module():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_package_attribute_matches_direct_import():
    assert timemarker.TimeMarker is TimeMarker


def test_reload_is_stable_and_keeps_same_class():
    before = TimeMarker
    reloaded = _reload_package()
    assert reloaded.TimeMarker is before


def test_version_check_passes_on_current_interpreter():
    # The current interpreter satisfies the >= 3.6 requirement, so reload must succeed.
    assert sys.version_info >= (3, 6)
    _reload_package()


def test_version_check_passes_at_minimum_supported_version(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 0, "final", 0))
    try:
        _reload_package()
    finally:
        monkeypatch.undo()
        _reload_package()


def test_version_check_fails_below_minimum(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 9, "final", 0))
    try:
        with pytest.raises(AssertionError):
            _reload_package()
    finally:
        monkeypatch.undo()
        _reload_package()


def test_version_check_error_message(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (2, 7, 18, "final", 0))
    try:
        with pytest.raises(AssertionError, match="python >= 3.6"):
            _reload_package()
    finally:
        monkeypatch.undo()
        _reload_package()


def test_timemarker_signature_is_introspectable():
    try:
        inspect.signature(TimeMarker)
    except (TypeError, ValueError):
        pytest.skip("TimeMarker signature is not introspectable")


def test_timemarker_can_be_instantiated_without_arguments():
    try:
        sig = inspect.signature(TimeMarker)
        sig.bind()
    except (TypeError, ValueError):
        pytest.skip("TimeMarker requires constructor arguments")
    instance = TimeMarker()
    assert isinstance(instance, TimeMarker)


def test_timemarker_instances_are_distinct_objects():
    try:
        inspect.signature(TimeMarker).bind()
    except (TypeError, ValueError):
        pytest.skip("TimeMarker requires constructor arguments")
    first = TimeMarker()
    second = TimeMarker()
    assert first is not second
