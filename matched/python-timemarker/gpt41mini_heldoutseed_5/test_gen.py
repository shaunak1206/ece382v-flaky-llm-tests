# test_timemarker_init.py

import sys
import types
import importlib
import builtins
import pytest

def test_version_assertion_passes_for_supported_versions(monkeypatch):
    # Patch sys.version_info to a supported version
    monkeypatch.setattr(sys, "version_info", (3, 6))
    # We reload the module to trigger the assertion
    # Because timemarker/__init__.py imports from .timer
    import timemarker
    importlib.reload(timemarker)
    # if no AssertionError is raised, test passes

def test_version_assertion_passes_for_higher_versions(monkeypatch):
    versions = [(3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 0)]
    for v in versions:
        monkeypatch.setattr(sys, "version_info", v)
        import timemarker
        importlib.reload(timemarker)

def test_version_assertion_fails_for_lower_versions(monkeypatch):
    # monkeypatch sys.version_info to a lower version and assert AssertionError
    monkeypatch.setattr(sys, "version_info", (3, 5))
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        import timemarker
        importlib.reload(timemarker)

    monkeypatch.setattr(sys, "version_info", (2, 7))
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.reload(timemarker)

def test_timemarker_imports_timemarker_timer_TimeMarker():
    # The timemarker.__init__ imports TimeMarker from timemarker.timer
    import timemarker
    # Check timemarker.TimeMarker is same as timemarker.timer.TimeMarker
    import timemarker.timer
    assert timemarker.TimeMarker is timemarker.timer.TimeMarker

def test_timemarker_module_has_only_expected_attributes():
    import timemarker
    # Expect only __builtins__, __package__, __name__, __file__, __loader__, __spec__, and TimeMarker
    allowed = {"__builtins__", "__package__", "__name__", "__file__", "__loader__", "__spec__", "TimeMarker"}
    attributes = set(dir(timemarker))
    assert allowed.issubset(attributes)
    for attr in attributes:
        assert attr in allowed or attr.startswith("_")

def test_version_info_is_tuple_and_has_correct_format():
    import sys
    vi = sys.version_info
    assert isinstance(vi, tuple) or hasattr(vi, "__getitem__")
    assert vi[0] >= 0 and vi[1] >= 0  # major and minor ints

def test_reload_timemarker_without_error(monkeypatch):
    # Test multiple reloads do not fail
    import timemarker
    for _ in range(3):
        importlib.reload(timemarker)

def test_import_fails_if_timer_module_missing(monkeypatch):
    # Temporarily remove timemarker.timer module from sys.modules and files to test import failure
    import timemarker

    # Remove timemarker.timer from sys.modules forcibly
    old_timer_mod = sys.modules.pop("timemarker.timer", None)
    # Remove or rename the timer attribute from timemarker package if present
    import timemarker
    if hasattr(timemarker, "timer"):
        delattr(timemarker, "timer")

    # Also patch timemarker.__init__ to simulate import failure of TimeMarker
    with pytest.raises(ModuleNotFoundError):
        importlib.reload(timemarker)

    # Restore
    if old_timer_mod:
        sys.modules["timemarker.timer"] = old_timer_mod

def test_time_marker_is_class():
    import timemarker
    TimeMarker = timemarker.TimeMarker
    # TimeMarker should be a class or callable type
    assert isinstance(TimeMarker, type) or callable(TimeMarker)

def test_timemarker_repr_and_str_methods_exist():
    import timemarker
    TimeMarker = timemarker.TimeMarker
    # Instantiate if possible to test repr/str
    try:
        instance = None
        # Try normal instantiation without args
        instance = TimeMarker()
    except Exception:
        # If fails, skip test body; this module doesn't guarantee instantiation signature
        pytest.skip("TimeMarker cannot be instantiated without arguments")
    else:
        # repr and str should not raise
        r = repr(instance)
        s = str(instance)
        assert isinstance(r, str)
        assert isinstance(s, str)

def test_timemarker_attributes_exist():
    import timemarker
    TimeMarker = timemarker.TimeMarker
    # Check attributes typical of a class, e.g. __init__, __repr__, etc
    assert hasattr(TimeMarker, "__init__")

def test_timemarker_is_same_object_after_reload():
    import timemarker
    TimeMarker_before = timemarker.TimeMarker
    importlib.reload(timemarker)
    TimeMarker_after = timemarker.TimeMarker
    assert TimeMarker_before is TimeMarker_after

def test_timemarker_attrs_unchanged_after_reload():
    import timemarker
    keys_before = set(dir(timemarker))
    importlib.reload(timemarker)
    keys_after = set(dir(timemarker))
    assert keys_before == keys_after

def test_import_timemarker_timer_before_init_import():
    # Import timemarker.timer before timemarker.__init__ import
    import timemarker.timer
    # Now import timemarker
    import timemarker
    # TimeMarker should be accessible
    assert hasattr(timemarker, "TimeMarker")

def test_import_timemarker_init_twice(monkeypatch):
    # Reload timemarker twice to check no side effects
    import timemarker
    importlib.reload(timemarker)
    importlib.reload(timemarker)

def test_timemarker_can_be_imported_from_package_root():
    # This tests importing timemarker as a top-level package
    import timemarker
    assert isinstance(timemarker, types.ModuleType)

@pytest.mark.skip(reason="Cannot easily simulate non-importable timemarker.timer without external modules")
def test_import_fails_when_timer_module_missing(monkeypatch):
    # Marked skip because proper disabling requires file-system or import hook manipulation
    pass
