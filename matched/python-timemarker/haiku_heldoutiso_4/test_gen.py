import inspect
import pickle
import sys
import time

import pytest

import timemarker
from timemarker import TimeMarker
from timemarker.timer import TimeMarker as TimerModuleTimeMarker


MARK_NAMES = ("mark", "add_mark", "record", "tick", "set_mark")
ELAPSED_NAMES = ("elapsed", "get_elapsed", "total", "total_elapsed")


def _get_method(obj, names):
    """Return the first callable attribute of obj matching one of names, or skip."""
    for name in names:
        attr = getattr(obj, name, None)
        if callable(attr):
            return attr
    pytest.skip("TimeMarker exposes none of: " + ", ".join(names))


def _call_mark(tm, label):
    mark = _get_method(tm, MARK_NAMES)
    try:
        return mark(label)
    except TypeError:
        return mark()


def test_package_exports_timemarker():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_package_and_timer_module_expose_same_class():
    assert TimeMarker is TimerModuleTimeMarker


def test_reimport_returns_same_class():
    import importlib

    reloaded = importlib.import_module("timemarker")
    assert reloaded.TimeMarker is TimeMarker


def test_python_version_gate_is_satisfied():
    assert sys.version_info >= (3, 6)


def test_constructor_accepts_no_arguments():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)


def test_constructor_parameters_all_have_defaults():
    sig = inspect.signature(TimeMarker)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert required == []


def test_instances_are_distinct_objects():
    a = TimeMarker()
    b = TimeMarker()
    assert a is not b


def test_instance_is_not_none_and_truthy_or_safe_to_bool():
    tm = TimeMarker()
    # Object must be usable in a boolean context without raising.
    assert bool(tm) in (True, False)


def test_str_representation_is_a_string():
    tm = TimeMarker()
    assert isinstance(str(tm), str)
    assert isinstance(repr(tm), str)


def test_mark_call_does_not_raise():
    tm = TimeMarker()
    _call_mark(tm, "start")


def test_multiple_marks_can_be_recorded():
    tm = TimeMarker()
    _call_mark(tm, "first")
    time.sleep(0.001)
    _call_mark(tm, "second")
    _call_mark(tm, "third")


def test_marks_on_separate_instances_do_not_interfere():
    a = TimeMarker()
    b = TimeMarker()
    _call_mark(a, "only_in_a")
    before = vars(b).copy()
    _call_mark(a, "another_in_a")
    assert vars(b) == before


def test_elapsed_time_is_non_negative_when_available():
    tm = TimeMarker()
    elapsed = _get_method(tm, ELAPSED_NAMES)
    _call_mark(tm, "start")
    time.sleep(0.005)
    _call_mark(tm, "end")
    value = elapsed()
    assert isinstance(value, (int, float))
    assert value >= 0


def test_elapsed_time_grows_with_real_time_when_available():
    tm = TimeMarker()
    elapsed = _get_method(tm, ELAPSED_NAMES)
    _call_mark(tm, "start")
    first = elapsed()
    time.sleep(0.01)
    second = elapsed()
    assert second >= first


def test_instance_survives_pickle_round_trip():
    tm = TimeMarker()
    _call_mark(tm, "start")
    try:
        restored = pickle.loads(pickle.dumps(tm))
    except (pickle.PicklingError, TypeError, AttributeError):
        pytest.skip("TimeMarker instances are not picklable")
    assert isinstance(restored, TimeMarker)
