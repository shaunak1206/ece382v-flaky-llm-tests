"""
Unit tests for:
  - rambutan3/check_args/annotation/INT.py (INT)
  - rambutan3/check_args/RCheckArgsError.py (RCheckArgsError)
  - rambutan3/check_args/base/RNotNoneTypeMatcher.py (RNotNoneTypeMatcher)
  - rambutan3/__init__.py (package import)

RTestUtil is not covered: its source file was not provided, so its API is unknown.

None of the code under test uses random numbers, the clock, hash() of str/bytes,
or module-level mutable state that is changed here, so no seeding or monkeypatching
is required. All assertions are exact and deterministic.
"""

from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _RecordingTypeMatcherError:
    """Minimal stand-in for RTypeMatcherError that records add_failed_match() calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


class _MyInt(int):
    """Subclass of int (but not of bool) used to check INT accepts int subclasses."""
    pass


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_matches_positive_int():
    assert INT.matches(1) is True
    assert INT.matches(42) is True


def test_int_matches_zero_and_negative_int():
    assert INT.matches(0) is True
    assert INT.matches(-7) is True


def test_int_rejects_true():
    # bool is a subclass of int, but INT must reject it explicitly.
    assert INT.matches(True) is False


def test_int_rejects_false():
    assert INT.matches(False) is False


def test_int_rejects_float():
    assert INT.matches(1.0) is False
    assert INT.matches(0.5) is False


def test_int_rejects_str_none_and_list():
    assert INT.matches("1") is False
    assert INT.matches(None) is False
    assert INT.matches([1]) is False


def test_int_matches_int_subclass():
    assert INT.matches(_MyInt(3)) is True


def test_int_matches_large_int():
    assert INT.matches(10 ** 100) is True
    assert INT.matches(-(10 ** 100)) is True


def test_int_matches_is_repeatable():
    # Repeated calls with the same input must return the same result.
    for _ in range(3):
        assert INT.matches(5) is True
        assert INT.matches(True) is False


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_carries_message():
    try:
        raise RCheckArgsError("bad argument")
    except RCheckArgsError as e:
        assert str(e) == "bad argument"
        assert e.args == ("bad argument",)
    else:
        raise AssertionError("RCheckArgsError was not raised")


def test_check_args_error_not_caught_by_value_error():
    # RCheckArgsError derives from Exception directly, not ValueError.
    assert not issubclass(RCheckArgsError, ValueError)


def test_check_args_error_without_message():
    err = RCheckArgsError()
    assert str(err) == ""
    assert err.args == ()


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_matches_falsy_non_none_values():
    m = RNotNoneTypeMatcher()
    for value in (0, "", False, [], {}, 0.0):
        assert m.matches(value) is True


def test_not_none_rejects_none():
    m = RNotNoneTypeMatcher()
    assert m.matches(None) is False


def test_not_none_none_without_error_collector():
    # Default matcher_error=None must not raise when the value is None.
    m = RNotNoneTypeMatcher()
    assert m.matches(None, None) is False


def test_not_none_records_failed_match_for_none():
    m = RNotNoneTypeMatcher()
    err = _RecordingTypeMatcherError()
    assert m.matches(None, err) is False
    assert len(err.calls) == 1
    matcher, value = err.calls[0]
    assert matcher is m
    assert value is None


def test_not_none_no_record_on_success():
    m = RNotNoneTypeMatcher()
    err = _RecordingTypeMatcherError()
    assert m.matches("value", err) is True
    assert err.calls == []


def test_not_none_eq_same_type():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()
    assert (RNotNoneTypeMatcher() == RNotNoneTypeMatcher()) is True


def test_not_none_neq_other_types():
    m = RNotNoneTypeMatcher()
    assert (m == "not None") is False
    assert (m == None) is False  # noqa: E711
    assert (m == INT) is False


def test_not_none_hash_is_defined():
    # __eq__ is overridden, so __hash__ must be explicitly defined to stay hashable.
    # Only the class attribute is checked here; calling hash() on the instance depends
    # on the current super(object, self) implementation, which is not asserted.
    assert RNotNoneTypeMatcher.__hash__ is not None


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_is_abstract_type_matcher_subclass():
    assert issubclass(RNotNoneTypeMatcher, RAbstractTypeMatcher)


# ---------------------------------------------------------------------------
# rambutan3 package
# ---------------------------------------------------------------------------

def test_package_imports():
    import rambutan3

    assert rambutan3 is not None
    assert rambutan3.__file__.endswith("__init__.py")
