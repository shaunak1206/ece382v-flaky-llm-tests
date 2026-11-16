"""
Unit tests for:
  - rambutan3.check_args.annotation.INT (INT)
  - rambutan3.check_args.RCheckArgsError (RCheckArgsError)
  - rambutan3.check_args.base.RNotNoneTypeMatcher (RNotNoneTypeMatcher)
  - rambutan3/__init__.py (package import)

Note: RTestUtil is not defined in the provided source files (INT.py does not
contain it), so no tests are written for it here.
"""

import rambutan3
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _RecordingError:
    """Duck-typed stand-in for RTypeMatcherError that records add_failed_match calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


class _IntSubclass(int):
    pass


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_package_is_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_matches_positive_int():
    assert INT.matches(5)


def test_int_matches_negative_int():
    assert INT.matches(-3)


def test_int_matches_zero():
    assert INT.matches(0)


def test_int_matches_large_int():
    assert INT.matches(10 ** 30)


def test_int_rejects_true():
    assert not INT.matches(True)


def test_int_rejects_false():
    assert not INT.matches(False)


def test_int_rejects_float():
    assert not INT.matches(1.0)


def test_int_rejects_str():
    assert not INT.matches("1")


def test_int_rejects_none():
    assert not INT.matches(None)


def test_int_matches_int_subclass():
    assert INT.matches(_IntSubclass(7))


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    try:
        raise RCheckArgsError("bad argument")
    except RCheckArgsError as e:
        assert str(e) == "bad argument"
    else:
        raise AssertionError("RCheckArgsError was not raised")


def test_check_args_error_caught_as_exception():
    try:
        raise RCheckArgsError()
    except Exception as e:
        assert isinstance(e, RCheckArgsError)


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_matcher_is_abstract_type_matcher():
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)


def test_not_none_matcher_matches_non_none_values():
    m = RNotNoneTypeMatcher()
    assert m.matches(0)
    assert m.matches(False)
    assert m.matches("")
    assert m.matches([])
    assert m.matches(object())


def test_not_none_matcher_rejects_none():
    m = RNotNoneTypeMatcher()
    assert not m.matches(None)


def test_not_none_matcher_records_failure_for_none():
    m = RNotNoneTypeMatcher()
    err = _RecordingError()
    assert not m.matches(None, err)
    assert err.calls == [(m, None)]


def test_not_none_matcher_does_not_record_success():
    m = RNotNoneTypeMatcher()
    err = _RecordingError()
    assert m.matches(42, err)
    assert err.calls == []


def test_not_none_matcher_equals_other_instance():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_matcher_not_equal_to_int_matcher():
    assert not (RNotNoneTypeMatcher() == INT)


def test_not_none_matcher_not_equal_to_none_or_object():
    m = RNotNoneTypeMatcher()
    assert not (m == None)  # noqa: E711
    assert not (m == object())


def test_not_none_matcher_hash_is_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


def test_not_none_matcher_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
