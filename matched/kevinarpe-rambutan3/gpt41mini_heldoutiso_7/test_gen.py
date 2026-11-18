import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


# Tests for INT (RRestrictedInstanceMatcher)
def test_int_matches_int():
    assert INT.matches(42)

def test_int_does_not_match_bool_true():
    assert not INT.matches(True)

def test_int_does_not_match_bool_false():
    assert not INT.matches(False)

def test_int_does_not_match_str():
    assert not INT.matches("42")

def test_int_does_not_match_float():
    assert not INT.matches(3.14)


# Tests for RCheckArgsError
def test_rcheck_argserror_is_exception_subclass():
    e = RCheckArgsError("error message")
    assert isinstance(e, Exception)
    assert str(e) == "error message"


# Helper class for testing RNotNoneTypeMatcher.matches with matcher_error
class DummyMatcherError:
    def __init__(self):
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


# Tests for RNotNoneTypeMatcher
def test_rnotnonetypematcher_matches_not_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0)
    assert matcher.matches("")
    assert matcher.matches([])
    assert matcher.matches(object())

def test_rnotnonetypematcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert not matcher.matches(None)

def test_rnotnonetypematcher_matches_none_with_error_reports_failure():
    matcher = RNotNoneTypeMatcher()
    dummy_error = DummyMatcherError()
    result = matcher.matches(None, dummy_error)
    assert result is False
    # Ensure the failed match was recorded
    assert len(dummy_error.failed) == 1
    recorded_matcher, recorded_value = dummy_error.failed[0]
    assert recorded_matcher == matcher
    assert recorded_value is None

def test_rnotnonetypematcher_matches_not_none_with_error_does_not_report_failure():
    matcher = RNotNoneTypeMatcher()
    dummy_error = DummyMatcherError()
    result = matcher.matches(123, dummy_error)
    assert result is True
    assert len(dummy_error.failed) == 0

def test_rnotnonetypematcher_eq_same_type():
    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()
    assert m1 == m2

def test_rnotnonetypematcher_eq_different_type():
    m = RNotNoneTypeMatcher()
    assert m != 42
    assert m != "not a matcher"
    assert m != object()

def test_rnotnonetypematcher_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    h = hash(matcher)
    assert isinstance(h, int)

def test_rnotnonetypematcher_str():
    matcher = RNotNoneTypeMatcher()
    s = str(matcher)
    assert s == "not None"

