import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError

# Tests for INT (RRestrictedInstanceMatcher configured for int without bool)
def test_INT_matches_int():
    assert INT.matches(42) is True
    assert INT.matches(-10) is True
    assert INT.matches(0) is True

def test_INT_does_not_match_bool():
    assert INT.matches(True) is False
    assert INT.matches(False) is False

def test_INT_does_not_match_other_types():
    assert INT.matches(3.14) is False
    assert INT.matches("100") is False
    assert INT.matches(None) is False
    assert INT.matches([]) is False


# Tests for RCheckArgsError exception
def test_rcheckargserror_is_exception():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError()

def test_rcheckargserror_can_hold_message():
    msg = "error occurred"
    with pytest.raises(RCheckArgsError, match=msg):
        raise RCheckArgsError(msg)


# Tests for RNotNoneTypeMatcher
def test_rnotnonetypematcher_matches_not_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(False) is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True

def test_rnotnonetypematcher_does_not_match_none_without_error():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False

def test_rnotnonetypematcher_does_not_match_none_with_error():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()
    assert matcher.matches(None, matcher_error=error) is False
    assert len(error.failed_matches) == 1
    failed_self, failed_value = error.failed_matches[0]
    assert failed_self == matcher
    assert failed_value is None

def test_rnotnonetypematcher_eq_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    class Dummy:
        pass
    dummy = Dummy()

    assert matcher1 == matcher2
    assert matcher2 == matcher1
    assert matcher1 != dummy

    # Check hash returns int without error (exact hash value arbitrary)
    assert isinstance(hash(matcher1), int)
    assert isinstance(hash(matcher2), int)
    # Hashes can differ but should be consistent across calls for the same object
    h1 = hash(matcher1)
    h2 = hash(matcher1)
    assert h1 == h2

def test_rnotnonetypematcher_str():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == 'not None'


# Additional tests on INT.matches with matcher_error (inferred behavior)
def test_INT_matches_with_matcher_error(monkeypatch):
    error = RTypeMatcherError()

    # INT does not provide matches with matcher_error argument, so test default
    # Accept call without error argument
    assert INT.matches(123) is True
    assert INT.matches(False) is False

# Test RCheckArgsError inheritance
def test_rcheckargserror_is_subclass_of_exception():
    assert issubclass(RCheckArgsError, Exception)

# Test RNotNoneTypeMatcher equality with unrelated types
def test_rnotnonetypematcher_eq_with_other_types():
    matcher = RNotNoneTypeMatcher()
    assert matcher != None
    assert matcher != 42
    assert matcher != "string"
    assert matcher != object()

# Test RNotNoneTypeMatcher matches with explicit None matcher_error argument
def test_rnotnonetypematcher_matches_without_matcher_error_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(100, matcher_error=None) is True
    assert matcher.matches(None, matcher_error=None) is False

# Test RNotNoneTypeMatcher matches adds one error only
def test_rnotnonetypematcher_matches_adds_only_one_error():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()
    ret1 = matcher.matches(None, matcher_error=error)
    ret2 = matcher.matches(None, matcher_error=error)
    assert ret1 is False
    assert ret2 is False
    assert len(error.failed_matches) == 2  # called twice -> two errors

# Test RTypeMatcherError basic behavior for completeness (it is used by RNotNoneTypeMatcher)
def test_rtypematchererror_add_failed_match_and_repr():
    error = RTypeMatcherError()
    matcher = RNotNoneTypeMatcher()
    value = None
    error.add_failed_match(matcher, value)
    assert len(error.failed_matches) == 1
    # __repr__ produces a string containing failed_matches info
    r = repr(error)
    assert "failed_matches" in r
    assert str(matcher) in r

