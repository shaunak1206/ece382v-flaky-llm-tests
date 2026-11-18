import pytest
import builtins
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


# Tests for INT (RRestrictedInstanceMatcher)
def test_INT_matches_int():
    assert INT.matches(0)
    assert INT.matches(42)
    assert INT.matches(-1)
    assert INT.matches(int(5))

def test_INT_does_not_match_bool():
    # bool is a subclass of int, but should not match
    assert not INT.matches(True)
    assert not INT.matches(False)

def test_INT_does_not_match_str_or_float():
    assert not INT.matches("string")
    assert not INT.matches(3.14)
    assert not INT.matches(None)
    assert not INT.matches([])

def test_INT_str_representation():
    # RRestrictedInstanceMatcher has __str__ returning a formatted string or class representation;
    # Check it mentions int but not bool.
    s = str(INT)
    assert 'int' in s
    assert 'bool' not in s  # not_allowed_class_or_type should not appear


# Tests for RCheckArgsError (subclass of Exception)
def test_RCheckArgsError_is_exception_subclass():
    exc = RCheckArgsError()
    assert isinstance(exc, Exception)
    # it can be raised and caught
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError()

def test_RCheckArgsError_message():
    msg = "some error"
    with pytest.raises(RCheckArgsError) as e:
        raise RCheckArgsError(msg)
    # Exception stores message in args[0]
    assert e.value.args[0] == msg


# Tests for RNotNoneTypeMatcher
def test_RNotNoneTypeMatcher_matches_non_none():
    matcher = RNotNoneTypeMatcher()
    # matches any non-None value
    assert matcher.matches(0)
    assert matcher.matches("")
    assert matcher.matches([])
    assert matcher.matches(False)  # False is not None
    assert matcher.matches(object())

def test_RNotNoneTypeMatcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert not matcher.matches(None)

def test_RNotNoneTypeMatcher_matches_adds_error_on_none():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError.RTypeMatcherError()
    # The error should track failed matches
    assert not matcher.matches(None, error)
    # error should contain this failure
    failures = error.failed_matches()
    assert any(fm.matcher == matcher and fm.value is None for fm in failures)

def test_RNotNoneTypeMatcher_matches_does_not_add_error_on_match():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError.RTypeMatcherError()
    # Some value not None should not add a failure
    assert matcher.matches("not none", error)
    failures = error.failed_matches()
    assert all(fm.value is None for fm in failures) or len(failures) == 0  # No failure for this value

def test_RNotNoneTypeMatcher_eq_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    assert matcher1 == matcher2
    # Comparing to other types returns False
    assert matcher1 != object()
    # hash must work without error and be int
    h = hash(matcher1)
    assert isinstance(h, int)
    # Equal objects must have equal hash
    assert h == hash(matcher2)

def test_RNotNoneTypeMatcher_str():
    matcher = RNotNoneTypeMatcher()
    s = str(matcher)
    assert isinstance(s, str)
    assert s == 'not None'


# Additional sanity tests ensuring no unexpected exceptions
def test_INT_matches_handles_edge_cases():
    # Passing unusual types to check robustness
    assert not INT.matches(None)
    class DummyIntSubclass(int):
        pass
    assert INT.matches(DummyIntSubclass(5))
    assert not INT.matches(bool(True))

def test_RNotNoneTypeMatcher_matches_without_error_obj():
    matcher = RNotNoneTypeMatcher()
    # When matcher_error=None, no exception and just return bool
    assert matcher.matches(123, matcher_error=None)
    assert not matcher.matches(None, matcher_error=None)

