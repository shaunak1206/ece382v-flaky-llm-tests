import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError

# Tests for INT (RRestrictedInstanceMatcher configured for int but not bool)

def test_INT_matches_int():
    assert INT.matches(123) is True

def test_INT_does_not_match_bool():
    # bool is subclass of int but DISALLOWED
    assert INT.matches(True) is False
    assert INT.matches(False) is False

def test_INT_does_not_match_str():
    assert INT.matches("123") is False

def test_INT_does_not_match_float():
    assert INT.matches(3.14) is False

def test_INT_matches_only_int_not_bool_with_matcher_error():
    from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError
    matcher_error = RTypeMatcherError()
    # For int, no error added
    assert INT.matches(5, matcher_error) is True
    assert len(matcher_error.failed_matches) == 0
    # For bool, error added
    matcher_error = RTypeMatcherError()
    assert INT.matches(True, matcher_error) is False
    assert len(matcher_error.failed_matches) == 1

# RCheckArgsError tests

def test_RCheckArgsError_is_exception():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError()

def test_RCheckArgsError_can_be_instantiated_with_message():
    err = RCheckArgsError("error message")
    assert isinstance(err, Exception)
    assert str(err) == "error message"

# RNotNoneTypeMatcher tests

def test_RNotNoneTypeMatcher_matches_non_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True

def test_RNotNoneTypeMatcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False

def test_RNotNoneTypeMatcher_adds_error_when_no_match_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    matcher_error = RTypeMatcherError()
    result = matcher.matches(None, matcher_error)
    assert result is False
    # ensure matcher_error recorded failure
    assert len(matcher_error.failed_matches) == 1
    # The failed_match info contains the matcher and the value None
    fm = matcher_error.failed_matches[0]
    assert fm.matcher == matcher
    assert fm.value is None

def test_RNotNoneTypeMatcher_does_not_add_error_when_match_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    matcher_error = RTypeMatcherError()
    result = matcher.matches(123, matcher_error)
    assert result is True
    assert len(matcher_error.failed_matches) == 0

def test_RNotNoneTypeMatcher_eq_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()

    # __eq__
    assert matcher1 == matcher2
    assert matcher1 == matcher1
    assert matcher1 != object()
    # __hash__
    h1 = hash(matcher1)
    h2 = hash(matcher2)
    assert isinstance(h1, int)
    assert h1 == h2

def test_RNotNoneTypeMatcher_str():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == 'not None'

# This test covers __hash__ calling super(object, self).__hash__()
def test_RNotNoneTypeMatcher_hash_consistent_with_object_hash():
    matcher = RNotNoneTypeMatcher()
    # Since super(object, self).__hash__ calls object.__hash__ on self,
    # the hash is some integer id-based hash
    h = hash(matcher)
    assert isinstance(h, int)

# Additional sanity check for INT's allowed and not allowed types attributes
def test_INT_allowed_and_not_allowed_attrs():
    # The class holds these attributes as defined in RRestrictedInstanceMatcher
    assert hasattr(INT, 'allowed_class_or_type_non_empty_tuple')
    assert isinstance(INT.allowed_class_or_type_non_empty_tuple, tuple)
    assert int in INT.allowed_class_or_type_non_empty_tuple
    assert hasattr(INT, 'not_allowed_class_or_type_iterable')
    assert bool in INT.not_allowed_class_or_type_iterable

# Check that INT rejects None
def test_INT_rejects_none():
    assert INT.matches(None) is False

# Check that RNotNoneTypeMatcher __eq__ returns False for unrelated object
def test_RNotNoneTypeMatcher_eq_with_unrelated_object():
    matcher = RNotNoneTypeMatcher()
    assert not (matcher == 42)
