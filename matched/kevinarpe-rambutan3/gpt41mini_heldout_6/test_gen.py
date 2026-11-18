import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


class DummyMatcherError(RTypeMatcherError):
    def __init__(self):
        self.failed_matches = []

    def add_failed_match(self, matcher, value):
        self.failed_matches.append((matcher, value))


def test_RCheckArgsError_is_subclass_of_Exception():
    e = RCheckArgsError("error")
    assert isinstance(e, Exception)
    assert str(e) == "error"


@pytest.mark.parametrize("value,expected", [
    (1, True),
    (0, True),
    (-5, True),
    (123456789, True),
])
def test_INT_matches_int(value, expected):
    # INT should match int values (but not bool)
    assert INT.matches(value) == expected


@pytest.mark.parametrize("value", [True, False])
def test_INT_does_not_match_bool(value):
    # bool is subclass of int, but INT matcher excludes bool
    assert not INT.matches(value)


@pytest.mark.parametrize("value", ["string", None, 3.14, [], {}, object()])
def test_INT_does_not_match_other_types(value):
    assert not INT.matches(value)


def test_RNotNoneTypeMatcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(1) is True
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True


def test_RNotNoneTypeMatcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_RNotNoneTypeMatcher_adds_failed_match_when_none_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()
    result = matcher.matches(None, matcher_error=error)
    assert result is False
    assert len(error.failed_matches) == 1
    failed_matcher, failed_value = error.failed_matches[0]
    assert failed_matcher == matcher
    assert failed_value is None


def test_RNotNoneTypeMatcher_does_not_add_failed_match_when_non_none_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()
    result = matcher.matches(42, matcher_error=error)
    assert result is True
    assert error.failed_matches == []


def test_RNotNoneTypeMatcher_equality_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    assert matcher1 == matcher2
    # Inequality with other types
    assert matcher1 != 42
    assert matcher1 != object()
    # Hash returns int
    h = hash(matcher1)
    assert isinstance(h, int)


def test_RNotNoneTypeMatcher_string_representation():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_INT_repr_and_str_methods_exist():
    # RRestrictedInstanceMatcher class normally has __str__ and __repr__?
    # We check that INT has __str__ and __repr__ callable and returns string.
    s = str(INT)
    r = repr(INT)
    assert isinstance(s, str)
    assert isinstance(r, str)


def test_INT_allowed_and_not_allowed_classes():
    # Check internally that INT allowed classes include int but not bool
    assert int in INT.allowed_class_or_type_non_empty_tuple
    assert bool in INT.not_allowed_class_or_type_iterable


def test_INT_matches_callable():
    # matches should accept matcher_error but behave the same - no error collection here
    assert INT.matches(42) is True
    assert INT.matches(True) is False
    # Passing matcher_error should not break (even if unused)
    assert INT.matches(42, matcher_error=None) is True


def test_RCheckArgsError_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("fail")


def test_RNotNoneTypeMatcher_eq_false_for_other_types():
    matcher = RNotNoneTypeMatcher()
    assert matcher != "not a matcher"
    assert matcher != 123
    assert matcher != None


def test_RNotNoneTypeMatcher_matches_with_explicit_none_and_no_error_returns_false():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_RNotNoneTypeMatcher_matches_with_explicit_none_and_error_adds_failed_match():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()
    assert matcher.matches(None, matcher_error=error) is False
    assert error.failed_matches == [(matcher, None)]
