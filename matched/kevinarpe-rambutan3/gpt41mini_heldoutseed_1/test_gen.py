import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


class DummyMatcherError(RTypeMatcherError):
    """A Dummy subclass to test matcher_error behavior of RNotNoneTypeMatcher"""
    def __init__(self):
        super().__init__()
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


def test_RCheckArgsError_is_exception():
    e = RCheckArgsError("error message")
    assert isinstance(e, Exception)
    assert str(e) == "error message"


def test_INT_matches_int_but_not_bool():
    # INT is a RRestrictedInstanceMatcher for int but not bool
    assert INT.matches(10)
    assert INT.matches(-5)
    # bool is subclass of int but must not match
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_INT_str_representation_contains_int_and_bool():
    # The __str__ should mention int and bool classes (rough check)
    s = str(INT)
    assert 'int' in s or 'int' in repr(INT)
    assert 'bool' in s or 'bool' in repr(INT)


def test_RNotNoneTypeMatcher_matches_non_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0)
    assert matcher.matches('')
    assert matcher.matches([])
    assert matcher.matches(object())


def test_RNotNoneTypeMatcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert not matcher.matches(None)


def test_RNotNoneTypeMatcher_adds_failed_match_when_none_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    dummy_error = DummyMatcherError()

    result = matcher.matches(None, matcher_error=dummy_error)
    assert result is False
    # The dummy_error should have recorded the failed match
    assert len(dummy_error.failed) == 1
    failed_matcher, failed_value = dummy_error.failed[0]
    assert isinstance(failed_matcher, RNotNoneTypeMatcher)
    assert failed_value is None


def test_RNotNoneTypeMatcher_no_failed_match_added_when_value_not_none():
    matcher = RNotNoneTypeMatcher()
    dummy_error = DummyMatcherError()
    result = matcher.matches(42, matcher_error=dummy_error)
    assert result is True
    assert len(dummy_error.failed) == 0


def test_RNotNoneTypeMatcher_eq_true_for_other_instance():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    assert matcher1 == matcher2


def test_RNotNoneTypeMatcher_eq_false_for_other_types():
    matcher = RNotNoneTypeMatcher()
    assert matcher != 123
    assert matcher != "string"
    assert matcher != object()


def test_RNotNoneTypeMatcher_hash_is_callable_and_consistent():
    matcher = RNotNoneTypeMatcher()
    h1 = hash(matcher)
    h2 = hash(matcher)
    assert isinstance(h1, int)
    assert h1 == h2


def test_RNotNoneTypeMatcher_str_is_not_none_and_descriptive():
    matcher = RNotNoneTypeMatcher()
    s = str(matcher)
    assert s == "not None"
    assert isinstance(s, str)


def test_INT_repr_and_str_are_defined():
    # The RRestrictedInstanceMatcher should have __repr__ and __str__ defined
    s = str(INT)
    r = repr(INT)
    assert isinstance(s, str)
    assert isinstance(r, str)
    # The string repr should mention the class names int and bool somewhere
    assert 'int' in s or 'int' in r
    assert 'bool' in s or 'bool' in r

# Additional sanity test for INT.matches with other types
@pytest.mark.parametrize("val", [3.14, "string", [], {}, (), None, object()])
def test_INT_does_not_match_non_int_non_bool(val):
    assert not INT.matches(val)

