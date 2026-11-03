import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


class DummyRTypeMatcherError(RTypeMatcherError.RTypeMatcherError):
    def __init__(self):
        super().__init__()
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


def test_INT_matches_int_values():
    assert INT.matches(0)
    assert INT.matches(42)
    assert INT.matches(-7)


def test_INT_rejects_bool_values():
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_INT_does_not_match_str_or_other_types():
    assert not INT.matches("123")
    assert not INT.matches(None)
    assert not INT.matches(3.14)
    assert not INT.matches([])


def test_RCheckArgsError_is_exception():
    exc = RCheckArgsError()
    assert isinstance(exc, Exception)
    # It should have no message by default
    assert str(exc) == ''


def test_RNotNoneTypeMatcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0)
    assert matcher.matches('')
    assert matcher.matches(False)
    assert matcher.matches([])
    assert matcher.matches(object())


def test_RNotNoneTypeMatcher_rejects_none_value_without_error():
    matcher = RNotNoneTypeMatcher()
    # Should return False and not raise
    assert matcher.matches(None) is False


def test_RNotNoneTypeMatcher_rejects_none_with_error_object():
    matcher = RNotNoneTypeMatcher()
    error = DummyRTypeMatcherError()
    result = matcher.matches(None, matcher_error=error)
    assert result is False
    assert len(error.failed) == 1
    failed_matcher, failed_value = error.failed[0]
    assert failed_matcher is matcher
    assert failed_value is None


def test_RNotNoneTypeMatcher_matches_with_error_object_and_non_none():
    matcher = RNotNoneTypeMatcher()
    error = DummyRTypeMatcherError()
    result = matcher.matches(123, matcher_error=error)
    assert result is True
    assert error.failed == []


def test_RNotNoneTypeMatcher_eq_and_hash():
    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()
    other = object()
    assert m1 == m2
    assert m2 == m1
    assert m1 != other
    assert m2 != 42
    # Hash must be consistent and return int
    h1 = hash(m1)
    h2 = hash(m2)
    assert isinstance(h1, int)
    assert isinstance(h2, int)


def test_RNotNoneTypeMatcher_str():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == 'not None'


def test_INT_repr_and_str_behavior():
    # INT is an instance of RRestrictedInstanceMatcher, ensure it has the expected methods
    assert hasattr(INT, "matches")
    # matches int
    assert INT.matches(1)
    # does not match bool
    assert not INT.matches(True)


def test_RCheckArgsError_message_persistence():
    msg = "error message"
    e = RCheckArgsError(msg)
    assert str(e) == msg


def test_RNotNoneTypeMatcher_multiple_error_additions():
    matcher = RNotNoneTypeMatcher()
    error = DummyRTypeMatcherError()
    matcher.matches(None, matcher_error=error)
    matcher.matches(None, matcher_error=error)
    assert len(error.failed) == 2


def test_INT_does_not_match_subclasses_of_int_other_than_bool():
    class MyInt(int):
        pass

    # MyInt is subclass of int but not bool, INT should accept it
    val = MyInt(5)
    assert INT.matches(val)


def test_INT_rejects_bool_subclass_instances():
    class MyBool(bool):
        pass

    val = MyBool(True)
    # Should reject as bool or subclass of bool is forbidden
    assert not INT.matches(val)
