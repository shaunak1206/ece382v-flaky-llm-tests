import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class DummyTypeMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


def test_rcheckargserror_string_representation_contains_message():
    err = RCheckArgsError("something went wrong")
    assert "something went wrong" in str(err)


def test_rnotnonetypematcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True


def test_rnotnonetypematcher_does_not_match_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_rnotnonetypematcher_adds_failed_match_to_error_when_none():
    matcher = RNotNoneTypeMatcher()
    error = DummyTypeMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_rnotnonetypematcher_does_not_touch_error_when_match_succeeds():
    matcher = RNotNoneTypeMatcher()
    error = DummyTypeMatcherError()

    result = matcher.matches("x", error)

    assert result is True
    assert error.calls == []


def test_rnotnonetypematcher_equals_other_instance_of_same_class():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_rnotnonetypematcher_not_equal_to_other_objects():
    matcher = RNotNoneTypeMatcher()
    assert matcher != None
    assert matcher != object()


def test_rnotnonetypematcher_str_is_not_none():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_rnotnonetypematcher_hash_is_an_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


def test_int_matcher_matches_regular_ints():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-123456) is True


def test_int_matcher_does_not_match_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_matcher_does_not_match_non_int_values():
    assert INT.matches(3.14) is False
    assert INT.matches("5") is False
    assert INT.matches(None) is False


def test_int_matcher_string_representation_mentions_int():
    text = str(INT)
    assert "int" in text.lower()
    assert "bool" not in text.lower()


def test_int_matcher_only_accepts_int_not_subclasses_like_bool():
    # bool is a subclass of int in Python, but this matcher explicitly excludes it.
    assert issubclass(bool, int)
    assert INT.matches(True) is False
    assert INT.matches(False) is False
