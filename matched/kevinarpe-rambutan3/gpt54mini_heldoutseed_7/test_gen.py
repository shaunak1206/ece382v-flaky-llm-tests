import importlib

import pytest

from rambutan3 import __name__ as rambutan3_package_name
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeTypeMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_package_importable():
    module = importlib.import_module("rambutan3")
    assert module.__name__ == rambutan3_package_name


def test_int_matcher_accepts_regular_int():
    assert INT.matches(0)
    assert INT.matches(1)
    assert INT.matches(-123)


def test_int_matcher_rejects_bool_false():
    assert INT.matches(False) is False


def test_int_matcher_rejects_bool_true():
    assert INT.matches(True) is False


def test_int_matcher_rejects_non_int_value():
    assert INT.matches("123") is False
    assert INT.matches(1.5) is False
    assert INT.matches(None) is False


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError, match="bad args"):
        raise RCheckArgsError("bad args")


def test_rnotnonetypematcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True


def test_rnotnonetypematcher_rejects_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_rnotnonetypematcher_adds_failed_match_when_none_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    error = _FakeTypeMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_rnotnonetypematcher_does_not_call_error_for_non_none():
    matcher = RNotNoneTypeMatcher()
    error = _FakeTypeMatcherError()

    result = matcher.matches("x", error)

    assert result is True
    assert error.calls == []


def test_rnotnonetypematcher_equality_depends_on_type_only():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()

    assert matcher1 == matcher2
    assert matcher1 == matcher1
    assert matcher1 != object()
    assert matcher1 != None  # noqa: E711


def test_rnotnonetypematcher_hash_and_str():
    matcher = RNotNoneTypeMatcher()

    h = hash(matcher)
    assert isinstance(h, int)
    assert hash(matcher) == h
    assert str(matcher) == "not None"
