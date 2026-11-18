import pytest

from rambutan3 import __name__ as rambutan3_package_name
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class DummyMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_package_imports_successfully():
    assert rambutan3_package_name == "rambutan3"


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_str_roundtrip():
    err = RCheckArgsError("bad args")
    assert str(err) == "bad args"


def test_int_matcher_accepts_positive_int():
    assert INT.matches(123) is True


def test_int_matcher_accepts_negative_int():
    assert INT.matches(-456) is True


def test_int_matcher_rejects_bool_true():
    assert INT.matches(True) is False


def test_int_matcher_rejects_bool_false():
    assert INT.matches(False) is False


def test_int_matcher_rejects_non_int_type():
    assert INT.matches("123") is False
    assert INT.matches(12.3) is False
    assert INT.matches(None) is False


def test_not_none_matcher_accepts_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True


def test_not_none_matcher_rejects_none_and_reports_error():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_not_none_matcher_rejects_none_without_error_object():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_matcher_equality_works_for_same_type():
    left = RNotNoneTypeMatcher()
    right = RNotNoneTypeMatcher()

    assert left == right
    assert not (left != right)


def test_not_none_matcher_inequality_against_other_objects():
    matcher = RNotNoneTypeMatcher()

    assert matcher != object()
    assert matcher != None  # noqa: E711


def test_not_none_matcher_str_is_human_readable():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_not_none_matcher_hash_is_int():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)
