import importlib

import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class DummyMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))

    def __bool__(self):
        return True


class FalseyMatcherError(DummyMatcherError):
    def __bool__(self):
        return False


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


def test_rcheckargserror_str_includes_message():
    err = RCheckArgsError("hello")
    assert "hello" in str(err)


def test_int_matches_regular_int_values():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-123) is True


def test_int_does_not_match_bool_true():
    assert INT.matches(True) is False


def test_int_does_not_match_bool_false():
    assert INT.matches(False) is False


def test_int_does_not_match_non_int_types():
    assert INT.matches(3.14) is False
    assert INT.matches("7") is False
    assert INT.matches(None) is False
    assert INT.matches([]) is False


def test_int_string_representation_mentions_int():
    text = str(INT)
    assert "int" in text.lower()


def test_int_matches_method_does_not_raise_when_matcher_error_is_none():
    assert INT.matches(5, None) is True
    assert INT.matches(True, None) is False


def test_not_none_type_matcher_matches_non_none_and_rejects_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(123) is True
    assert matcher.matches("x") is True
    assert matcher.matches(None) is False


def test_not_none_type_matcher_records_failed_match_when_truthy_error_passed():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    assert matcher.matches(None, err) is False
    assert err.calls == [(matcher, None)]


def test_not_none_type_matcher_does_not_record_failed_match_when_error_is_falsey():
    matcher = RNotNoneTypeMatcher()
    err = FalseyMatcherError()

    assert matcher.matches(None, err) is False
    assert err.calls == []


def test_not_none_type_matcher_equality_semantics():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()

    assert a == b
    assert a == a
    assert not (a != b)


def test_not_none_type_matcher_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)


def test_not_none_type_matcher_string_is_expected():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_rambutan3_package_imports_cleanly_and_has_name():
    mod = importlib.import_module("rambutan3")
    assert mod is rambutan3
    assert mod.__name__ == "rambutan3"
