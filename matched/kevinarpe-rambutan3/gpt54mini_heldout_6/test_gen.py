import importlib

import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError, match="boom"):
        raise RCheckArgsError("boom")


def test_importing_rambutan3_package_succeeds():
    mod = importlib.import_module("rambutan3")
    assert mod.__name__ == "rambutan3"


def test_int_matches_regular_int_positive():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-123456) is True


def test_int_matches_large_int_positive():
    assert INT.matches(10**100) is True


def test_int_rejects_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_rejects_non_int_values():
    assert INT.matches("1") is False
    assert INT.matches(1.0) is False
    assert INT.matches(None) is False
    assert INT.matches([1]) is False


def test_int_does_not_add_failure_for_success():
    err = RTypeMatcherError()
    assert INT.matches(42, err) is True
    assert len(err) == 0


def test_int_adds_failure_for_bool_value():
    err = RTypeMatcherError()
    assert INT.matches(True, err) is False
    assert len(err) == 1
    assert err[0].matcher == INT
    assert err[0].value is True


def test_not_none_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(object()) is True


def test_not_none_rejects_none_and_records_error():
    matcher = RNotNoneTypeMatcher()
    err = RTypeMatcherError()
    assert matcher.matches(None, err) is False
    assert len(err) == 1
    assert err[0].matcher == matcher
    assert err[0].value is None


def test_not_none_equality_semantics():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a != b)
    assert a != object()


def test_not_none_str_and_hash_are_stable_types():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"
    assert isinstance(hash(matcher), int)
    assert hash(matcher) == hash(matcher)
