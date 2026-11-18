import importlib

import pytest

from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class DummyMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_rcheckargserror_is_exception_subclass_and_message():
    err = RCheckArgsError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


def test_rnotnone_matches_none_false():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_rnotnone_matches_non_none_true():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(object()) is True


def test_rnotnone_records_failed_match_when_value_is_none_and_error_provided():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    result = matcher.matches(None, err)

    assert result is False
    assert err.calls == [(matcher, None)]


def test_rnotnone_does_not_record_failure_when_value_is_not_none():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    result = matcher.matches("x", err)

    assert result is True
    assert err.calls == []


def test_rnotnone_equality_is_based_on_type():
    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()

    assert m1 == m2
    assert m1 == m1
    assert not (m1 == object())
    assert m1 != object()


def test_rnotnone_hash_is_an_int_and_stable_for_same_instance():
    matcher = RNotNoneTypeMatcher()

    h1 = hash(matcher)
    h2 = hash(matcher)

    assert isinstance(h1, int)
    assert h1 == h2


def test_rnotnone_str_is_expected():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_int_accepts_regular_int_values():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-7) is True
    assert INT.matches(10**6) is True


def test_int_rejects_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_does_not_reject_subclass_of_int_that_is_not_bool():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(5)) is True


def test_int_is_strictly_more_restrictive_than_object_isinstance():
    # A bool is an int subclass, but the matcher must exclude it.
    assert isinstance(True, int) is True
    assert INT.matches(True) is False


def test_int_module_exports_singleton_like_object():
    mod = importlib.import_module("rambutan3.check_args.annotation.INT")
    assert mod.INT is INT
    assert str(mod.INT) == str(INT)
