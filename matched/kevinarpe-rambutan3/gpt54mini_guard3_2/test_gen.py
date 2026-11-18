import importlib

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


def test_package_importable():
    pkg = importlib.import_module(rambutan3_package_name)
    assert pkg.__name__ == "rambutan3"


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError, match="problem"):
        raise RCheckArgsError("problem")


def test_rnotnone_instantiation_succeeds():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(matcher, RNotNoneTypeMatcher)


def test_rnotnone_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_rnotnone_equals_same_type_instance():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_rnotnone_not_equals_unrelated_object():
    assert not (RNotNoneTypeMatcher() == object())


def test_rnotnone_matches_none_without_error_object():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_rnotnone_matches_none_records_failure():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    result = matcher.matches(None, err)

    assert result is False
    assert err.calls == [(matcher, None)]


def test_rnotnone_matches_non_none_true_and_no_error_recorded():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    assert matcher.matches(0, err) is True
    assert err.calls == []


def test_rnotnone_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)


def test_int_accepts_regular_ints():
    assert INT.matches(0) is True
    assert INT.matches(-123) is True
    assert INT.matches(10**20) is True


def test_int_rejects_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_rejects_none():
    assert INT.matches(None) is False


def test_int_failure_records_error():
    err = DummyMatcherError()

    result = INT.matches(True, err)

    assert result is False
    assert err.calls == [(INT, True)]


def test_int_success_does_not_record_error():
    err = DummyMatcherError()

    result = INT.matches(7, err)

    assert result is True
    assert err.calls == []
