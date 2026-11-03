import builtins
import pytest

from rambutan3 import __dict__ as rambutan3_package_dict  # noqa: F401
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class DummyMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_package_import_is_available():
    import rambutan3  # noqa: F401


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("boom")


def test_int_matches_regular_int_values():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-123) is True
    assert INT.matches(10**6) is True


def test_int_rejects_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_rejects_non_int_values():
    assert INT.matches("1") is False
    assert INT.matches(1.0) is False
    assert INT.matches(None) is False
    assert INT.matches([]) is False


def test_int_does_not_mutate_error_object_when_match_succeeds():
    err = DummyMatcherError()
    assert INT.matches(7, err) is True
    assert err.calls == []


def test_rnotnone_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True


def test_rnotnone_rejects_none_value():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_rnotnone_adds_failed_match_to_error_object_for_none():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    assert matcher.matches(None, err) is False
    assert err.calls == [(matcher, None)]


def test_rnotnone_does_not_call_error_object_for_non_none():
    matcher = RNotNoneTypeMatcher()
    err = DummyMatcherError()

    assert matcher.matches("x", err) is True
    assert err.calls == []


def test_rnotnone_equality_depends_on_type_not_identity():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()

    assert a == b
    assert a == a
    assert not (a == None)  # noqa: E711
    assert not (a == object())


def test_rnotnone_str_is_human_readable():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_rnotnone_hash_raises_attribute_error_due_to_current_implementation():
    matcher = RNotNoneTypeMatcher()

    with pytest.raises(AttributeError):
        hash(matcher)
