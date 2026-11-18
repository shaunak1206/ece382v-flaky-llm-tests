import importlib

import pytest

from rambutan3 import __dict__ as rambutan3_namespace
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError, match="boom"):
        raise RCheckArgsError("boom")


def test_rambutan3_root_package_imports_cleanly():
    mod = importlib.import_module("rambutan3")
    assert mod is not None


def test_rambutan3_root_package_is_effectively_empty():
    # The package root intentionally exports no public symbols.
    assert "INT" not in rambutan3_namespace
    assert "RCheckArgsError" not in rambutan3_namespace


def test_int_matcher_accepts_plain_int():
    assert INT.matches(0) is True
    assert INT.matches(123) is True
    assert INT.matches(-7) is True


def test_int_matcher_rejects_bool_true():
    assert INT.matches(True) is False


def test_int_matcher_rejects_bool_false():
    assert INT.matches(False) is False


def test_int_matcher_rejects_non_int_types():
    assert INT.matches("123") is False
    assert INT.matches(1.0) is False
    assert INT.matches(None) is False
    assert INT.matches([1]) is False


def test_int_matcher_does_not_report_error_for_success():
    class DummyError:
        def __init__(self):
            self.calls = []

        def add_failed_match(self, matcher, value):
            self.calls.append((matcher, value))

    err = DummyError()
    assert INT.matches(5, err) is True
    assert err.calls == []


def test_int_matcher_reports_error_for_bool():
    class DummyError:
        def __init__(self):
            self.calls = []

        def add_failed_match(self, matcher, value):
            self.calls.append((matcher, value))

    err = DummyError()
    assert INT.matches(True, err) is False
    assert err.calls == [(INT, True)]


def test_int_matcher_string_representation_mentions_int():
    s = str(INT)
    assert "int" in s.lower()
    assert "bool" not in s.lower()


def test_not_none_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(object()) is True


def test_not_none_rejects_none_and_adds_error():
    matcher = RNotNoneTypeMatcher()

    class DummyError:
        def __init__(self):
            self.calls = []

        def add_failed_match(self, matcher_obj, value):
            self.calls.append((matcher_obj, value))

    err = DummyError()
    assert matcher.matches(None, err) is False
    assert err.calls == [(matcher, None)]


def test_not_none_equality_and_inequality():
    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()

    assert m1 == m2
    assert m1 == m1
    assert not (m1 == object())
    assert m1 != object()


def test_not_none_hash_is_int_and_str_is_expected():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)
    assert str(matcher) == "not None"
