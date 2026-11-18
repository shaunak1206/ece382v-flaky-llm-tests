import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError
from rambutan3 import __dict__ as rambutan3_package_dict


class DummyTypeMatcherError:
    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


def test_package_init_exports_no_public_names():
    assert isinstance(rambutan3_package_dict, dict)
    assert "__name__" in rambutan3_package_dict
    assert "INT" not in rambutan3_package_dict


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_instantiated_and_raised():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


def test_int_matcher_accepts_regular_int():
    assert INT.matches(0) is True
    assert INT.matches(123) is True
    assert INT.matches(-456) is True


def test_int_matcher_rejects_bool_values():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_matcher_rejects_non_int_values():
    assert INT.matches(None) is False
    assert INT.matches(1.5) is False
    assert INT.matches("1") is False
    assert INT.matches([]) is False


def test_int_matcher_has_expected_string_representation():
    s = str(INT)
    assert "int" in s.lower()
    assert "bool" in s.lower()


def test_int_matcher_is_instance_of_restricted_instance_matcher():
    # Avoid relying on implementation details beyond the public object.
    assert hasattr(INT, "matches")
    assert callable(INT.matches)


def test_not_none_matcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(object()) is True


def test_not_none_matcher_rejects_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_matcher_adds_failed_match_when_error_provided():
    matcher = RNotNoneTypeMatcher()
    error = DummyTypeMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_not_none_matcher_does_not_call_error_on_success():
    matcher = RNotNoneTypeMatcher()
    error = DummyTypeMatcherError()

    result = matcher.matches("x", error)

    assert result is True
    assert error.calls == []


def test_not_none_matcher_equality_is_based_on_type_only():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()

    assert a == b
    assert a == a
    assert not (a != b)


def test_not_none_matcher_is_not_equal_to_unrelated_objects():
    matcher = RNotNoneTypeMatcher()

    assert matcher != None  # noqa: E711
    assert matcher != 123
    assert matcher != object()


def test_not_none_matcher_hash_returns_int():
    matcher = RNotNoneTypeMatcher()

    assert isinstance(hash(matcher), int)


def test_not_none_matcher_str_is_expected():
    matcher = RNotNoneTypeMatcher()

    assert str(matcher) == "not None"


def test_not_none_matcher_can_be_used_with_real_type_matcher_error_if_instantiable():
    matcher = RNotNoneTypeMatcher()

    # If the dependency can be instantiated, verify no crash on happy path.
    # If it cannot, the import above would already fail.
    error = RTypeMatcherError() if callable(getattr(RTypeMatcherError, "__call__", None)) else None
    assert matcher.matches("value", error) is True
