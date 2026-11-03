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


def test_package_import_smoke():
    assert rambutan3_package_name == "rambutan3"


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


@pytest.mark.parametrize("value", [0, 1, -1, 42, 10**6])
def test_int_matcher_accepts_int_values(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_matcher_rejects_bool_values(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [], {}, object()])
def test_int_matcher_rejects_non_int_values(value):
    assert INT.matches(value) is False


def test_int_matcher_is_not_bool():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_matcher_string_representation_mentions_int():
    text = str(INT)
    assert isinstance(text, str)
    assert "int" in text.lower()


def test_not_none_matcher_accepts_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(object()) is True


def test_not_none_matcher_rejects_none_without_error_object():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_matcher_rejects_none_and_records_failure():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_not_none_matcher_does_not_record_successful_match():
    matcher = RNotNoneTypeMatcher()
    error = DummyMatcherError()

    result = matcher.matches("x", error)

    assert result is True
    assert error.calls == []


def test_not_none_matcher_equality():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()

    assert matcher1 == matcher2
    assert matcher1 == matcher1
    assert not (matcher1 != matcher2)


def test_not_none_matcher_not_equal_to_other_types():
    matcher = RNotNoneTypeMatcher()

    assert matcher != None
    assert matcher != object()
    assert matcher != "not a matcher"


def test_not_none_matcher_hash_is_int():
    matcher = RNotNoneTypeMatcher()
    value = hash(matcher)

    assert isinstance(value, int)


def test_not_none_matcher_string_representation():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"
