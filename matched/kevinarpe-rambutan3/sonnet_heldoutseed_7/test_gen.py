import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _StubMatcherError:
    """Minimal stand-in for RTypeMatcherError that records failed matches."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


class _IntSubclass(int):
    pass


# ---------------------------------------------------------------- INT

@pytest.mark.parametrize("value", [0, 1, -1, 42, 10 ** 30, -10 ** 30])
def test_int_matches_ints(value):
    assert INT.matches(value) is True


def test_int_matches_int_subclass():
    assert INT.matches(_IntSubclass(5)) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, 0.5, "1", b"1", None, [1], (1,), {1: 1}, 1 + 0j])
def test_int_rejects_non_ints(value):
    assert INT.matches(value) is False


def test_int_records_failure_in_matcher_error():
    error = _StubMatcherError()
    assert INT.matches("abc", error) is False
    assert len(error.calls) >= 1
    assert error.calls[0][1] == "abc"


def test_int_does_not_record_failure_on_success():
    error = _StubMatcherError()
    assert INT.matches(7, error) is True
    assert error.calls == []


# ---------------------------------------------------------------- RCheckArgsError

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught_with_message():
    with pytest.raises(RCheckArgsError) as exc_info:
        raise RCheckArgsError("bad argument")
    assert str(exc_info.value) == "bad argument"


def test_check_args_error_caught_as_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ---------------------------------------------------------------- RNotNoneTypeMatcher

@pytest.mark.parametrize("value", [0, "", False, [], {}, (), 1, "x", object(), 0.0])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failure_for_none():
    matcher = RNotNoneTypeMatcher()
    error = _StubMatcherError()
    assert matcher.matches(None, error) is False
    assert error.calls == [(matcher, None)]


def test_not_none_no_failure_recorded_for_non_none():
    error = _StubMatcherError()
    assert RNotNoneTypeMatcher().matches(5, error) is True
    assert error.calls == []


def test_not_none_none_without_matcher_error_does_not_raise():
    assert RNotNoneTypeMatcher().matches(None, None) is False


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a != b)


def test_not_none_inequality_with_other_types():
    a = RNotNoneTypeMatcher()
    assert a != object()
    assert a != "not None"
    assert a != None  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


# ---------------------------------------------------------------- package

def test_rambutan3_package_importable():
    assert rambutan3.__name__ == "rambutan3"
    assert hasattr(rambutan3, "__path__")
