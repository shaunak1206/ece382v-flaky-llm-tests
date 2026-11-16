from unittest.mock import MagicMock

import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _MyInt(int):
    pass


class _MyBool_Holder:
    pass


# ---------------------------------------------------------------- INT

@pytest.mark.parametrize("value", [0, 1, -1, 123456789012345678901234567890])
def test_int_matches_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", b"1", None, [1], (1,), {1}, {1: 1}, 1j])
def test_int_rejects_non_ints(value):
    assert INT.matches(value) is False


def test_int_matches_int_subclass():
    assert INT.matches(_MyInt(5)) is True


def test_int_rejects_arbitrary_object():
    assert INT.matches(_MyBool_Holder()) is False


# ---------------------------------------------------- RCheckArgsError

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("some message")
    assert str(err) == "some message"
    assert err.args == ("some message",)


def test_check_args_error_caught_as_generic_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ------------------------------------------------ RNotNoneTypeMatcher

def test_not_none_matches_none_is_false():
    assert RNotNoneTypeMatcher().matches(None) is False


@pytest.mark.parametrize("value", [0, "", False, [], {}, (), 1, "abc", object(), 0.0])
def test_not_none_matches_everything_else(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_failure_reports_to_matcher_error():
    matcher = RNotNoneTypeMatcher()
    error = MagicMock()
    assert matcher.matches(None, error) is False
    error.add_failed_match.assert_called_once_with(matcher, None)


def test_not_none_success_does_not_report_to_matcher_error():
    matcher = RNotNoneTypeMatcher()
    error = MagicMock()
    assert matcher.matches("x", error) is True
    error.add_failed_match.assert_not_called()


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a != b)


def test_not_none_not_equal_to_other_types():
    a = RNotNoneTypeMatcher()
    assert a != "not None"
    assert a != None  # noqa: E711
    assert a != object()


# ------------------------------------------------------- rambutan3

def test_rambutan3_package_importable():
    assert rambutan3.__name__ == "rambutan3"
    assert hasattr(rambutan3, "__file__")
