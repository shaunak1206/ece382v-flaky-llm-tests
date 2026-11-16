from unittest import mock

import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


# ---------------------------------------------------------------------------
# rambutan3 package
# ---------------------------------------------------------------------------

def test_package_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad arg")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("some message")
    assert str(err) == "some message"
    assert err.args == ("some message",)


def test_check_args_error_caught_as_generic_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 1, -1, 123456789012345678901234567890])
def test_int_matches_integers(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), {1}, 1j, b"1"])
def test_int_rejects_non_int(value):
    assert INT.matches(value) is False


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_is_abstract_type_matcher_subclass():
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)


@pytest.mark.parametrize("value", [0, "", [], {}, False, object(), 1, "abc", (), 0.0])
def test_not_none_matches_everything_but_none(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_failed_match_reports_to_matcher_error():
    matcher = RNotNoneTypeMatcher()
    error = mock.MagicMock()
    assert matcher.matches(None, error) is False
    error.add_failed_match.assert_called_once_with(matcher, None)


def test_not_none_successful_match_does_not_report_error():
    matcher = RNotNoneTypeMatcher()
    error = mock.MagicMock()
    assert matcher.matches("value", error) is True
    error.add_failed_match.assert_not_called()


def test_not_none_none_without_matcher_error_does_not_raise():
    assert RNotNoneTypeMatcher().matches(None, None) is False


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a == object())
    assert not (a == "not None")
    assert a != None  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)
