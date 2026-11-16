"""Unit tests for INT, RCheckArgsError, RNotNoneTypeMatcher, and the rambutan3 package.

Note: RTestUtil was not included in the provided source, so it is not tested here.
"""
import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _RecordingTypeMatcherError:
    """Minimal stand-in for RTypeMatcherError that records failed matches."""

    def __init__(self):
        self.failed_matches = []

    def add_failed_match(self, matcher, value):
        self.failed_matches.append((matcher, value))


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_package_is_importable():
    assert rambutan3 is not None


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError()


def test_check_args_error_preserves_message():
    with pytest.raises(RCheckArgsError, match="bad argument"):
        raise RCheckArgsError("bad argument")


def test_check_args_error_is_not_value_error():
    assert not issubclass(RCheckArgsError, ValueError)


# ---------------------------------------------------------------------------
# INT (RRestrictedInstanceMatcher allowing int, excluding bool)
# ---------------------------------------------------------------------------

def test_int_matches_positive_int():
    assert INT.matches(5) is True


def test_int_matches_zero_and_negative_int():
    assert INT.matches(0) is True
    assert INT.matches(-42) is True


def test_int_rejects_true():
    assert INT.matches(True) is False


def test_int_rejects_false():
    assert INT.matches(False) is False


def test_int_rejects_float():
    assert INT.matches(1.0) is False


def test_int_rejects_string():
    assert INT.matches("1") is False


def test_int_rejects_none():
    assert INT.matches(None) is False


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(False) is True
    assert matcher.matches([]) is True


def test_not_none_rejects_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_records_failed_match_on_none():
    matcher = RNotNoneTypeMatcher()
    error = _RecordingTypeMatcherError()
    result = matcher.matches(None, error)
    assert result is False
    assert len(error.failed_matches) == 1
    assert error.failed_matches[0] == (matcher, None)


def test_not_none_does_not_record_failed_match_on_success():
    matcher = RNotNoneTypeMatcher()
    error = _RecordingTypeMatcherError()
    assert matcher.matches(123, error) is True
    assert error.failed_matches == []


def test_not_none_equality_between_instances():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_not_equal_to_other_types():
    matcher = RNotNoneTypeMatcher()
    assert not (matcher == object())
    assert not (matcher == "not None")
    assert not (matcher == None)  # noqa: E711


def test_not_none_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)


def test_not_none_is_usable_in_set():
    matcher = RNotNoneTypeMatcher()
    assert matcher in {matcher}


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
