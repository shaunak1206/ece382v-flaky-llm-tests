"""Unit tests for INT, RCheckArgsError, RNotNoneTypeMatcher, and rambutan3/__init__.py.

RTestUtil is not covered here because its source was not provided, so its API is unknown.
These tests use no randomness, no wall-clock time, no module-level mutation, and no
str/bytes hashing, so every result is deterministic.
"""
import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher

import pytest


class _RecordingMatcherError(object):
    """Minimal stand-in for RTypeMatcherError that records add_failed_match() calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------- INT

def test_int_matches_positive_int():
    assert INT.matches(1) is True
    assert INT.matches(42) is True


def test_int_matches_zero_and_negative_int():
    assert INT.matches(0) is True
    assert INT.matches(-7) is True


def test_int_matches_large_int():
    assert INT.matches(10 ** 30) is True
    assert INT.matches(-(10 ** 30)) is True


def test_int_rejects_true():
    assert INT.matches(True) is False


def test_int_rejects_false():
    assert INT.matches(False) is False


@pytest.mark.parametrize("value", [1.0, "1", b"1", [1], (1,), {1: 1}, 1j])
def test_int_rejects_non_int_types(value):
    assert INT.matches(value) is False


def test_int_rejects_none():
    assert INT.matches(None) is False


def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


def test_int_failure_records_error_once():
    error = _RecordingMatcherError()
    result = INT.matches("not an int", matcher_error=error)
    assert result is False
    assert len(error.calls) == 1
    assert error.calls[0][0] is INT
    assert error.calls[0][1] == "not an int"


def test_int_success_records_nothing():
    error = _RecordingMatcherError()
    result = INT.matches(5, matcher_error=error)
    assert result is True
    assert error.calls == []


# ---------------------------------------------------------------- RCheckArgsError

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_raise_and_catch_keeps_message():
    with pytest.raises(RCheckArgsError) as excinfo:
        raise RCheckArgsError("bad argument")
    assert str(excinfo.value) == "bad argument"


def test_check_args_error_without_message_has_empty_str():
    error = RCheckArgsError()
    assert str(error) == ""
    assert isinstance(error, Exception)


# ---------------------------------------------------------------- RNotNoneTypeMatcher

@pytest.mark.parametrize("value", [0, 1, "", "text", False, True, [], b"", 0.0])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_none_without_error_collector_does_not_raise():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None, matcher_error=None) is False


def test_not_none_failure_records_error_once():
    matcher = RNotNoneTypeMatcher()
    error = _RecordingMatcherError()
    assert matcher.matches(None, matcher_error=error) is False
    assert len(error.calls) == 1
    assert error.calls[0][0] is matcher
    assert error.calls[0][1] is None


def test_not_none_success_records_nothing():
    error = _RecordingMatcherError()
    assert RNotNoneTypeMatcher().matches("x", matcher_error=error) is True
    assert error.calls == []


def test_not_none_instances_are_equal():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()
    assert not (RNotNoneTypeMatcher() != RNotNoneTypeMatcher())


def test_not_none_not_equal_to_other_types():
    matcher = RNotNoneTypeMatcher()
    assert (matcher == None) is False  # noqa: E711
    assert (matcher == "not None") is False
    assert (matcher == INT) is False
    assert matcher != 0


def test_not_none_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(hash(matcher), int)


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


# ---------------------------------------------------------------- rambutan3 package

def test_rambutan3_is_importable_package():
    assert rambutan3.__name__ == "rambutan3"
    assert hasattr(rambutan3, "__path__")
