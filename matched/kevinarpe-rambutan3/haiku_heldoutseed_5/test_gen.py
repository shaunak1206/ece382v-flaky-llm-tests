"""
Unit tests for:
- rambutan3.check_args.annotation.INT.INT
- rambutan3.check_args.RCheckArgsError.RCheckArgsError
- rambutan3.check_args.base.RNotNoneTypeMatcher.RNotNoneTypeMatcher
- rambutan3.__init__

Note: RTestUtil was listed in the focus but its source was not provided,
so it is not tested here.
"""
import pytest

import rambutan3
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _RecordingError:
    """Minimal stand-in for RTypeMatcherError that records add_failed_match calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


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
        raise RCheckArgsError("bad argument")


def test_check_args_error_preserves_message():
    with pytest.raises(RCheckArgsError, match="bad argument"):
        raise RCheckArgsError("bad argument")


def test_check_args_error_caught_as_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ---------------------------------------------------------------------------
# INT (RRestrictedInstanceMatcher allowing int, excluding bool)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 1, -1, 42, 10 ** 30])
def test_int_matches_int_values(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, 0.5, "1", b"1", None, [1], (1,), {"a": 1}])
def test_int_rejects_non_int_types(value):
    assert INT.matches(value) is False


def test_int_matches_with_no_error_argument():
    # Default matcher_error=None must not raise.
    assert INT.matches(5, None) is True
    assert INT.matches("x", None) is False


def test_int_is_type_matcher_instance():
    assert isinstance(INT, RAbstractTypeMatcher)


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, "", [], False, 0.0, object()])
def test_not_none_matcher_accepts_non_none(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_matcher_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_matcher_records_failure_on_none():
    matcher = RNotNoneTypeMatcher()
    err = _RecordingError()
    assert matcher.matches(None, err) is False
    assert len(err.calls) == 1
    recorded_matcher, recorded_value = err.calls[0]
    assert recorded_matcher is matcher
    assert recorded_value is None


def test_not_none_matcher_does_not_record_failure_on_non_none():
    err = _RecordingError()
    assert RNotNoneTypeMatcher().matches("value", err) is True
    assert err.calls == []


def test_not_none_matcher_equal_to_another_instance():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_matcher_not_equal_to_other_types():
    matcher = RNotNoneTypeMatcher()
    assert matcher != None  # noqa: E711
    assert matcher != object()
    assert matcher != "not None"
    assert not (matcher == INT)


def test_not_none_matcher_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


def test_not_none_matcher_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_matcher_is_type_matcher_instance():
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)
