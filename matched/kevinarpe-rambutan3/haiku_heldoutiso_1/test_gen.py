import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


# NOTE: The source for RTestUtil was not provided, so its API is not tested here.


class _RecordingTypeMatcherError:
    """Minimal stand-in for RTypeMatcherError that records add_failed_match calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


@pytest.mark.parametrize("value", [0, 1, -1, 42, -999999999999999999999])
def test_int_matches_int_values(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    # bool is a subclass of int, but must not match INT.
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, 0.5, "1", "", b"1", None, [1], (1,), {1: 1}])
def test_int_rejects_non_int_values(value):
    assert INT.matches(value) is False


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


def test_check_args_error_is_not_value_error():
    # Ensure it is its own exception type, not a built-in alias.
    assert RCheckArgsError is not ValueError
    assert not issubclass(RCheckArgsError, ValueError)


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_constructor_takes_no_arguments():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(matcher, RNotNoneTypeMatcher)


@pytest.mark.parametrize("value", [0, "", False, [], {}, object(), "None"])
def test_not_none_matches_non_none_values(value):
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(value) is True


def test_not_none_rejects_none_without_error_object():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_records_failure_for_none():
    matcher = RNotNoneTypeMatcher()
    error = _RecordingTypeMatcherError()

    assert matcher.matches(None, error) is False
    assert len(error.calls) == 1
    recorded_matcher, recorded_value = error.calls[0]
    assert recorded_matcher is matcher
    assert recorded_value is None


def test_not_none_does_not_record_failure_for_non_none():
    matcher = RNotNoneTypeMatcher()
    error = _RecordingTypeMatcherError()

    assert matcher.matches(0, error) is True
    assert error.calls == []


def test_not_none_equality_between_instances():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_not_equal_to_unrelated_object():
    assert (RNotNoneTypeMatcher() == object()) is False
    assert (RNotNoneTypeMatcher() != object()) is True


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
