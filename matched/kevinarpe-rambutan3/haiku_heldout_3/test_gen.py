import types

import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher


class _FakeMatcherError:
    """Minimal stand-in for RTypeMatcherError that records add_failed_match() calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_rambutan3_package_is_importable():
    assert isinstance(rambutan3, types.ModuleType)


# ---------------------------------------------------------------------------
# rambutan3/check_args/annotation/INT.py
# ---------------------------------------------------------------------------

def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


@pytest.mark.parametrize("value", [0, 1, -1, 42, 10 ** 30])
def test_int_matches_plain_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    # bool is a subclass of int, but must not match INT.
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, 0.0, "1", b"1", None, [1], (1,), {"a": 1}])
def test_int_rejects_non_int_values(value):
    assert INT.matches(value) is False


def test_int_matches_returns_bool_type():
    assert isinstance(INT.matches(1), bool)
    assert isinstance(INT.matches(True), bool)


def test_int_matches_accepts_none_matcher_error_argument():
    assert INT.matches(5, None) is True
    assert INT.matches(True, None) is False


# ---------------------------------------------------------------------------
# rambutan3/check_args/RCheckArgsError.py
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught_with_message():
    with pytest.raises(RCheckArgsError, match="bad argument"):
        raise RCheckArgsError("bad argument")


def test_check_args_error_caught_as_generic_exception():
    with pytest.raises(Exception) as exc_info:
        raise RCheckArgsError("oops")
    assert isinstance(exc_info.value, RCheckArgsError)
    assert str(exc_info.value) == "oops"


def test_check_args_error_without_message_has_empty_str():
    assert str(RCheckArgsError()) == ""


# ---------------------------------------------------------------------------
# rambutan3/check_args/base/RNotNoneTypeMatcher.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, "", False, [], {}, object(), "text", 3.14])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failure_when_value_is_none():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()
    result = matcher.matches(None, error)
    assert result is False
    assert len(error.calls) == 1
    recorded_matcher, recorded_value = error.calls[0]
    assert recorded_matcher is matcher
    assert recorded_value is None


def test_not_none_does_not_record_failure_when_value_is_not_none():
    error = _FakeMatcherError()
    assert RNotNoneTypeMatcher().matches("x", error) is True
    assert error.calls == []


def test_not_none_instances_are_equal_to_each_other():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


@pytest.mark.parametrize("other", ["not None", None, 0, object()])
def test_not_none_not_equal_to_other_types(other):
    matcher = RNotNoneTypeMatcher()
    assert (matcher == other) is False
    assert (matcher != other) is True


def test_not_none_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


def test_not_none_str_representation():
    assert str(RNotNoneTypeMatcher()) == "not None"
