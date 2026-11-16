# Tests for INT, RCheckArgsError, RNotNoneTypeMatcher, and the rambutan3 package.
#
# Note: RTestUtil is named in the request, but its source was not provided,
# so it is not tested here.

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher


class _RecordingError:
    """Minimal stand-in for RTypeMatcherError that records add_failed_match calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_package_importable():
    assert rambutan3 is not None


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    try:
        raise RCheckArgsError("bad argument")
    except RCheckArgsError as e:
        assert str(e) == "bad argument"
    else:
        raise AssertionError("RCheckArgsError was not raised")


def test_check_args_error_not_a_value_error():
    assert not issubclass(RCheckArgsError, ValueError)


def test_check_args_error_no_message():
    err = RCheckArgsError()
    assert str(err) == ""


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_matches_zero_and_empty_values():
    m = RNotNoneTypeMatcher()
    assert m.matches(0) is True
    assert m.matches("") is True
    assert m.matches(False) is True
    assert m.matches([]) is True


def test_not_none_rejects_none():
    m = RNotNoneTypeMatcher()
    assert m.matches(None) is False


def test_not_none_records_failure_on_none():
    m = RNotNoneTypeMatcher()
    err = _RecordingError()
    assert m.matches(None, err) is False
    assert err.calls == [(m, None)]


def test_not_none_does_not_record_on_success():
    m = RNotNoneTypeMatcher()
    err = _RecordingError()
    assert m.matches("value", err) is True
    assert err.calls == []


def test_not_none_equal_to_other_instance():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_not_equal_to_other_types():
    m = RNotNoneTypeMatcher()
    assert (m == object()) is False
    assert (m == None) is False  # noqa: E711
    assert (m == "not None") is False


def test_not_none_hash_returns_int():
    m = RNotNoneTypeMatcher()
    assert isinstance(hash(m), int)


def test_not_none_usable_in_set_by_identity():
    m = RNotNoneTypeMatcher()
    assert m in {m}


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


def test_int_matches_int_values():
    assert INT.matches(0) is True
    assert INT.matches(1) is True
    assert INT.matches(-42) is True
    assert INT.matches(10 ** 30) is True


def test_int_rejects_bool_true_and_false():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_rejects_non_int_types():
    assert INT.matches(1.0) is False
    assert INT.matches("1") is False
    assert INT.matches(None) is False
    assert INT.matches([1]) is False


def test_int_matches_int_subclass():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(3)) is True


def test_int_records_failure_for_bool():
    err = _RecordingError()
    assert INT.matches(True, err) is False
    assert len(err.calls) == 1
    assert err.calls[0][1] is True


def test_int_does_not_record_on_success():
    err = _RecordingError()
    assert INT.matches(7, err) is True
    assert err.calls == []
