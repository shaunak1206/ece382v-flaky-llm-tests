import pytest

from rambutan3 import __init__ as rambutan3_init
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher


class _RecordingMatcherError:
    """Minimal stand-in for RTypeMatcherError that records failed matches."""

    def __init__(self):
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


class _MyInt(int):
    pass


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_package_init_module_imports():
    assert rambutan3_init is not None


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


def test_int_matches_positive_int():
    assert INT.matches(42)


def test_int_matches_zero():
    assert INT.matches(0)


def test_int_matches_negative_int():
    assert INT.matches(-7)


def test_int_matches_large_int():
    assert INT.matches(10 ** 30)


def test_int_rejects_true():
    assert not INT.matches(True)


def test_int_rejects_false():
    assert not INT.matches(False)


def test_int_rejects_float():
    assert not INT.matches(1.0)


def test_int_rejects_string_digits():
    assert not INT.matches("1")


def test_int_rejects_none():
    assert not INT.matches(None)


def test_int_matches_int_subclass():
    assert INT.matches(_MyInt(3))


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad argument")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("bad argument")
    assert str(err) == "bad argument"


def test_check_args_error_not_value_error():
    assert not issubclass(RCheckArgsError, ValueError)


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_matcher_is_abstract_type_matcher():
    from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)


@pytest.mark.parametrize("value", [0, "", [], False, 3.5, object()])
def test_not_none_matcher_accepts_non_none(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_matcher_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_matcher_records_failure_on_none():
    matcher = RNotNoneTypeMatcher()
    err = _RecordingMatcherError()
    result = matcher.matches(None, err)
    assert result is False
    assert err.failed == [(matcher, None)]


def test_not_none_matcher_no_failure_recorded_on_non_none():
    matcher = RNotNoneTypeMatcher()
    err = _RecordingMatcherError()
    result = matcher.matches(5, err)
    assert result is True
    assert err.failed == []


def test_not_none_matcher_equal_to_another_instance():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_matcher_not_equal_to_none():
    assert not (RNotNoneTypeMatcher() == None)  # noqa: E711


def test_not_none_matcher_not_equal_to_other_type():
    assert not (RNotNoneTypeMatcher() == "not None")


def test_not_none_matcher_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
