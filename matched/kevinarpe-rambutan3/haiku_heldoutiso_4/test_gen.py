import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher

# Note: RTestUtil is referenced in the task but its source was not provided,
# so no tests for RTestUtil are included in this file.


class _RecordingErrorCollector:
    """Minimal stand-in for RTypeMatcherError that records failed matches."""

    def __init__(self):
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


# ---------------------------------------------------------------------------
# rambutan3/__init__.py
# ---------------------------------------------------------------------------

def test_package_importable():
    import rambutan3
    assert rambutan3 is not None


def test_package_init_file_exists():
    import os
    import rambutan3
    assert os.path.basename(rambutan3.__file__) == "__init__.py"


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


@pytest.mark.parametrize("value", [0, 1, -1, 42, 10 ** 30])
def test_int_matches_int_values(value):
    assert INT.matches(value)


def test_int_rejects_true():
    assert not INT.matches(True)


def test_int_rejects_false():
    assert not INT.matches(False)


@pytest.mark.parametrize("value", [1.0, 0.5, "1", b"1", None, [1], (1,), {1: 1}])
def test_int_rejects_non_int_values(value):
    assert not INT.matches(value)


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad argument")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("expected int")
    assert str(err) == "expected int"


def test_check_args_error_caught_as_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

def test_not_none_constructs_without_arguments():
    matcher = RNotNoneTypeMatcher()
    assert isinstance(matcher, RNotNoneTypeMatcher)


@pytest.mark.parametrize("value", [0, "", False, [], object(), 1.5])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value)


def test_not_none_rejects_none():
    assert not RNotNoneTypeMatcher().matches(None)


def test_not_none_reports_failure_to_error_collector_on_none():
    matcher = RNotNoneTypeMatcher()
    collector = _RecordingErrorCollector()
    result = matcher.matches(None, collector)
    assert result is False
    assert collector.failed == [(matcher, None)]


def test_not_none_does_not_report_failure_for_non_none_value():
    matcher = RNotNoneTypeMatcher()
    collector = _RecordingErrorCollector()
    result = matcher.matches("value", collector)
    assert result is True
    assert collector.failed == []


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_equality_between_instances():
    assert RNotNoneTypeMatcher() == RNotNoneTypeMatcher()


def test_not_none_not_equal_to_other_types():
    assert RNotNoneTypeMatcher() != object()
    assert RNotNoneTypeMatcher() != "not None"
    assert RNotNoneTypeMatcher() != INT


def test_not_none_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)
