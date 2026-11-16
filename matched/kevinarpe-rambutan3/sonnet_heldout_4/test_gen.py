import pytest

from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeMatcherError:
    """Minimal stand-in for RTypeMatcherError that records failed matches."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------------------
# RCheckArgsError
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, "", [], {}, False, 1, "abc", object(), 3.14])
def test_not_none_matcher_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_matcher_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_matcher_records_failure_in_error_on_none():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()
    assert matcher.matches(None, error) is False
    assert error.calls == [(matcher, None)]


def test_not_none_matcher_does_not_record_on_success():
    error = _FakeMatcherError()
    assert RNotNoneTypeMatcher().matches(5, error) is True
    assert error.calls == []


def test_not_none_matcher_none_error_is_ok_on_failure():
    assert RNotNoneTypeMatcher().matches(None, None) is False


def test_not_none_matcher_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert a != "not None"
    assert a != None  # noqa: E711
    assert a != object()


def test_not_none_matcher_str():
    assert str(RNotNoneTypeMatcher()) == 'not None'


def test_not_none_matcher_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 1, -1, 10 ** 30])
def test_int_matches_integers(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), 1 + 0j])
def test_int_rejects_non_integers(value):
    assert INT.matches(value) is False
