import pytest

from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeMatcherError:
    """Minimal stand-in that records calls to add_failed_match()."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------------------
# INT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 1, -1, 123456789, 10 ** 30])
def test_int_matches_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), {1: 1}, b"1", 1j])
def test_int_rejects_non_int(value):
    assert INT.matches(value) is False


def test_int_rejects_bool_subclass_instance_but_accepts_int_subclass():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(5)) is True


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
    with pytest.raises(Exception) as exc_info:
        raise RCheckArgsError("x")
    assert isinstance(exc_info.value, RCheckArgsError)


# ---------------------------------------------------------------------------
# RNotNoneTypeMatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, "", [], {}, False, 1, "abc", object(), 0.0])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_does_not_match_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failed_match_on_error_object():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert error.calls == [(matcher, None)]


def test_not_none_does_not_record_on_success():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()

    assert matcher.matches(5, error) is True
    assert error.calls == []


def test_not_none_failure_without_error_object_does_not_raise():
    assert RNotNoneTypeMatcher().matches(None, None) is False


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a == "not None")
    assert not (a == None)  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
