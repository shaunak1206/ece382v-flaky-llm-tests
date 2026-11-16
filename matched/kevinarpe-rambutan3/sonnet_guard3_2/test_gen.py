import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeMatcherError:
    """Duck-typed stand-in for RTypeMatcherError that records calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


class _IntSubclass(int):
    pass


# ---------- rambutan3 package ----------

def test_package_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------- RCheckArgsError ----------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("boom")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("bad argument")
    assert str(err) == "bad argument"
    assert err.args == ("bad argument",)


def test_check_args_error_caught_as_generic_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


def test_check_args_error_not_a_value_error():
    assert not issubclass(RCheckArgsError, ValueError)
    assert not issubclass(RCheckArgsError, TypeError)


# ---------- INT ----------

@pytest.mark.parametrize("value", [0, 1, -1, 10 ** 30])
def test_int_matches_plain_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), {1: 1}, 1j, b"1"])
def test_int_rejects_non_int_types(value):
    assert INT.matches(value) is False


def test_int_accepts_int_subclass():
    assert INT.matches(_IntSubclass(7)) is True


# ---------- RNotNoneTypeMatcher ----------

@pytest.mark.parametrize("value", [0, "", [], False, {}, "x", 1.5, object()])
def test_not_none_matches_any_non_none(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failure_in_matcher_error():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()
    assert matcher.matches(None, error) is False
    assert error.calls == [(matcher, None)]


def test_not_none_does_not_record_on_success():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()
    assert matcher.matches(5, error) is True
    assert error.calls == []


def test_not_none_equality_and_str():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert not (a != b)
    assert a != "not None"
    assert a != None  # noqa: E711
    assert str(a) == "not None"
