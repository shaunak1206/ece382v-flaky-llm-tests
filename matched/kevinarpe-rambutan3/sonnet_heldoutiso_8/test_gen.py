import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeMatcherError:
    """Minimal stand-in that records calls to add_failed_match()."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


# ---------------------------------------------------------------- package

def test_package_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------------------------------------------------------- RCheckArgsError

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("boom")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("bad argument")
    assert str(err) == "bad argument"
    assert err.args == ("bad argument",)


def test_check_args_error_caught_as_exception():
    with pytest.raises(Exception):
        raise RCheckArgsError()


# ------------------------------------------------------------------- INT

@pytest.mark.parametrize("value", [0, 1, -5, 10 ** 30])
def test_int_matches_ints(value):
    assert INT.matches(value)


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert not INT.matches(value)


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), 1 + 0j, b"1"])
def test_int_rejects_non_int(value):
    assert not INT.matches(value)


def test_int_matches_int_subclass_that_is_not_bool():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(3))


# ------------------------------------------------------ RNotNoneTypeMatcher

def test_not_none_is_abstract_type_matcher_subclass():
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)


@pytest.mark.parametrize("value", [0, "", [], {}, False, object(), 1.5, "abc"])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failure_in_matcher_error():
    matcher = RNotNoneTypeMatcher()
    error = _FakeMatcherError()
    assert matcher.matches(None, error) is False
    assert error.calls == [(matcher, None)]


def test_not_none_does_not_record_on_success():
    error = _FakeMatcherError()
    assert RNotNoneTypeMatcher().matches(5, error) is True
    assert error.calls == []


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert not (a != b)
    assert a != "not None"
    assert a != None  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"
