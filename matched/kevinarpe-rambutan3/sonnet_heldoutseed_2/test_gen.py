import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher


class _FakeMatcherError:
    """Minimal stand-in that records add_failed_match calls."""

    def __init__(self):
        self.calls = []

    def add_failed_match(self, matcher, value):
        self.calls.append((matcher, value))


class _MyInt(int):
    pass


# ---------- package ----------

def test_package_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------- RCheckArgsError ----------

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("boom")


def test_check_args_error_preserves_message():
    err = RCheckArgsError("bad arg")
    assert str(err) == "bad arg"


def test_check_args_error_caught_as_exception():
    try:
        raise RCheckArgsError("x")
    except Exception as e:
        assert isinstance(e, RCheckArgsError)


# ---------- RNotNoneTypeMatcher ----------

@pytest.mark.parametrize("value", [0, "", [], False, 1, "abc", object(), (), {}])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_failure_records_error():
    matcher = RNotNoneTypeMatcher()
    err = _FakeMatcherError()
    assert matcher.matches(None, err) is False
    assert err.calls == [(matcher, None)]


def test_not_none_success_does_not_record_error():
    err = _FakeMatcherError()
    assert RNotNoneTypeMatcher().matches(5, err) is True
    assert err.calls == []


def test_not_none_none_without_error_object_does_not_raise():
    assert RNotNoneTypeMatcher().matches(None, None) is False


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert not (a != b)
    assert a != "not None"
    assert a != None  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


# ---------- INT ----------

@pytest.mark.parametrize("value", [0, 1, -5, 10 ** 30, _MyInt(3)])
def test_int_matches_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert INT.matches(value) is False


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), 1 + 0j, b"1"])
def test_int_rejects_non_int(value):
    assert INT.matches(value) is False
