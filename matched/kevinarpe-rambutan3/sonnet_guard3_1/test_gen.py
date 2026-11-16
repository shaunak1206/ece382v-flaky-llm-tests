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


# ---------------------------------------------------------------- package

def test_rambutan3_package_importable():
    assert rambutan3.__name__ == "rambutan3"


# ---------------------------------------------------------------- RCheckArgsError

def test_check_args_error_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_check_args_error_can_be_raised_and_caught_with_message():
    with pytest.raises(RCheckArgsError) as exc_info:
        raise RCheckArgsError("bad args")
    assert str(exc_info.value) == "bad args"


def test_check_args_error_caught_as_exception():
    try:
        raise RCheckArgsError("x")
    except Exception as e:
        assert isinstance(e, RCheckArgsError)
    else:
        pytest.fail("exception not raised")


# ---------------------------------------------------------------- INT

@pytest.mark.parametrize("value", [0, 1, -1, 123456789012345678901234567890])
def test_int_matches_ints(value):
    assert INT.matches(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_int_rejects_bool(value):
    assert not INT.matches(value)


@pytest.mark.parametrize("value", [1.0, "1", None, [1], (1,), {1: 1}, b"1", 1 + 0j])
def test_int_rejects_non_int(value):
    assert not INT.matches(value)


def test_int_matches_int_subclass_instance():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(5))


# ---------------------------------------------------------------- RNotNoneTypeMatcher

@pytest.mark.parametrize("value", [0, "", False, [], {}, object(), "abc", 1.5])
def test_not_none_matches_non_none_values(value):
    assert RNotNoneTypeMatcher().matches(value) is True


def test_not_none_rejects_none():
    assert RNotNoneTypeMatcher().matches(None) is False


def test_not_none_records_failure_in_matcher_error():
    matcher = RNotNoneTypeMatcher()
    err = _FakeMatcherError()
    assert matcher.matches(None, err) is False
    assert err.calls == [(matcher, None)]


def test_not_none_does_not_record_on_success():
    matcher = RNotNoneTypeMatcher()
    err = _FakeMatcherError()
    assert matcher.matches(5, err) is True
    assert err.calls == []


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert a == a
    assert not (a == object())
    assert a != "not None"
    assert a != None  # noqa: E711


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_hash_raises_attribute_error():
    # super(object, self) has no further class in the MRO, so __hash__ lookup fails.
    with pytest.raises(AttributeError):
        hash(RNotNoneTypeMatcher())
