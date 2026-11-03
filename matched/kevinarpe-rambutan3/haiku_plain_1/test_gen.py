import pytest

from rambutan3 import RTestUtil
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError
from rambutan3.error.RIllegalStateError import RIllegalStateError


# INT: restricted int matcher (int allowed, bool excluded)

def test_int_accepts_ints():
    assert INT.matches(5) is True
    assert INT.matches(0) is True
    assert INT.matches(-1) is True
    assert INT.matches(10 ** 30) is True


def test_int_rejects_bools():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_rejects_float():
    assert INT.matches(1.0) is False


def test_int_rejects_str_and_none():
    assert INT.matches("1") is False
    assert INT.matches(None) is False


def test_int_check_arg_returns_value():
    assert INT.check_arg(42, "n") == 42


def test_int_check_arg_raises_for_bool():
    with pytest.raises(RCheckArgsError):
        INT.check_arg(True, "flag")


def test_int_check_arg_message_names_argument_and_actual_type():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg(True, "flag")
    msg = str(exc_info.value)
    assert "Argument name : flag" in msg
    assert "Actual   type : bool" in msg
    assert "Actual  value : True" in msg


def test_int_check_arg_formats_arg_name_args():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg("x", "items[{}]", 3)
    msg = str(exc_info.value)
    assert "Argument name : items[3]" in msg
    assert "Actual   type : str" in msg


def test_int_matches_collects_failure_for_bool():
    err = RTypeMatcherError()
    assert INT.matches(True, err) is False
    failed = list(err.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].value is True


def test_int_matches_success_records_no_failure():
    err = RTypeMatcherError()
    assert INT.matches(3, err) is True
    with pytest.raises(RIllegalStateError):
        _ = err.failed_match_tuple_set_view


def test_int_str_is_int():
    assert str(INT) == "int"


def test_int_eq_and_hash_via_rtestutil():
    RTestUtil.test_eq_ne_hash(INT, INT, is_equal=True)


# RTestUtil.test_eq_ne_hash helper

def test_rtestutil_eq_ne_hash_passes_for_equal_and_unequal_values():
    RTestUtil.test_eq_ne_hash(7, 7, is_equal=True)
    RTestUtil.test_eq_ne_hash("abc", "abc", is_equal=True)
    RTestUtil.test_eq_ne_hash(1, 2, is_equal=False)
    RTestUtil.test_eq_ne_hash("a", "b", is_equal=False)


def test_rtestutil_eq_ne_hash_fails_on_wrong_expectation():
    with pytest.raises(AssertionError):
        RTestUtil.test_eq_ne_hash(1, 2, is_equal=True)
    with pytest.raises(AssertionError):
        RTestUtil.test_eq_ne_hash(1, 1, is_equal=False)


# RCheckArgsError

def test_rcheckargserror_is_exception_with_message():
    assert issubclass(RCheckArgsError, Exception)
    assert not issubclass(RCheckArgsError, (TypeError, ValueError))
    err = RCheckArgsError("bad arg")
    assert err.args[0] == "bad arg"
    with pytest.raises(RCheckArgsError, match="bad arg"):
        raise err


# RNotNoneTypeMatcher

def test_not_none_matches_non_none_and_rejects_none():
    m = RNotNoneTypeMatcher()
    assert m.matches(0) is True
    assert m.matches("") is True
    assert m.matches(False) is True
    assert m.matches([]) is True
    assert m.matches(None) is False


def test_not_none_matches_collects_failure_for_none():
    m = RNotNoneTypeMatcher()
    err = RTypeMatcherError()
    assert m.matches(None, err) is False
    failed = list(err.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].matcher is m
    assert failed[0].value is None


def test_not_none_check_arg_raises_for_none():
    with pytest.raises(RCheckArgsError) as exc_info:
        RNotNoneTypeMatcher().check_arg(None, "thing")
    msg = str(exc_info.value)
    assert "Argument name : thing" in msg
    assert "Expected type : not None" in msg
    assert "Actual   type : NoneType" in msg


def test_not_none_check_arg_returns_value():
    assert RNotNoneTypeMatcher().check_arg("x", "thing") == "x"


def test_not_none_equality():
    assert (RNotNoneTypeMatcher() == RNotNoneTypeMatcher()) is True
    assert (RNotNoneTypeMatcher() != RNotNoneTypeMatcher()) is False


def test_not_none_inequality_with_other_matchers():
    m = RNotNoneTypeMatcher()
    assert m != INT
    assert INT != m
    assert m != None  # noqa: E711
    assert m != "not None"


def test_not_none_str_is_not_none():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_hash_returns_int():
    assert isinstance(hash(RNotNoneTypeMatcher()), int)


def test_not_none_or_with_int():
    combined = RNotNoneTypeMatcher() | INT
    assert combined.matches(None) is False
    assert combined.matches("x") is True
    assert combined.matches(3) is True
