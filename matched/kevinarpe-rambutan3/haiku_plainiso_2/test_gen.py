import os

import pytest

import rambutan3
import rambutan3.RTestUtil as rtu
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError
from rambutan3.error.RIllegalStateError import RIllegalStateError


class MyInt(int):
    pass


# INT: positive cases

def test_int_accepts_plain_ints():
    assert INT.matches(0)
    assert INT.matches(-5)
    assert INT.matches(10 ** 30)


def test_int_accepts_int_subclass():
    assert INT.matches(MyInt(3))


def test_int_str_is_type_names():
    assert str(INT) == "int"
    multi = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(int, float),
                                       not_allowed_class_or_type_iterable=())
    assert str(multi) == "int | float"


# INT: negative cases

def test_int_rejects_bool():
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_int_rejects_other_types():
    for value in (1.0, "1", None, [1], (1,), b"1"):
        assert not INT.matches(value)


def test_int_matcher_error_records_bool_failure():
    err = RTypeMatcherError()
    assert INT.matches(True, err) is False
    tuples = list(err.failed_match_tuple_set_view)
    assert len(tuples) == 1
    assert tuples[0].matcher is INT
    assert tuples[0].value is True


def test_int_matcher_error_records_inner_type_failure():
    err = RTypeMatcherError()
    assert INT.matches("x", err) is False
    tuples = list(err.failed_match_tuple_set_view)
    assert len(tuples) == 1
    # The failure comes from the delegate isinstance matcher, not from INT itself.
    assert tuples[0].matcher is not INT
    assert str(tuples[0].matcher) == "int"
    assert tuples[0].value == "x"


def test_int_matcher_error_untouched_on_match():
    err = RTypeMatcherError()
    assert INT.matches(5, err) is True
    with pytest.raises(RIllegalStateError):
        _ = err.failed_match_tuple_set_view


# INT: check_arg()

def test_int_check_arg_returns_value():
    assert INT.check_arg(42, "n") == 42


def test_int_check_arg_rejects_bool_with_message():
    assert issubclass(RCheckArgsError, Exception)
    with pytest.raises(RCheckArgsError) as excinfo:
        INT.check_arg(True, "n")
    msg = excinfo.value.args[0]
    assert "Argument name : n" in msg
    assert "Expected type : int" in msg
    assert "Actual   type : bool" in msg
    assert "Actual  value : True" in msg


def test_int_check_arg_formats_arg_name_placeholders():
    with pytest.raises(RCheckArgsError) as excinfo:
        INT.check_arg("s", "Index {}", 7)
    msg = excinfo.value.args[0]
    assert "Argument name : Index 7" in msg
    assert "Actual   type : str" in msg
    assert "Actual  value : 's'" in msg


# RRestrictedInstanceMatcher: constructor behaviour

def test_restricted_empty_not_allowed_permits_bool():
    m = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(int,),
                                   not_allowed_class_or_type_iterable=())
    assert m.matches(True)


def test_restricted_empty_allowed_raises_value_error():
    with pytest.raises(ValueError):
        RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(),
                                   not_allowed_class_or_type_iterable=())


def test_restricted_non_type_allowed_raises_type_error():
    with pytest.raises(TypeError):
        RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=("int",),
                                   not_allowed_class_or_type_iterable=())


def test_restricted_none_arguments_raise_type_error():
    with pytest.raises(TypeError):
        RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=None,
                                   not_allowed_class_or_type_iterable=())
    with pytest.raises(TypeError):
        RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(int,),
                                   not_allowed_class_or_type_iterable=None)


def test_int_equality_and_hash_via_eq_ne_hash_helper():
    same = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(int,),
                                      not_allowed_class_or_type_iterable=(bool,))
    other = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(float,),
                                       not_allowed_class_or_type_iterable=())
    rtu.test_eq_ne_hash(INT, same, is_equal=True)
    rtu.test_eq_ne_hash(INT, other, is_equal=False)


# RNotNoneTypeMatcher

def test_not_none_matches_non_none_values():
    m = RNotNoneTypeMatcher()
    for value in (0, "", False, [], 0.0):
        assert m.matches(value)


def test_not_none_rejects_none_and_records_failure():
    m = RNotNoneTypeMatcher()
    assert not m.matches(None)
    err = RTypeMatcherError()
    assert m.matches(None, err) is False
    tuples = list(err.failed_match_tuple_set_view)
    assert len(tuples) == 1
    assert tuples[0].matcher is m
    assert tuples[0].value is None


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == "not None"


def test_not_none_equality_only_with_same_class():
    m = RNotNoneTypeMatcher()
    assert m == RNotNoneTypeMatcher()
    assert not (m != RNotNoneTypeMatcher())
    assert m != INT
    assert INT != m
    assert m != None  # noqa: E711
    assert m != "not None"
    # __hash__ is defined and returns an int; no cross-instance hash check is made here.
    assert isinstance(hash(m), int)


def test_not_none_check_arg_none_message():
    with pytest.raises(RCheckArgsError) as excinfo:
        RNotNoneTypeMatcher().check_arg(None, "value")
    msg = excinfo.value.args[0]
    assert "Argument name : value" in msg
    assert "Expected type : not None" in msg
    assert "Actual   type : NoneType" in msg
    assert "Actual  value : None" in msg


# RTestUtil.test_eq_ne_hash helper

def test_eq_ne_hash_helper_accepts_correct_expectations():
    rtu.test_eq_ne_hash(1, 2, is_equal=False)
    rtu.test_eq_ne_hash(3, 3, is_equal=True)


def test_eq_ne_hash_helper_fails_on_wrong_expectation():
    with pytest.raises(AssertionError):
        rtu.test_eq_ne_hash(1, 2, is_equal=True)


# rambutan3/__init__.py

def test_package_init_is_empty():
    path = os.path.join(os.path.dirname(rambutan3.__file__), "__init__.py")
    with open(path, encoding="utf-8") as f:
        assert f.read().strip() == ""
