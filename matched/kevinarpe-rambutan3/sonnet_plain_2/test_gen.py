import pytest

import rambutan3
from rambutan3 import RTestUtil
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RInstanceMatcher import RInstanceMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.RRestrictedInstanceMatcher import RRestrictedInstanceMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


def test_package_importable():
    assert rambutan3.__name__ == 'rambutan3'


def test_int_matches_ints():
    assert INT.matches(0)
    assert INT.matches(1)
    assert INT.matches(-123456789)
    assert INT.matches(10 ** 30)


def test_int_rejects_bool():
    assert not INT.matches(True)
    assert not INT.matches(False)


@pytest.mark.parametrize('value', [1.0, '1', None, [1], (1,), {1}, {'a': 1}, b'1', 1j])
def test_int_rejects_non_int(value):
    assert not INT.matches(value)


def test_int_matches_int_subclass_that_is_not_bool():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(5))


def test_int_str_is_int():
    assert str(INT) == 'int'


def test_int_is_restricted_instance_matcher():
    assert isinstance(INT, RRestrictedInstanceMatcher)


def test_int_matcher_error_populated_on_failure():
    error = RTypeMatcherError()
    assert not INT.matches('abc', error)
    failed = list(error.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].value == 'abc'


def test_int_matcher_error_populated_for_bool():
    error = RTypeMatcherError()
    assert not INT.matches(True, error)
    failed = list(error.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].matcher is INT
    assert failed[0].value is True


def test_int_check_arg_returns_value():
    assert INT.check_arg(42, 'x') == 42


def test_int_check_arg_raises_with_message():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg('abc', 'my_arg')
    msg = exc_info.value.args[0]
    assert isinstance(msg, str)
    assert 'my_arg' in msg
    assert 'int' in msg
    assert 'str' in msg
    assert "'abc'" in msg


def test_int_check_arg_formats_arg_name():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg(2.5, 'Index {}', 7)
    assert 'Index 7' in exc_info.value.args[0]


def test_int_equality_and_hash():
    other = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(int,),
                                       not_allowed_class_or_type_iterable=(bool,))
    RTestUtil.test_eq_ne_hash(INT, other, is_equal=True)
    RTestUtil.test_eq_ne_hash(INT, INT, is_equal=True)
    different = RRestrictedInstanceMatcher(allowed_class_or_type_non_empty_tuple=(str,),
                                           not_allowed_class_or_type_iterable=())
    RTestUtil.test_eq_ne_hash(INT, different, is_equal=False)


def test_int_not_equal_to_plain_instance_matcher():
    assert INT != RInstanceMatcher(int)
    assert INT != 'int'


def test_not_none_matches_everything_but_none():
    matcher = RNotNoneTypeMatcher()
    for value in (0, False, '', [], {}, 'abc', 1.5, object()):
        assert matcher.matches(value)
    assert not matcher.matches(None)


def test_not_none_matcher_error_recorded_only_on_failure():
    matcher = RNotNoneTypeMatcher()

    ok_error = RTypeMatcherError()
    assert matcher.matches(1, ok_error)

    bad_error = RTypeMatcherError()
    assert not matcher.matches(None, bad_error)
    failed = list(bad_error.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].matcher is matcher
    assert failed[0].value is None


def test_not_none_str_and_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert str(a) == 'not None'
    assert a == b
    assert not (a != b)
    assert a != RInstanceMatcher(int)
    assert a != 'not None'


def test_not_none_check_arg():
    matcher = RNotNoneTypeMatcher()
    assert matcher.check_arg('v', 'arg') == 'v'
    with pytest.raises(RCheckArgsError) as exc_info:
        matcher.check_arg(None, 'some_arg')
    msg = exc_info.value.args[0]
    assert 'some_arg' in msg
    assert 'not None' in msg
    assert 'NoneType' in msg


def test_or_combination_of_matchers():
    combined = INT | RInstanceMatcher(str)
    assert combined.matches(3)
    assert combined.matches('s')
    assert not combined.matches(True)
    assert not combined.matches(None)
    assert str(combined) == 'int | str'


def test_rcheckargserror_is_exception():
    assert issubclass(RCheckArgsError, Exception)
    err = RCheckArgsError('boom')
    assert err.args[0] == 'boom'
    with pytest.raises(RCheckArgsError):
        raise err
