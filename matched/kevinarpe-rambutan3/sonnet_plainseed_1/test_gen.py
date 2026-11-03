import pytest

import rambutan3
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RAbstractTypeMatcher import RAbstractTypeMatcher
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


def test_package_importable():
    assert rambutan3.__name__ == 'rambutan3'


def test_int_matches_plain_ints():
    assert INT.matches(0)
    assert INT.matches(5)
    assert INT.matches(-7)
    assert INT.matches(10 ** 30)


def test_int_rejects_bool():
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_int_rejects_other_types():
    for value in (1.0, '1', None, [1], (1,), {1: 1}, b'1', 1j):
        assert not INT.matches(value)


def test_int_matches_int_subclass_not_bool():
    class MyInt(int):
        pass

    assert INT.matches(MyInt(3))


def test_int_is_abstract_type_matcher_and_str():
    assert isinstance(INT, RAbstractTypeMatcher)
    assert str(INT) == 'int'


def test_int_matcher_error_populated_on_bool():
    error = RTypeMatcherError()
    assert INT.matches(True, error) is False
    failed = list(error.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].value is True
    assert failed[0].matcher is INT


def test_int_matcher_error_populated_on_str():
    error = RTypeMatcherError()
    assert INT.matches('x', error) is False
    failed = list(error.failed_match_tuple_set_view)
    assert len(failed) == 1
    assert failed[0].value == 'x'


def test_int_check_arg_returns_value():
    assert INT.check_arg(42, 'x') == 42


def test_int_check_arg_raises_for_bool_with_message():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg(True, 'my_arg')
    msg = exc_info.value.args[0]
    assert isinstance(msg, str)
    assert 'my_arg' in msg
    assert 'int' in msg
    assert 'bool' in msg


def test_int_check_arg_formats_arg_name():
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check_arg('s', 'item {}', 3)
    assert 'item 3' in exc_info.value.args[0]


def test_int_equality():
    assert INT == INT
    assert not (INT != INT)
    assert INT != 5


def test_check_args_error_is_exception():
    assert issubclass(RCheckArgsError, Exception)
    err = RCheckArgsError('boom')
    assert err.args[0] == 'boom'
    with pytest.raises(RCheckArgsError):
        raise err


def test_not_none_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    for value in (0, '', [], False, 'abc', object(), 1.5):
        assert matcher.matches(value) is True


def test_not_none_rejects_none():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_str():
    assert str(RNotNoneTypeMatcher()) == 'not None'


def test_not_none_equality():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    assert a == b
    assert not (a != b)
    assert a != INT
    assert a != 'not None'


def test_not_none_is_abstract_type_matcher():
    assert isinstance(RNotNoneTypeMatcher(), RAbstractTypeMatcher)


def test_not_none_matcher_error_untouched_on_success():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()
    assert matcher.matches(5, error) is True
    assert error.traverse_path_str() == ''


def test_logical_or_with_int():
    combined = INT | RNotNoneTypeMatcher()
    assert combined.matches(3)
    assert combined.matches('abc')
    assert not combined.matches(None)
