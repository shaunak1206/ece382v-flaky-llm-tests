import pytest

from rambutan3 import __name__ as rambutan3_package_name
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


def test_package_import_name():
    assert rambutan3_package_name == "rambutan3"


def test_rcheckargserror_is_exception_subclass():
    assert issubclass(RCheckArgsError, Exception)


def test_rcheckargserror_can_be_raised_and_caught():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("bad args")


def test_int_matcher_allows_regular_int():
    assert INT.matches(0) is True
    assert INT.matches(123) is True
    assert INT.matches(-999) is True


def test_int_matcher_rejects_bool():
    assert INT.matches(True) is False
    assert INT.matches(False) is False


def test_int_matcher_rejects_non_int_types():
    assert INT.matches("1") is False
    assert INT.matches(1.5) is False
    assert INT.matches(None) is False
    assert INT.matches([]) is False


def test_int_matcher_str_representation_is_informative():
    assert str(INT) == "int"


def test_int_matcher_is_same_object_on_multiple_imports():
    from rambutan3.check_args.annotation.INT import INT as int_again

    assert INT is int_again


def test_not_none_matcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches(object()) is True


def test_not_none_matcher_rejects_none_without_error_object():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(None) is False


def test_not_none_matcher_adds_failed_match_to_error():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()

    result = matcher.matches(None, error)

    assert result is False
    assert len(error.failed_matches) == 1
    failed_matcher, failed_value = error.failed_matches[0]
    assert failed_matcher == matcher
    assert failed_value is None


def test_not_none_matcher_does_not_add_failed_match_for_non_none():
    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()

    result = matcher.matches(5, error)

    assert result is True
    assert len(error.failed_matches) == 0


def test_not_none_matcher_equality_is_type_based():
    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()

    assert a == b
    assert a == a
    assert not (a != b)


def test_not_none_matcher_not_equal_to_unrelated_object():
    matcher = RNotNoneTypeMatcher()

    assert matcher != None  # noqa: E711
    assert matcher != "not None"


def test_not_none_matcher_hash_is_int_and_usable_in_collections():
    matcher = RNotNoneTypeMatcher()
    value = hash(matcher)

    assert isinstance(value, int)
    assert matcher in {matcher}
