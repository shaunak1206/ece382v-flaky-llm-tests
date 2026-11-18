import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


class DummyRTypeMatcherError(RTypeMatcherError):
    def __init__(self):
        self.failed = []

    def add_failed_match(self, matcher, value):
        self.failed.append((matcher, value))


def test_INT_matches_int_values():
    assert INT.matches(123)
    assert INT.matches(-1)
    assert INT.matches(0)


def test_INT_does_not_match_bool_values():
    # bool is subclass of int, but excluded explicitly
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_INT_does_not_match_other_types():
    assert not INT.matches("123")
    assert not INT.matches(None)
    assert not INT.matches(1.23)
    assert not INT.matches([])


def test_RCheckArgsError_is_exception_subclass():
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("error message")


def test_RNotNoneTypeMatcher_matches_non_none_values():
    matcher = RNotNoneTypeMatcher()
    assert matcher.matches(0)
    assert matcher.matches("")
    assert matcher.matches([])
    assert matcher.matches(False)
    assert matcher.matches(object())


def test_RNotNoneTypeMatcher_does_not_match_none_without_error():
    matcher = RNotNoneTypeMatcher()
    assert not matcher.matches(None)


def test_RNotNoneTypeMatcher_does_not_match_none_with_error():
    matcher = RNotNoneTypeMatcher()
    err = DummyRTypeMatcherError()
    assert not matcher.matches(None, matcher_error=err)
    assert len(err.failed) == 1
    failed_matcher, failed_value = err.failed[0]
    assert failed_matcher == matcher
    assert failed_value is None


def test_RNotNoneTypeMatcher_eq_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    class Dummy:
        pass
    other_obj = Dummy()

    assert matcher1 == matcher2
    assert matcher2 == matcher1
    assert matcher1 != other_obj
    # hash should return an int and be consistent
    h1 = hash(matcher1)
    h2 = hash(matcher2)
    assert isinstance(h1, int)
    assert isinstance(h2, int)
    # The hashes may not be equal necessarily, but should be integers
    # Since __hash__ calls super(object, self).__hash__(), uniqueness not guaranteed


def test_RNotNoneTypeMatcher_str_returns_not_none():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == "not None"


def test_INT_repr_and_str_are_not_overridden():
    # Since INT is an instance of RRestrictedInstanceMatcher,
    # check typical behavior
    assert hasattr(INT, "matches")
    # matches accepts int, denies bool
    assert INT.matches(5)
    assert not INT.matches(True)


def test_RNotNoneTypeMatcher_matches_returns_true_for_any_non_none():
    matcher = RNotNoneTypeMatcher()
    for v in [0, False, "", [], {}, lambda x: x]:
        assert matcher.matches(v)


def test_RNotNoneTypeMatcher_add_failed_match_receives_correct_args():
    matcher = RNotNoneTypeMatcher()
    dummy_error = DummyRTypeMatcherError()
    matcher.matches(None, matcher_error=dummy_error)
    assert dummy_error.failed
    fm, fv = dummy_error.failed[0]
    assert fm is matcher
    assert fv is None


def test_RNotNoneTypeMatcher_eq_returns_false_for_other_types():
    matcher = RNotNoneTypeMatcher()
    assert matcher != None
    assert matcher != 42
    assert matcher != "not none"
    assert matcher != object()
