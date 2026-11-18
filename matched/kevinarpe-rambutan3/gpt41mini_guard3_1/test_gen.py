import random
import pytest

from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse.RTypeMatcherError import RTypeMatcherError


def test_INT_matches_int_but_not_bool():
    random.seed(123)  # seed random as instructed, although not used here
    
    # int values should match
    assert INT.matches(42)
    assert INT.matches(-1)
    assert INT.matches(0)

    # bool values should NOT match, since bool subclasses int but are excluded
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_INT_str_repr_and_hash_behavior():
    # INT should be an instance of RRestrictedInstanceMatcher
    # We test some properties, like that it matches ints and excludes bools
    random.seed(0)

    # No direct str or hash provided explicitly, so test default behaviors by proxy
    # Str consistency: RRestrictedInstanceMatcher likely does not override __str__, so test type
    # Just sanity checks
    s = str(INT)
    assert isinstance(s, str)
    # hash should be stable by Python definition for this frozen instance; test type and hash determinism within one run
    h1 = hash(INT)
    h2 = hash(INT)
    assert isinstance(h1, int)
    assert h1 == h2


def test_RCheckArgsError_inheritance_and_message():
    random.seed(1)
    
    err = RCheckArgsError()
    assert isinstance(err, Exception)
    assert str(err) == ''  # default exception message is empty string

    err2 = RCheckArgsError('message')
    assert str(err2) == 'message'


def test_RNotNoneTypeMatcher_matches_non_none_values(monkeypatch):
    random.seed(2)

    matcher = RNotNoneTypeMatcher()

    # value None should NOT match
    assert not matcher.matches(None)

    # other values should match
    assert matcher.matches(0)
    assert matcher.matches('')
    assert matcher.matches([])
    assert matcher.matches(object())
    assert matcher.matches(False)


def test_RNotNoneTypeMatcher_matches_adds_error_when_non_match(monkeypatch):
    random.seed(3)

    matcher = RNotNoneTypeMatcher()
    error = RTypeMatcherError()

    # Initially error should have no recorded failures
    assert len(error.failed_matches) == 0

    # Passing None with error -> error should record failure
    result = matcher.matches(None, matcher_error=error)
    assert not result
    # The RTypeMatcherError should have one failed match recorded with matcher and value None
    fm = error.failed_matches
    assert len(fm) == 1
    recorded_matcher, recorded_value = fm[0]
    assert recorded_matcher == matcher
    assert recorded_value is None

    # Passing non-None with error -> no failure added
    previous_len = len(error.failed_matches)
    result2 = matcher.matches(5, matcher_error=error)
    assert result2
    assert len(error.failed_matches) == previous_len


def test_RNotNoneTypeMatcher_eq_and_hash(monkeypatch):
    random.seed(4)

    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()
    
    # They should be equal to each other
    assert a == b
    assert b == a
    # Should not be equal to other types
    assert a != object()
    assert a != None

    # hash should return object hash, so test that it returns int
    h = hash(a)
    assert isinstance(h, int)

    # Different instances may have different hashes because default object __hash__ is based on id
    h2 = hash(b)
    # We cannot assert equality of hashes, but they must be int and stable within run
    assert isinstance(h2, int)


def test_RNotNoneTypeMatcher_str_representation():
    random.seed(5)

    matcher = RNotNoneTypeMatcher()
    s = str(matcher)
    assert s == 'not None'


def test_INT_does_not_match_unrelated_types():
    random.seed(6)

    # Should only match int and exclude bool, so any other types return False
    not_ints = [None, 3.14, "string", [], {}, set(), (1,), 1+2j, lambda x: x]

    for x in not_ints:
        assert not INT.matches(x)


def test_INT_rejects_bool_even_if_casting(monkeypatch):
    random.seed(7)

    # Even if user tries to cast bool to int, passing bool object directly is rejected
    assert not INT.matches(True)
    assert not INT.matches(False)

    # However int(True) == 1, so that should match because it is int, not bool
    assert INT.matches(int(True))
    assert INT.matches(int(False))


def test_RCheckArgsError_can_be_raised_and_caught():
    random.seed(8)

    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError()

    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError('error message')


def test_RNotNoneTypeMatcher_matches_with_none_and_no_error():
    random.seed(9)

    matcher = RNotNoneTypeMatcher()
    # Providing matcher_error as None, matches returns False but no error is thrown
    res = matcher.matches(None, matcher_error=None)
    assert res is False

    # Matches with non-None returns True always
    for v in [0, '', [], object()]:
        assert matcher.matches(v, matcher_error=None) is True


def test_RNotNoneTypeMatcher_eq_is_symmetric_and_reflexive():
    random.seed(10)

    a = RNotNoneTypeMatcher()
    b = RNotNoneTypeMatcher()

    assert a == a
    assert b == b
    assert a == b
    assert b == a


def test_INT_matches_edge_case_with_subclasses_of_int():
    random.seed(11)

    class MyInt(int):
        pass

    myint_instance = MyInt(5)

    # MyInt is a subclass of int, and bool is excluded.
    # Test if INT.matches accepts MyInt instances (should, since allowed_class_or_type_non_empty_tuple=(int,))
    # Because in isinstance(myint_instance, (int,)) is True.
    assert INT.matches(myint_instance)

    # But bool subclass is excluded explicitly
    assert not INT.matches(True)

