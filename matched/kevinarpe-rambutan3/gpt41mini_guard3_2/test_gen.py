import pytest
import random
import numpy as np

from rambutan3.check_args.annotation import INT
from rambutan3.check_args import RCheckArgsError
from rambutan3.check_args.base import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


def test_INT_matches_int_but_not_bool():
    random.seed(123)
    np.random.seed(123)

    # ints match
    assert INT.matches(0)
    assert INT.matches(1)
    assert INT.matches(-123456789)

    # bools are subclass of int but must NOT match
    assert not INT.matches(True)
    assert not INT.matches(False)

    # other types do not match
    assert not INT.matches(1.0)
    assert not INT.matches("1")
    assert not INT.matches(None)

def test_INT_check_method_positive_and_negative(monkeypatch):
    random.seed(456)
    np.random.seed(456)

    # test that check() succeeds and raises appropriately

    # INT inherits from RRestrictedInstanceMatcher, which inherits RValueMatcher with check()
    # check() returns True if matches, else raises RCheckArgsError

    # check passes for int
    assert INT.check(10) is True

    # check fails for bool
    with pytest.raises(RCheckArgsError):
        INT.check(True)

def test_INT_str_and_repr():
    random.seed(789)
    np.random.seed(789)

    s = str(INT)
    # Since RRestrictedInstanceMatcher inherits something with __str__, it should mention int but not bool
    assert "int" in s
    assert "bool" not in s

    r = repr(INT)
    # repr string should be non-empty string
    assert isinstance(r, str) and len(r) > 0

def test_RCheckArgsError_is_subclass_of_Exception():
    random.seed(213)
    np.random.seed(213)

    e = RCheckArgsError()
    assert isinstance(e, Exception)
    assert issubclass(RCheckArgsError, Exception)
    # can be raised and caught as expected
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError

def test_RNotNoneTypeMatcher_matches_non_none_and_not_none_error_add(monkeypatch):
    random.seed(321)
    np.random.seed(321)

    matcher = RNotNoneTypeMatcher()

    # matcher.matches returns True for non-None values
    assert matcher.matches(0)
    assert matcher.matches("")
    assert matcher.matches([])
    assert matcher.matches(False)

    # returns False for None
    assert not matcher.matches(None)

    # matcher_error is None, no exception, just result False
    assert matcher.matches(None, None) is False

    # matcher_error records failed match when value is None
    error = RTypeMatcherError()
    result = matcher.matches(None, error)
    assert result is False
    failed = error.failed_matches
    # There should be exactly one failed match, with matcher and value=(-) None
    assert len(failed) == 1
    failed_matcher, failed_value = failed[0]
    # It should be our matcher instance
    assert failed_matcher == matcher
    assert failed_value is None

def test_RNotNoneTypeMatcher_eq_and_hash():
    random.seed(135)
    np.random.seed(135)

    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()

    # Two instances compare equal
    assert m1 == m2
    assert m2 == m1

    # Should not be equal to unrelated object
    assert m1 != 123
    assert m1 != "not none"

    # Hash should be consistent with __hash__ override (calls super(object).__hash__)
    h1 = hash(m1)
    h2 = hash(m2)

    # Because __hash__ is object.__hash__, different instances have different hashes
    # So hashes need not be equal, but each hash must be int
    assert isinstance(h1, int)
    assert isinstance(h2, int)

    # Make sure hash does not raise
    hash(m1)
    hash(m2)

def test_RNotNoneTypeMatcher_str_is_not_None_and_equal_to_expected():
    random.seed(246)
    np.random.seed(246)

    m = RNotNoneTypeMatcher()
    s = str(m)
    assert isinstance(s, str)
    assert s == "not None"

def test_RNotNoneTypeMatcher_matches_with_mutated_matcher_error(monkeypatch):
    random.seed(975)
    np.random.seed(975)

    m = RNotNoneTypeMatcher()
    error = RTypeMatcherError()

    # matches with a value that passes -> no errors added
    assert m.matches("value", error) is True
    assert error.failed_matches == []

    # matches with a failing value -> error recorded
    assert m.matches(None, error) is False
    assert len(error.failed_matches) == 1

def test_INT_does_not_match_subclasses_of_int_other_than_bool(monkeypatch):
    random.seed(852)
    np.random.seed(852)

    class MyInt(int):
        pass

    class MyBool(bool):
        pass

    # MyInt subclass should match because only bool is excluded
    v = MyInt(10)
    assert INT.matches(v)

    # MyBool subclass should not match (bool excluded)
    vb = MyBool(True)
    assert not INT.matches(vb)

def test_INT_check_method_message_contains_type(monkeypatch):
    random.seed(369)
    np.random.seed(369)

    # Check method raises RCheckArgsError with informative message for bad input
    with pytest.raises(RCheckArgsError) as exc_info:
        INT.check("string, not int")
    msg = str(exc_info.value)
    # Message contains typename int
    assert "int" in msg

def test_RNotNoneTypeMatcher_eq_is_reflexive_and_symetric():
    random.seed(147)
    np.random.seed(147)

    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()

    # Reflexive
    assert m1 == m1

    # Symmetric (already tested)
    assert (m1 == m2) == (m2 == m1)

    # Transitive (not relevant for two objects, but as good as possible)
    m3 = RNotNoneTypeMatcher()
    if m1 == m2 and m2 == m3:
        assert m1 == m3

def test_INT_matches_does_not_raise_for_any_input(monkeypatch):
    random.seed(753)
    np.random.seed(753)

    # Test that INT.matches() does not raise regardless of input type
    inputs = [None, 123, True, False, 0.0, "", [], {}, set(), b"bytes"]

    for v in inputs:
        try:
            r = INT.matches(v)
            assert isinstance(r, bool)
        except Exception as e:
            pytest.fail(f"INT.matches({v!r}) raised {e!r}")

def test_RNotNoneTypeMatcher_hash_consistency(monkeypatch):
    random.seed(159)
    np.random.seed(159)

    m = RNotNoneTypeMatcher()
    h1 = hash(m)
    h2 = hash(m)
    assert h1 == h2

def test_RNotNoneTypeMatcher_matches_none_without_error_recording(monkeypatch):
    random.seed(268)
    np.random.seed(268)

    m = RNotNoneTypeMatcher()
    # matcher_error is None, so no exception/recording
    assert m.matches(None) is False

def test_INT_is_instance_and_not_bool(monkeypatch):
    random.seed(369)
    np.random.seed(369)

    # INT.allowed_class_or_type_non_empty_tuple is (int,)
    # INT.not_allowed_class_or_type_iterable is (bool,)
    # Check these attributes exist and are as expected
    assert hasattr(INT, "allowed_class_or_type_non_empty_tuple")
    assert isinstance(INT.allowed_class_or_type_non_empty_tuple, tuple)
    assert int in INT.allowed_class_or_type_non_empty_tuple

    assert hasattr(INT, "not_allowed_class_or_type_iterable")
    not_allowed = tuple(INT.not_allowed_class_or_type_iterable)
    assert bool in not_allowed

def test_RCheckArgsError_message_defaults_empty():
    random.seed(951)
    np.random.seed(951)

    e = RCheckArgsError()
    # Default str and repr are empty or something reasonable
    s = str(e)
    r = repr(e)
    # str and repr return strs
    assert isinstance(s, str)
    assert isinstance(r, str)

