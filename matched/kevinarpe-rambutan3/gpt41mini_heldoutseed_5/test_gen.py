import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


def test_INT_matches_int_but_not_bool():
    # INT should match int values
    assert INT.matches(10)
    assert INT.matches(-5)
    # INT should NOT match bool values, even though bool is subclass of int
    assert not INT.matches(True)
    assert not INT.matches(False)
    # INT should reject other types
    assert not INT.matches("10")
    assert not INT.matches(None)
    assert not INT.matches(10.0)


def test_INT_str_and_repr():
    # Should have __str__ method inherited; test it gives something reasonable
    s = str(INT)
    assert isinstance(s, str)
    # Contains 'int' but not 'bool'
    assert 'int' in s
    assert 'bool' not in s


def test_RCheckArgsError_is_exception():
    err = RCheckArgsError("my error")
    assert isinstance(err, Exception)
    assert str(err) == "my error"


def test_RNotNoneTypeMatcher_matches_none_and_non_none(tmp_path):
    matcher = RNotNoneTypeMatcher()

    # matches should return True for non-None values
    assert matcher.matches(0) is True
    assert matcher.matches("") is True
    assert matcher.matches([]) is True
    assert matcher.matches(False) is True

    # matches should return False for None
    assert matcher.matches(None) is False


def test_RNotNoneTypeMatcher_matches_add_failed_match_called(monkeypatch):
    matcher = RNotNoneTypeMatcher()

    class DummyError:
        def __init__(self):
            self.called = False
            self.called_with = None

        def add_failed_match(self, matcher_obj, value):
            self.called = True
            self.called_with = (matcher_obj, value)

    dummy = DummyError()

    # Call matches with None and dummy matcher_error
    matched = matcher.matches(None, matcher_error=dummy)
    assert matched is False
    assert dummy.called is True
    assert dummy.called_with[0] == matcher
    assert dummy.called_with[1] is None

    # Call matches with non-None value and dummy matcher_error
    dummy.called = False
    matched = matcher.matches(123, matcher_error=dummy)
    assert matched is True
    assert dummy.called is False


def test_RNotNoneTypeMatcher_eq():
    m1 = RNotNoneTypeMatcher()
    m2 = RNotNoneTypeMatcher()

    assert m1 == m2
    assert m1 == m1
    assert m2 == m1

    # Not equal to different types or objects
    assert m1 != object()
    assert m1 != 5
    assert m1 != None


def test_RNotNoneTypeMatcher_hash_returns_int():
    matcher = RNotNoneTypeMatcher()
    h = hash(matcher)
    assert isinstance(h, int)


def test_RNotNoneTypeMatcher_str():
    matcher = RNotNoneTypeMatcher()
    s = str(matcher)
    assert isinstance(s, str)
    assert s == 'not None'


def test_INT_matches_with_custom_class_subclassing_int():
    # Define a custom subclass of int, but not bool
    class MyInt(int):
        pass

    obj = MyInt(3)
    # INT should match MyInt, since allowed tuple includes int
    assert INT.matches(obj)

    # But not if subclass is bool
    class MyBool(bool):
        pass

    b = MyBool(True)
    assert not INT.matches(b)


def test_RCheckArgsError_can_be_raised_and_caught():
    msg = "error message"
    with pytest.raises(RCheckArgsError) as exc_info:
        raise RCheckArgsError(msg)
    assert str(exc_info.value) == msg


def test_RNotNoneTypeMatcher_matches_no_error_object():
    matcher = RNotNoneTypeMatcher()
    # Should return False still for None even if no matcher_error given
    assert matcher.matches(None) is False


def test_INT_does_not_match_other_types():
    # INT should reject float, string, list, etc.
    for val in [3.14, "string", [], {}, object(), None]:
        assert not INT.matches(val)


def test_RNotNoneTypeMatcher_hash_consistent():
    matcher = RNotNoneTypeMatcher()
    h1 = hash(matcher)
    h2 = hash(matcher)
    assert h1 == h2


def test_RNotNoneTypeMatcher_eq_with_other_class_instance():
    matcher = RNotNoneTypeMatcher()

    class Other:
        pass

    other = Other()
    assert (matcher == other) is False


def test_INT_repr_contains_int():
    r = repr(INT)
    assert 'int' in r or 'INT' in r


def test_INT_equality_negotiation():
    # INT is an instance of RRestrictedInstanceMatcher, test __eq__
    # Should be equal to itself
    assert INT == INT
    # Not equal to object()
    assert not (INT == object())
