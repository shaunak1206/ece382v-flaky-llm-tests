import pytest
from rambutan3.check_args.annotation.INT import INT
from rambutan3.check_args.RCheckArgsError import RCheckArgsError
from rambutan3.check_args.base.RNotNoneTypeMatcher import RNotNoneTypeMatcher
from rambutan3.check_args.base.traverse import RTypeMatcherError


def test_INT_matches_int_but_not_bool():
    # INT should match int instances, but NOT bool (which is subclass of int)
    assert INT.matches(1)
    assert INT.matches(-100)
    assert INT.matches(0)
    assert not INT.matches(True)
    assert not INT.matches(False)
    assert not INT.matches("string")
    assert not INT.matches(None)
    assert not INT.matches(3.14)


def test_INT_matcher_type_and_properties():
    # INT is instance of RRestrictedInstanceMatcher
    # and has allowed_class_or_type_non_empty_tuple and not_allowed_class_or_type_iterable set correctly
    cls = INT.__class__
    allowed = cls.allowed_class_or_type_non_empty_tuple
    not_allowed = cls.not_allowed_class_or_type_iterable
    # Because these attributes come from the class in RRestrictedInstanceMatcher implementation,
    # we expect allowed to contain int and not_allowed to contain bool
    # But since the snippet did not show RRestrictedInstanceMatcher code,
    # Just test what INT.matches does because it is the public behavior
    assert INT.matches(42)
    assert not INT.matches(True)
    assert not INT.matches(False)


def test_RCheckArgsError_is_exception_subclass():
    err = RCheckArgsError()
    assert isinstance(err, Exception)
    assert str(err) == ''
    with pytest.raises(RCheckArgsError):
        raise RCheckArgsError("test error")


def test_RCheckArgsError_message():
    msg = "something went wrong"
    err = RCheckArgsError(msg)
    assert str(err) == msg


def test_RNotNoneTypeMatcher_matches_with_none_and_not_none():
    matcher = RNotNoneTypeMatcher()

    # Should return True for non-None values
    assert matcher.matches(1)
    assert matcher.matches("")
    assert matcher.matches([])
    assert matcher.matches(object())

    # Should return False for None
    assert matcher.matches(None) is False


def test_RNotNoneTypeMatcher_matches_adds_error_if_provided_and_value_none():
    matcher = RNotNoneTypeMatcher()

    class DummyMatcherError:
        def __init__(self):
            self.failed = []

        def add_failed_match(self, matcher, value):
            self.failed.append((matcher, value))

    err = DummyMatcherError()

    # When value is None, matches returns False and adds failed match
    result = matcher.matches(None, matcher_error=err)
    assert result is False
    assert len(err.failed) == 1
    assert err.failed[0][0] is matcher
    assert err.failed[0][1] is None

    # When value is not None, no failed match added
    err2 = DummyMatcherError()
    result2 = matcher.matches(1, matcher_error=err2)
    assert result2 is True
    assert len(err2.failed) == 0


def test_RNotNoneTypeMatcher_eq_and_hash():
    matcher1 = RNotNoneTypeMatcher()
    matcher2 = RNotNoneTypeMatcher()
    class AnotherMatcher:
        pass

    # Instances should be equal to each other
    assert matcher1 == matcher2
    # Should not equal object of other type
    assert matcher1 != AnotherMatcher()
    assert matcher1 != 123

    # Hash is implemented and returns an int
    h1 = hash(matcher1)
    h2 = hash(matcher2)
    assert isinstance(h1, int)
    assert isinstance(h2, int)

    # Hash values for two instances may not necessarily be equal because using super(object, self).__hash__()
    # But just ensure that hash is stable (equal for same instance)
    assert h1 == hash(matcher1)


def test_RNotNoneTypeMatcher_str():
    matcher = RNotNoneTypeMatcher()
    assert str(matcher) == 'not None'


def test_INT_does_not_accept_bool_subclass_instances():
    # Custom subclass of bool should also not be accepted because bool is excluded
    class MyBool(bool):
        pass

    val = MyBool(True)
    # Because MyBool is subclass of bool, it should be excluded
    assert not INT.matches(val)


def test_INT_rejects_non_int_types():
    values = [None, 3.14, "100", [], {}, object()]
    for v in values:
        assert not INT.matches(v)


def test_RNotNoneTypeMatcher_matches_without_error_object():
    m = RNotNoneTypeMatcher()
    # Should not fail or raise even if matcher_error is None
    assert m.matches(None, matcher_error=None) is False
    assert m.matches("value", matcher_error=None) is True


def test_assert_INT_with_edge_cases():
    # INT matches should accept integers but not bool or subclasses of bool (direct or indirect)
    class SubBool(bool):
        pass
    assert INT.matches(0)
    assert INT.matches(-1)
    assert not INT.matches(bool())
    assert not INT.matches(True)
    assert not INT.matches(SubBool(True))

