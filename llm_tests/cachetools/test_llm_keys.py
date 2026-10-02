"""Tests for cachetools.keys: hashkey, methodkey, typedkey, typedmethodkey."""

import pytest

from cachetools.keys import hashkey, methodkey, typedkey, typedmethodkey


def test_hashkey_returns_tuple_of_args():
    key = hashkey(1, 2, 3)
    assert tuple(key) == (1, 2, 3)


def test_hashkey_equal_for_same_args():
    assert hashkey(1, 2) == hashkey(1, 2)


def test_hashkey_different_for_different_args():
    assert hashkey(1, 2) != hashkey(2, 1)


def test_hashkey_is_hashable():
    key = hashkey(1, "a", (2, 3))
    # should not raise
    hash(key)


def test_hashkey_kwargs_order_independent():
    k1 = hashkey(1, a=1, b=2)
    k2 = hashkey(1, b=2, a=1)
    assert k1 == k2


def test_hashkey_does_not_distinguish_types():
    # 1 == 1.0 and hash(1) == hash(1.0), so hashkey treats them the same
    assert hashkey(1) == hashkey(1.0)


def test_methodkey_ignores_self():
    class Foo:
        pass

    a, b = Foo(), Foo()
    assert methodkey(a, 1, 2) == methodkey(b, 1, 2)


def test_typedkey_distinguishes_types():
    # unlike hashkey, typedkey should treat 1 and 1.0 differently
    assert typedkey(1) != typedkey(1.0)


def test_typedkey_same_type_same_value_equal():
    assert typedkey(1, 2) == typedkey(1, 2)


def test_typedkey_kwargs_typed():
    k1 = typedkey(a=1)
    k2 = typedkey(a=1.0)
    assert k1 != k2


def test_typedmethodkey_ignores_self_but_types_args():
    class Foo:
        pass

    a, b = Foo(), Foo()
    assert typedmethodkey(a, 1) == typedmethodkey(b, 1)
    assert typedmethodkey(a, 1) != typedmethodkey(b, 1.0)
