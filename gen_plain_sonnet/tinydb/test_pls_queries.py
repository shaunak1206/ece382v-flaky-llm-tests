import pytest

from tinydb import Query, where
from tinydb.queries import QueryInstance


def test_eq_and_ne():
    assert (where('a') == 1)({'a': 1})
    assert not (where('a') == 1)({'a': 2})
    assert (where('a') != 1)({'a': 2})
    assert not (where('a') != 1)({'a': 1})
    # An unresolvable path never matches, not even for !=
    assert not (where('a') == 1)({'b': 1})
    assert not (where('a') != 1)({})
    # Containers compare by value
    assert (where('a') == {'x': [1, 2]})({'a': {'x': [1, 2]}})
    assert not (where('a') == {'x': [1, 2]})({'a': {'x': [2, 1]}})


def test_ordering_comparisons():
    doc = {'n': 5}
    assert (where('n') < 6)(doc)
    assert not (where('n') < 5)(doc)
    assert (where('n') <= 5)(doc)
    assert not (where('n') <= 4)(doc)
    assert (where('n') > 4)(doc)
    assert not (where('n') > 5)(doc)
    assert (where('n') >= 5)(doc)
    assert not (where('n') >= 6)(doc)
    assert not (where('n') < 100)({'other': 1})


def test_path_resolution_variants():
    nested = Query().outer.inner == 3
    assert nested({'outer': {'inner': 3}})
    assert not nested({'outer': {'inner': 4}})
    assert not nested({'outer': 5})
    assert (Query()['logged-in'] == True)({'logged-in': True})  # noqa: E712
    assert (where('x') == 1) == (Query()['x'] == 1)
    assert not (Query().a['b'] == 1)({'a': [1, 2]})


def test_exists():
    q = where('a').exists()
    assert q({'a': None})
    assert q({'a': 0})
    assert not q({})


def test_matches_requires_match_at_start_but_search_does_not():
    assert where('s').matches(r'\d+')({'s': '123abc'})
    assert not where('s').matches(r'\d+')({'s': 'abc123'})
    assert where('s').search(r'\d+')({'s': 'abc123'})
    assert not where('s').search(r'\d+')({'s': 'abc'})


def test_regex_queries_non_strings_and_flags():
    assert not where('s').matches(r'.*')({'s': 12})
    assert not where('s').search(r'.*')({'s': None})
    assert where('s').matches('hello', 2)({'s': 'HELLO'})  # re.IGNORECASE == 2
    assert not where('s').matches('hello')({'s': 'HELLO'})


def test_test_with_extra_args():
    q = where('n').test(lambda v, lo, hi: lo <= v <= hi, 1, 5)
    assert q({'n': 3})
    assert not q({'n': 9})


def test_any():
    sub = where('items').any(where('k') == 1)
    assert sub({'items': [{'k': 0}, {'k': 1}]})
    assert not sub({'items': [{'k': 0}, {'k': 2}]})
    assert not sub({'items': []})
    members = where('vals').any([1, 2, 3])
    assert members({'vals': [3, 4, 5]})
    assert not members({'vals': [4, 5]})
    assert not members({'vals': 5})


def test_all():
    sub = where('items').all(where('k') == 1)
    assert sub({'items': [{'k': 1}, {'k': 1}]})
    assert not sub({'items': [{'k': 1}, {'k': 2}]})
    members = where('vals').all([1, 2, 3])
    assert members({'vals': [1, 2, 3, 4, 5]})
    assert not members({'vals': [1, 2]})


def test_one_of():
    q = where('c').one_of(['red', 'green'])
    assert q({'c': 'red'})
    assert not q({'c': 'blue'})


def test_fragment_matches_subset_of_keys():
    q = Query().fragment({'a': 1, 'b': 2})
    assert q({'a': 1, 'b': 2, 'c': 3})
    assert not q({'a': 1, 'b': 3})
    assert not q({'a': 1})


def test_noop_always_true_and_composable():
    assert Query().noop()({})
    q = Query().noop() & (where('a') == 1)
    assert q({'a': 1})
    assert not q({'a': 2})


def test_map_applies_function_on_path():
    q = Query().name.map(len) == 3
    assert q({'name': 'abc'})
    assert not q({'name': 'abcd'})


def test_and_or_not_combinators():
    a = where('a') == 1
    b = where('b') == 2
    doc = {'a': 1, 'b': 3}
    assert not (a & b)(doc)
    assert (a | b)(doc)
    assert (~b)(doc)
    assert not (~a)(doc)
    assert (~a)({})  # negation of a failed lookup is True


def test_query_without_path_errors():
    with pytest.raises(RuntimeError):
        Query()({})
    with pytest.raises(ValueError):
        Query() == 1
    with pytest.raises(ValueError):
        Query().exists()


def test_equal_queries_have_equal_hashes():
    q1 = where('a') == 1
    q2 = where('a') == 1
    assert q1 == q2
    assert hash(q1) == hash(q2)
    assert not (q1 == (where('a') == 2))
    assert not (q1 == 5)
    a = where('a') == 1
    b = where('b') == 2
    assert (a & b) == (b & a)
    assert hash(a | b) == hash(b | a)
    assert not ((a & b) == (a | b))


def test_uncacheable_queries():
    mapped = Query().name.map(len) == 3
    assert not mapped.is_cacheable()
    assert not (mapped & (where('a') == 1)).is_cacheable()
    assert not (~mapped).is_cacheable()
    assert (where('a') == 1).is_cacheable()

    class Unhashable:
        __hash__ = None

    q = where('a') == Unhashable()
    assert isinstance(q, QueryInstance)
    assert not q.is_cacheable()
    assert not q({'a': 1})


def test_repr_of_query_and_query_instance():
    assert repr(Query()) == 'Query()'
    assert repr(where('a') == 1).startswith('QueryImpl')
