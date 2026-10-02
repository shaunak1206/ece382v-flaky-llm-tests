"""LLM-generated unit tests for toolz.curried, based on source/docstrings only."""
from toolz import curried


def test_curried_get_basic():
    assert curried.get(0, ('a', 'b')) == 'a'


def test_curried_get_partial_in_map():
    data = [(1, 2), (11, 22), (111, 222)]
    result = list(map(curried.get(0), data))
    assert result == [1, 11, 111]


def test_curried_map_and_filter():
    result = list(curried.map(str)([1, 2, 3]))
    assert result == ['1', '2', '3']
    iseven = lambda x: x % 2 == 0
    result2 = list(curried.filter(iseven)([1, 2, 3, 4]))
    assert result2 == [2, 4]


def test_curried_merge_and_merge_with():
    assert curried.merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}
    assert curried.merge_with(sum)({1: 1}, {1: 10}) == {1: 11}


def test_curried_valmap_keymap_partial():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert curried.valmap(sum)(bills) == {'Alice': 65, 'Bob': 45}
    assert curried.keymap(str.upper)({'a': 1}) == {'A': 1}


def test_curried_assoc_and_dissoc():
    # assoc(d, key, value): partial application fixes d first, then key/value
    set_x = curried.assoc(key='x')
    assert set_x(d={'x': 1}, value=2) == {'x': 2}
    assert curried.dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_curried_take_and_pluck():
    assert list(curried.take(2)([10, 20, 30])) == [10, 20]
    data = [{'id': 1}, {'id': 2}]
    assert list(curried.pluck('id')(data)) == [1, 2]


def test_curried_reduceby_and_groupby():
    from operator import add
    iseven = lambda x: x % 2 == 0
    assert curried.reduceby(iseven, add)([1, 2, 3, 4]) == {False: 4, True: 6}
    assert curried.groupby(len)(['a', 'bb', 'c']) == {1: ['a', 'c'], 2: ['bb']}
