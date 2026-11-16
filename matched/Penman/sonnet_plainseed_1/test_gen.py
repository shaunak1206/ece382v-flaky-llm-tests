import pytest

from penman import layout
from penman.codec import PENMANCodec
from penman.model import Model
from penman.tree import Tree


def roles(node):
    return [role for role, _ in node[1]]


def ident(role):
    return role


def test_sorts_branches_by_key():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG2', 'z'),
                    (':ARG0', 'x'),
                    (':ARG1', 'y')]))
    layout.rearrange(t, key=ident)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1', ':ARG2']


def test_returns_none_and_sorts_in_place():
    branches = [('/', 'alpha'), (':b', 'x'), (':a', 'y')]
    t = Tree(('a', branches))
    result = layout.rearrange(t, key=ident)
    assert result is None
    assert t.node[1] is branches
    assert [r for r, _ in branches] == ['/', ':a', ':b']


def test_instance_branch_stays_first_regardless_of_key():
    t = Tree(('a', [('/', 'alpha'), (':b', 'x'), (':a', 'y')]))
    # key that would put '/' last if it were sorted
    layout.rearrange(t, key=lambda r: 1 if r == '/' else 0)
    assert roles(t.node) == ['/', ':b', ':a'] or roles(t.node) == ['/', ':a', ':b']
    assert t.node[1][0] == ('/', 'alpha')


def test_key_none_preserves_order():
    original = [('/', 'alpha'), (':z', 'x'), (':a', 'y'), (':m', 'w')]
    t = Tree(('a', list(original)))
    layout.rearrange(t)
    assert t.node[1] == original


def test_stable_for_equal_keys():
    t = Tree(('a', [('/', 'alpha'),
                    (':x', '1'),
                    (':y', '2'),
                    (':x', '3'),
                    (':y', '4')]))
    layout.rearrange(t, key=ident)
    assert t.node[1] == [('/', 'alpha'),
                         (':x', '1'),
                         (':x', '3'),
                         (':y', '2'),
                         (':y', '4')]


def test_recurses_into_nested_nodes():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG1', ('b', [('/', 'beta'),
                                     (':ARG1', 'q'),
                                     (':ARG0', 'p')])),
                    (':ARG0', ('c', [('/', 'gamma'),
                                     (':mod', 'm'),
                                     (':ARG0', 'n')]))]))
    layout.rearrange(t, key=ident)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']
    c = t.node[1][1][1]
    b = t.node[1][2][1]
    assert c[0] == 'c'
    assert roles(c) == ['/', ':ARG0', ':mod']
    assert b[0] == 'b'
    assert roles(b) == ['/', ':ARG0', ':ARG1']


def test_deeply_nested_recursion():
    t = Tree(('a', [('/', 'alpha'),
                    (':x', ('b', [('/', 'beta'),
                                  (':x', ('c', [('/', 'gamma'),
                                                (':z', '1'),
                                                (':y', '2')])),
                                  (':w', '3')]))]))
    layout.rearrange(t, key=ident)
    b = t.node[1][1][1]
    assert roles(b) == ['/', ':w', ':x']
    c = b[1][2][1]
    assert roles(c) == ['/', ':y', ':z']


def test_empty_branches_node():
    t = Tree(('a', []))
    layout.rearrange(t, key=ident)
    assert t.node == ('a', [])


def test_node_without_instance_is_fully_sorted():
    t = Tree(('a', [(':c', '1'), (':b', '2'), (':a', '3')]))
    layout.rearrange(t, key=ident)
    assert roles(t.node) == [':a', ':b', ':c']


def test_key_is_called_only_with_roles_not_instance():
    seen = []

    def key(role):
        seen.append(role)
        return role

    t = Tree(('a', [('/', 'alpha'), (':b', 'x'), (':a', 'y')]))
    layout.rearrange(t, key=key)
    assert '/' not in seen
    assert set(seen) == {':a', ':b'}


def test_attributes_first_with_no_key_is_stable_partition():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG0', ('b', [('/', 'beta')])),
                    (':mod', 'x'),
                    (':ARG1', 'a'),
                    (':polarity', '-')]))
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':polarity', ':ARG0', ':ARG1']


def test_attributes_first_with_key_sorts_within_groups():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG1', ('b', [('/', 'beta')])),
                    (':zattr', 'x'),
                    (':ARG0', ('c', [('/', 'gamma')])),
                    (':aattr', 'y')]))
    layout.rearrange(t, key=ident, attributes_first=True)
    assert roles(t.node) == ['/', ':aattr', ':zattr', ':ARG0', ':ARG1']


def test_attributes_first_false_mixes_by_key_only():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG1', ('b', [('/', 'beta')])),
                    (':zattr', 'x'),
                    (':ARG0', ('c', [('/', 'gamma')])),
                    (':aattr', 'y')]))
    layout.rearrange(t, key=ident, attributes_first=False)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1', ':aattr', ':zattr']


def test_attributes_first_treats_reentrant_variable_as_edge():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG0', 'a'),
                    (':mod', 'x')]))
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':ARG0']


def test_attributes_first_applies_in_nested_nodes():
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG0', ('b', [('/', 'beta'),
                                     (':ARG0', 'a'),
                                     (':mod', 'x')]))]))
    layout.rearrange(t, attributes_first=True)
    b = t.node[1][1][1]
    assert roles(b) == ['/', ':mod', ':ARG0']


def test_canonical_order_with_codec():
    c = PENMANCodec()
    t = c.parse('(s / see-01'
                '   :ARG1 (c / cat)'
                '   :ARG0 (d / dog))')
    layout.rearrange(t, key=Model().canonical_order)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']
    assert t.node[1][1][1][0] == 'd'
    assert t.node[1][2][1][0] == 'c'


def test_rearranged_tree_formats_with_new_order():
    c = PENMANCodec()
    t = c.parse('(a / alpha :b (x / ex) :a (y / why))')
    layout.rearrange(t, key=ident)
    s = c.format(t)
    assert s.index(':a') < s.index(':b')
    assert s.index('y / why') < s.index('x / ex')
