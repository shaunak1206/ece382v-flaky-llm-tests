import pytest

from penman import layout
from penman.codec import PENMANCodec
from penman.model import Model
from penman.tree import Tree


def roles(node):
    return [role for role, _ in node[1]]


def test_no_key_preserves_order():
    t = Tree(('a', [('/', 'alpha'), (':ARG1', 'x'), (':ARG0', 'y')]))
    layout.rearrange(t)
    assert t.node == ('a', [('/', 'alpha'), (':ARG1', 'x'), (':ARG0', 'y')])


def test_key_sorts_by_role():
    t = Tree(('a', [('/', 'alpha'), (':ARG1', 'x'), (':ARG0', 'y')]))
    layout.rearrange(t, key=lambda role: role)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_returns_none_and_sorts_in_place():
    t = Tree(('a', [('/', 'alpha'), (':b', 'x'), (':a', 'y')]))
    branches = t.node[1]
    result = layout.rearrange(t, key=lambda role: role)
    assert result is None
    assert t.node[1] is branches
    assert roles(t.node) == ['/', ':a', ':b']


def test_instance_branch_stays_first_with_reverse_key():
    t = Tree(('a', [('/', 'alpha'), (':a', 'x'), (':b', 'y')]))
    # a key that would put '/' last if it were sorted
    layout.rearrange(t, key=lambda role: [':b', ':a', '/'].index(role))
    assert roles(t.node) == ['/', ':b', ':a']


def test_empty_branches():
    t = Tree(('a', []))
    layout.rearrange(t, key=lambda role: role)
    assert t.node == ('a', [])
    layout.rearrange(t, attributes_first=True)
    assert t.node == ('a', [])


def test_node_without_instance_branch():
    t = Tree(('a', [(':b', 'x'), (':a', 'y')]))
    layout.rearrange(t, key=lambda role: role)
    assert t.node == ('a', [(':a', 'y'), (':b', 'x')])


def test_nested_nodes_are_sorted_recursively():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':z', ('b', [
            ('/', 'beta'),
            (':y', 'q'),
            (':x', 'p')])),
        (':c', 'v')]))
    layout.rearrange(t, key=lambda role: role)
    assert roles(t.node) == ['/', ':c', ':z']
    nested = t.node[1][2][1]
    assert nested[0] == 'b'
    assert [r for r, _ in nested[1]] == ['/', ':x', ':y']


def test_attributes_first_moves_attributes_before_edges():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':mod', 'val'),
        (':ARG1', ('c', [('/', 'gamma')])),
        (':name', 'foo')]))
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':name', ':ARG0', ':ARG1']


def test_attributes_not_moved_when_attributes_first_false():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':mod', 'val')]))
    layout.rearrange(t)
    assert roles(t.node) == ['/', ':ARG0', ':mod']


def test_attributes_first_combined_with_key():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':z-edge', ('b', [('/', 'beta')])),
        (':b-attr', 'v2'),
        (':a-edge', ('c', [('/', 'gamma')])),
        (':a-attr', 'v1')]))
    layout.rearrange(t, key=lambda role: role, attributes_first=True)
    assert roles(t.node) == ['/', ':a-attr', ':b-attr', ':a-edge', ':z-edge']


def test_attributes_first_treats_variable_reference_as_edge():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':ARG1', 'b'),
        (':mod', 'val')]))
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':ARG0', ':ARG1']


def test_attributes_first_applies_in_nested_nodes():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [
            ('/', 'beta'),
            (':ARG0', ('c', [('/', 'gamma')])),
            (':mod', 'val')]))]))
    layout.rearrange(t, attributes_first=True)
    nested = t.node[1][1][1]
    assert [r for r, _ in nested[1]] == ['/', ':mod', ':ARG0']


def test_sort_is_stable_for_equal_keys():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', 'x'),
        (':ARG0', 'y'),
        (':ARG0', 'z')]))
    layout.rearrange(t, key=lambda role: 0)
    assert [tgt for _, tgt in t.node[1]] == ['alpha', 'x', 'y', 'z']


def test_canonical_order_with_codec():
    c = PENMANCodec()
    t = c.parse(
        '(s / see-01'
        '   :ARG1 (c / cat)'
        '   :ARG0 (d / dog))')
    layout.rearrange(t, key=Model().canonical_order)
    assert c.format(t) == (
        '(s / see-01\n'
        '   :ARG0 (d / dog)\n'
        '   :ARG1 (c / cat))')


def test_attributes_first_with_codec_roundtrip():
    c = PENMANCodec()
    t = c.parse('(a / alpha :ARG0 (b / beta) :mod 5)')
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':ARG0']
    assert c.format(t) == (
        '(a / alpha\n'
        '   :mod 5\n'
        '   :ARG0 (b / beta))')
