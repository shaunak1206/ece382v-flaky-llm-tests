import pytest

from penman.tree import Tree
from penman.layout import rearrange
from penman.model import Model
from penman.codec import PENMANCodec


def roles(node):
    return [role for role, _ in node[1]]


def make_tree():
    return Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG1', ('b', [('/', 'beta')])),
            (':ARG0', ('c', [('/', 'gamma')])),
        ])
    )


def test_returns_none():
    t = make_tree()
    assert rearrange(t, key=str) is None


def test_no_key_keeps_order():
    t = make_tree()
    rearrange(t)
    assert roles(t.node) == ['/', ':ARG1', ':ARG0']


def test_key_sorts_roles():
    t = make_tree()
    rearrange(t, key=str)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_sort_is_in_place_on_branch_list():
    t = make_tree()
    branches = t.node[1]
    rearrange(t, key=str)
    assert t.node[1] is branches
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_instance_branch_stays_first():
    t = make_tree()
    # key that would place the instance role last if it were sorted
    rearrange(t, key=lambda r: 1 if r == '/' else 0)
    assert roles(t.node)[0] == '/'
    assert t.node[1][0] == ('/', 'alpha')


def test_nested_nodes_are_sorted():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [
                ('/', 'beta'),
                (':ARG2', ('c', [('/', 'gamma')])),
                (':ARG1', ('d', [('/', 'delta')])),
            ])),
        ])
    )
    rearrange(t, key=str)
    inner = t.node[1][1][1]
    assert roles(inner) == ['/', ':ARG1', ':ARG2']


def test_deeply_nested_sorting():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [
                ('/', 'beta'),
                (':ARG0', ('c', [
                    ('/', 'gamma'),
                    (':mod', 'x'),
                    (':ARG9', 'y'),
                ])),
            ])),
        ])
    )
    rearrange(t, key=str)
    c = t.node[1][1][1][1][1][1]
    assert roles(c) == ['/', ':ARG9', ':mod']


def test_stable_for_equal_keys():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':mod', 'one'),
            (':mod', 'two'),
            (':mod', 'three'),
        ])
    )
    rearrange(t, key=str)
    assert [tgt for _, tgt in t.node[1][1:]] == ['one', 'two', 'three']


def test_empty_branches_ok():
    t = Tree(('a', []))
    rearrange(t, key=str)
    assert t.node == ('a', [])


def test_only_instance_branch():
    t = Tree(('a', [('/', 'alpha')]))
    rearrange(t, key=str)
    assert t.node == ('a', [('/', 'alpha')])


def test_no_instance_branch_sorts_everything():
    t = Tree(
        ('a', [
            (':ARG1', 'x'),
            (':ARG0', 'y'),
        ])
    )
    rearrange(t, key=str)
    assert roles(t.node) == [':ARG0', ':ARG1']


def test_attributes_not_moved_by_default():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [('/', 'beta')])),
            (':mod', 'val'),
        ])
    )
    rearrange(t)
    assert roles(t.node) == ['/', ':ARG0', ':mod']


def test_attributes_first_moves_attributes_before_edges():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [('/', 'beta')])),
            (':mod', 'val'),
        ])
    )
    rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', ':mod', ':ARG0']


def test_attributes_first_with_key_sorts_within_groups():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG1', ('b', [('/', 'beta')])),
            (':zeta', 'z'),
            (':ARG0', ('c', [('/', 'gamma')])),
            (':alpha', 'q'),
        ])
    )
    rearrange(t, key=str, attributes_first=True)
    assert roles(t.node) == ['/', ':alpha', ':zeta', ':ARG0', ':ARG1']


def test_attributes_first_reentrant_variable_is_edge():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [('/', 'beta')])),
            (':ARG1', 'b'),
            (':mod', 'val'),
        ])
    )
    rearrange(t, attributes_first=True)
    # 'b' is a variable, so :ARG1 b is an edge, not an attribute
    assert roles(t.node) == ['/', ':mod', ':ARG0', ':ARG1']


def test_attributes_first_applies_to_nested_nodes():
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [
                ('/', 'beta'),
                (':ARG0', ('c', [('/', 'gamma')])),
                (':mod', 'val'),
            ])),
        ])
    )
    rearrange(t, attributes_first=True)
    inner = t.node[1][1][1]
    assert roles(inner) == ['/', ':mod', ':ARG0']


def test_canonical_order_on_parsed_tree():
    c = PENMANCodec()
    t = c.parse(
        '(s / see-01'
        '   :ARG1 (c / cat)'
        '   :ARG0 (d / dog))'
    )
    rearrange(t, key=Model().canonical_order)
    assert roles(t.node) == ['/', ':ARG0', ':ARG1']
    assert t.node[1][1][1][0] == 'd'
    assert t.node[1][2][1][0] == 'c'


def test_rearranged_tree_formats_in_new_order():
    c = PENMANCodec()
    t = c.parse('(a / alpha :ARG1 (b / beta) :ARG0 (c / gamma))')
    rearrange(t, key=str)
    s = c.format(t)
    assert s.index(':ARG0') < s.index(':ARG1')
    assert s.index('(c / gamma)') < s.index('(b / beta)')
