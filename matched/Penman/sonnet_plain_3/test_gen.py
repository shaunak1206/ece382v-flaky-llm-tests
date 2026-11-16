import pytest

from penman import layout
from penman.codec import PENMANCodec
from penman.model import Model
from penman.tree import Tree


def _by_last_char(role):
    return role[-1]


def test_sorts_branches_by_key():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG2', 'x'),
        (':ARG0', 'y'),
        (':ARG1', 'z'),
    ]))
    layout.rearrange(t, key=_by_last_char)
    assert t.node == ('a', [
        ('/', 'alpha'),
        (':ARG0', 'y'),
        (':ARG1', 'z'),
        (':ARG2', 'x'),
    ])


def test_returns_none_and_modifies_in_place():
    t = Tree(('a', [('/', 'alpha'), (':ARG1', 'x'), (':ARG0', 'y')]))
    branches = t.node[1]
    result = layout.rearrange(t, key=_by_last_char)
    assert result is None
    # the same list object was modified
    assert t.node[1] is branches
    assert branches == [('/', 'alpha'), (':ARG0', 'y'), (':ARG1', 'x')]


def test_no_key_leaves_order_unchanged():
    branches = [
        ('/', 'alpha'),
        (':ARG2', 'x'),
        (':ARG0', 'y'),
        (':ARG1', 'z'),
    ]
    t = Tree(('a', list(branches)))
    layout.rearrange(t)
    assert t.node == ('a', branches)


def test_instance_branch_stays_first_with_reverse_key():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', 'x'),
        (':ARG1', 'y'),
    ]))
    layout.rearrange(t, key=lambda r: -int(r[-1]))
    assert t.node == ('a', [
        ('/', 'alpha'),
        (':ARG1', 'y'),
        (':ARG0', 'x'),
    ])


def test_sort_is_stable_for_equal_keys():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':mod', 'c'),
        (':mod', 'a'),
        (':mod', 'b'),
    ]))
    layout.rearrange(t, key=lambda r: 0)
    assert [target for _, target in t.node[1][1:]] == ['c', 'a', 'b']


def test_recursively_sorts_nested_nodes():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG1', ('b', [
            ('/', 'beta'),
            (':ARG1', 'q'),
            (':ARG0', ('c', [
                ('/', 'gamma'),
                (':ARG1', 'm'),
                (':ARG0', 'n'),
            ])),
        ])),
        (':ARG0', 'z'),
    ]))
    layout.rearrange(t, key=_by_last_char)
    assert t.node == ('a', [
        ('/', 'alpha'),
        (':ARG0', 'z'),
        (':ARG1', ('b', [
            ('/', 'beta'),
            (':ARG0', ('c', [
                ('/', 'gamma'),
                (':ARG0', 'n'),
                (':ARG1', 'm'),
            ])),
            (':ARG1', 'q'),
        ])),
    ])


def test_empty_branches_node():
    t = Tree(('a', []))
    layout.rearrange(t, key=_by_last_char)
    assert t.node == ('a', [])
    layout.rearrange(t, attributes_first=True)
    assert t.node == ('a', [])


def test_node_without_instance_branch_is_sorted():
    t = Tree(('a', [
        (':ARG1', 'x'),
        (':ARG0', 'y'),
    ]))
    layout.rearrange(t, key=_by_last_char)
    assert t.node == ('a', [
        (':ARG0', 'y'),
        (':ARG1', 'x'),
    ])


def test_attributes_first_moves_attributes_before_edges():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':mod', 'x'),
        (':ARG1', 'b'),
        (':polarity', '-'),
    ]))
    layout.rearrange(t, attributes_first=True)
    assert t.node == ('a', [
        ('/', 'alpha'),
        (':mod', 'x'),
        (':polarity', '-'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':ARG1', 'b'),
    ])


def test_attributes_not_moved_without_flag():
    branches = [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':mod', 'x'),
    ]
    t = Tree(('a', list(branches)))
    layout.rearrange(t)
    assert t.node == ('a', branches)


def test_attributes_first_with_key_sorts_within_groups():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG2', ('b', [('/', 'beta')])),
        (':attr2', 'y'),
        (':ARG1', ('c', [('/', 'gamma')])),
        (':attr1', 'x'),
    ]))
    layout.rearrange(t, key=_by_last_char, attributes_first=True)
    roles = [role for role, _ in t.node[1]]
    assert roles == ['/', ':attr1', ':attr2', ':ARG1', ':ARG2']


def test_attributes_first_applies_to_nested_nodes():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [
            ('/', 'beta'),
            (':ARG0', ('c', [('/', 'gamma')])),
            (':polarity', '-'),
        ])),
    ]))
    layout.rearrange(t, attributes_first=True)
    inner = t.node[1][1][1]
    assert [role for role, _ in inner[1]] == ['/', ':polarity', ':ARG0']


def test_variable_atomic_target_counts_as_edge():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', 'b'),
        (':mod', 'val'),
        (':ARG1', ('b', [('/', 'beta')])),
    ]))
    layout.rearrange(t, attributes_first=True)
    assert [role for role, _ in t.node[1]] == [
        '/', ':mod', ':ARG0', ':ARG1']


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


def test_rearrange_parsed_tree_attributes_first_codec():
    c = PENMANCodec()
    t = c.parse(
        '(a / alpha'
        '   :ARG0 (b / beta)'
        '   :mod x)')
    layout.rearrange(t, attributes_first=True)
    assert c.format(t) == (
        '(a / alpha\n'
        '   :mod x\n'
        '   :ARG0 (b / beta))')


def test_rearrange_is_idempotent():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG1', ('b', [('/', 'beta'), (':ARG1', 'q'), (':ARG0', 'r')])),
        (':ARG0', 'z'),
        (':mod', 'x'),
    ]))
    layout.rearrange(t, key=_by_last_char, attributes_first=True)
    first = repr(t.node)
    layout.rearrange(t, key=_by_last_char, attributes_first=True)
    assert repr(t.node) == first
