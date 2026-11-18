import pytest

from penman import layout
from penman.tree import Tree
from penman.model import Model
from penman.exceptions import LayoutError


def make_tree():
    return Tree(
        ('s', [
            ('ARG1', ('c', [
                ('/', 'cat'),
                ('mod', 'small'),
            ])),
            ('ARG0', ('d', [
                ('/', 'dog'),
                ('mod', 'big'),
            ])),
            ('/', 'see-01'),
        ])
    )


def test_rearrange_places_instance_branch_first():
    t = Tree(('x', [('ARG1', 'y'), ('/', 'x-thing'), ('ARG0', 'z')]))
    layout.rearrange(t)
    assert t.node[1][0] == ('/', 'x-thing')


def test_rearrange_sorts_atomic_branches_before_node_branches_by_default():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG1', ('c', [('/', 'cat')])),
        ('attr', 'val'),
        ('ARG0', ('d', [('/', 'dog')])),
    ]))
    layout.rearrange(t)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('attr', 'val'),
        ('ARG0', ('d', [('/', 'dog')])),
        ('ARG1', ('c', [('/', 'cat')])),
    ]


def test_rearrange_sorts_by_key_among_same_branch_type():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG2', 'two'),
        ('ARG0', 'zero'),
        ('ARG1', 'one'),
    ]))
    layout.rearrange(t, key=lambda role: role)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('ARG0', 'zero'),
        ('ARG1', 'one'),
        ('ARG2', 'two'),
    ]


def test_rearrange_recurses_into_nested_nodes():
    t = make_tree()
    layout.rearrange(t, key=Model().canonical_order)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('ARG0', ('d', [
            ('/', 'dog'),
            ('mod', 'big'),
        ])),
        ('ARG1', ('c', [
            ('/', 'cat'),
            ('mod', 'small'),
        ])),
    ]


def test_rearrange_attributes_first_puts_nonvariable_atomic_branches_before_variable_targets():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG0', 'd'),
        ('attr', 'val'),
        ('ARG1', ('x', [('/', 'thing')])),
    ]))
    layout.rearrange(t, attributes_first=True)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('attr', 'val'),
        ('ARG0', 'd'),
        ('ARG1', ('x', [('/', 'thing')])),
    ]


def test_rearrange_attributes_first_still_keeps_instance_first():
    t = Tree(('s', [
        ('ARG0', 'd'),
        ('/', 'see-01'),
        ('attr', 'val'),
    ]))
    layout.rearrange(t, attributes_first=True)
    assert t.node[1][0] == ('/', 'see-01')


def test_rearrange_is_in_place_and_returns_none():
    t = Tree(('s', [('ARG1', 'b'), ('/', 'see-01'), ('ARG0', 'a')]))
    original_id = id(t.node[1])
    result = layout.rearrange(t)
    assert result is None
    assert id(t.node[1]) == original_id
    assert t.node[1] == [('/', 'see-01'), ('ARG0', 'a'), ('ARG1', 'b')]


def test_rearrange_does_not_reorder_instance_branch_against_key():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG1', 'b'),
        ('ARG0', 'a'),
    ]))
    layout.rearrange(t, key=lambda role: {'ARG0': 1, 'ARG1': 0}.get(role, 99))
    assert t.node[1][0] == ('/', 'see-01')
    assert t.node[1][1:] == [('ARG1', 'b'), ('ARG0', 'a')]


def test_rearrange_handles_deeply_nested_structure():
    t = Tree(('a', [
        ('/', 'alpha'),
        ('ARG1', ('b', [
            ('/', 'beta'),
            ('ARG1', ('c', [
                ('/', 'gamma'),
                ('ARG0', 'x'),
            ])),
            ('ARG0', 'y'),
        ])),
        ('ARG0', 'z'),
    ]))
    layout.rearrange(t, key=Model().canonical_order)
    assert t.node[1] == [
        ('/', 'alpha'),
        ('ARG0', 'z'),
        ('ARG1', ('b', [
            ('/', 'beta'),
            ('ARG0', 'y'),
            ('ARG1', ('c', [
                ('/', 'gamma'),
                ('ARG0', 'x'),
            ])),
        ])),
    ]


def test_rearrange_with_equal_keys_preserves_sorted_order_stability():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG2', 'c'),
        ('ARG1', 'b'),
        ('ARG0', 'a'),
    ]))
    layout.rearrange(t, key=lambda role: 0)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('ARG2', 'c'),
        ('ARG1', 'b'),
        ('ARG0', 'a'),
    ]


def test_rearrange_on_single_node_tree_keeps_tree_unchanged():
    t = Tree(('x', [('/', 'thing')]))
    layout.rearrange(t, key=Model().canonical_order)
    assert t.node == ('x', [('/', 'thing')])


def test_rearrange_with_attributes_first_and_nested_variable_targets_orders_atomic_branches_before_edges():
    t = Tree(('s', [
        ('/', 'see-01'),
        ('ARG1', ('c', [('/', 'cat')])),
        ('attr', 'val'),
        ('ARG0', ('d', [('/', 'dog')])),
        ('ARG2', 'x'),
    ]))
    layout.rearrange(t, attributes_first=True, key=Model().canonical_order)
    assert t.node[1] == [
        ('/', 'see-01'),
        ('attr', 'val'),
        ('ARG2', 'x'),
        ('ARG0', ('d', [('/', 'dog')])),
        ('ARG1', ('c', [('/', 'cat')])),
    ]
