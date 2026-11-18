import pytest

from penman import layout
from penman.tree import Tree


def make_tree(node, metadata=None):
    return Tree(node, metadata=metadata or {})


def branch_roles(node):
    return [role for role, _ in node[1]]


def test_rearrange_sorts_non_instance_branches_by_key():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':ARG0', ('d', [('/', 'dog')])),
        ])
    )

    layout.rearrange(t)

    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_rearrange_keeps_instance_branch_first():
    t = make_tree(
        ('s', [
            (':ARG1', ('c', [('/', 'cat')])),
            ('/', 'see-01'),
            (':ARG0', ('d', [('/', 'dog')])),
        ])
    )

    layout.rearrange(t)

    assert branch_roles(t.node)[0] == '/'
    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_rearrange_uses_custom_key_for_branch_order():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG2', ('x', [('/', 'x')])) ,
            (':ARG0', ('a', [('/', 'a')])),
            (':ARG1', ('b', [('/', 'b')])),
        ])
    )

    layout.rearrange(t, key=lambda role: role)

    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1', ':ARG2']


def test_rearrange_attributes_first_places_atomic_branches_before_node_branches():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':manner', 'quickly'),
            (':ARG0', ('d', [('/', 'dog')])),
            (':time', 'yesterday'),
        ])
    )

    layout.rearrange(t, attributes_first=True)

    assert branch_roles(t.node) == ['/', ':manner', ':time', ':ARG0', ':ARG1']


def test_rearrange_attributes_first_still_keeps_instance_branch_first():
    t = make_tree(
        ('s', [
            (':ARG0', ('d', [('/', 'dog')])),
            ('/', 'see-01'),
            (':mod', 'fast'),
        ])
    )

    layout.rearrange(t, attributes_first=True)

    assert branch_roles(t.node) == ['/', ':mod', ':ARG0']


def test_rearrange_recursively_sorts_nested_nodes():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [
                ('/', 'cat'),
                (':mod', 'black'),
                (':ARG0', ('x', [('/', 'x')])),
            ])),
            (':ARG0', ('d', [
                ('/', 'dog'),
                (':mod', 'big'),
                (':ARG1', ('y', [('/', 'y')])),
            ])),
        ])
    )

    layout.rearrange(t, attributes_first=True)

    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1']
    assert branch_roles(t.node[1][1][1]) == ['/', ':mod', ':ARG1']
    assert branch_roles(t.node[1][2][1]) == ['/', ':mod', ':ARG1']


def test_rearrange_nested_node_order_respects_key_with_attributes_first():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG2', ('x', [('/', 'x')])),
            (':ARG0', ('a', [('/', 'a')])),
            (':ARG1', ('b', [('/', 'b')])),
        ])
    )

    layout.rearrange(t, key=lambda role: {'/': -1, ':ARG0': 0, ':ARG1': 1, ':ARG2': 2}[role])

    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1', ':ARG2']


def test_rearrange_is_in_place_and_returns_none():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':ARG0', ('d', [('/', 'dog')])),
        ])
    )
    original_id = id(t.node[1])

    result = layout.rearrange(t)

    assert result is None
    assert id(t.node[1]) == original_id
    assert branch_roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_rearrange_on_empty_branches_is_noop():
    t = make_tree(('x', []))

    layout.rearrange(t)

    assert t.node == ('x', [])


def test_rearrange_keeps_string_attributes_before_node_branches_when_requested():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':name', 'Alice'),
            (':ARG0', ('d', [('/', 'dog')])),
            (':polarity', '-'),
        ])
    )

    layout.rearrange(t, attributes_first=True)

    assert branch_roles(t.node) == ['/', ':name', ':polarity', ':ARG0', ':ARG1']


def test_rearrange_preserves_branch_order_among_equal_keys():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':ARG0', ('d', [('/', 'dog')])),
            (':ARG2', ('e', [('/', 'elephant')])),
        ])
    )

    layout.rearrange(t, key=lambda role: 0)

    assert branch_roles(t.node) == ['/', ':ARG1', ':ARG0', ':ARG2']


def test_rearrange_does_not_change_existing_order_when_already_sorted():
    t = make_tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG0', ('d', [('/', 'dog')])),
            (':ARG1', ('c', [('/', 'cat')])),
        ])
    )

    before = repr(t)
    layout.rearrange(t)
    after = repr(t)

    assert before == after
