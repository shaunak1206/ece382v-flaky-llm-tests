import pytest

from penman.layout import rearrange
from penman.tree import Tree


def make_tree(node):
    return Tree(node)


def branches(node):
    return node[1]


def test_rearrange_keeps_instance_branch_first():
    t = make_tree((
        'x',
        [
            (':ARG1', 'b'),
            ('/', 'alpha'),
            (':ARG0', 'a'),
        ],
    ))

    rearrange(t)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':ARG1', 'b'),
        (':ARG0', 'a'),
    ]


def test_rearrange_sorts_by_custom_key_on_roles():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG2', 'b'),
            (':ARG0', 'a'),
            (':mod', 'c'),
        ],
    ))

    rearrange(t, key=lambda role: role)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':ARG0', 'a'),
        (':ARG2', 'b'),
        (':mod', 'c'),
    ]


def test_rearrange_preserves_original_order_without_key():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':mod', 'c'),
            (':ARG2', 'b'),
            (':ARG0', 'a'),
        ],
    ))

    rearrange(t)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':mod', 'c'),
        (':ARG2', 'b'),
        (':ARG0', 'a'),
    ]


def test_rearrange_attributes_first_places_atomic_attributes_before_edges():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG0', ('b', [(' /'.strip(), 'beta')])),
            (':name', 'val'),
            (':mod', 'z'),
        ],
    ))

    rearrange(t, attributes_first=True)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':name', 'val'),
        (':mod', 'z'),
        (':ARG0', ('b', [(' /'.strip(), 'beta')])),
    ]


def test_rearrange_attributes_first_uses_key_within_each_group():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':mod', 'z'),
            (':ARG0', ('b', [(' /'.strip(), 'beta')])),
            (':name', 'val'),
            (':ARG2', 'w'),
        ],
    ))

    rearrange(t, key=lambda role: role, attributes_first=True)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':mod', 'z'),
        (':name', 'val'),
        (':ARG0', ('b', [(' /'.strip(), 'beta')])),
        (':ARG2', 'w'),
    ]


def test_rearrange_recursively_sorts_nested_nodes():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG0', (
                'b',
                [
                    ('/', 'beta'),
                    (':mod', 'z'),
                    (':ARG1', 'a'),
                ],
            )),
            (':ARG1', 'c'),
        ],
    ))

    rearrange(t, key=lambda role: role)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta'), (':ARG1', 'a'), (':mod', 'z')])),
        (':ARG1', 'c'),
    ]


def test_rearrange_does_not_reorder_atomic_targets_inside_nested_nodes_unless_needed():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG0', (
                'b',
                [
                    ('/', 'beta'),
                    (':z', '3'),
                    (':a', '1'),
                ],
            )),
        ],
    ))

    rearrange(t, key=lambda role: role)

    assert branches(t.node[1][1]) == [
        ('/', 'beta'),
        (':a', '1'),
        (':z', '3'),
    ]


def test_rearrange_handles_tree_without_instance_branch():
    t = make_tree((
        'x',
        [
            (':ARG1', 'b'),
            (':ARG0', 'a'),
        ],
    ))

    rearrange(t, key=lambda role: role)

    assert branches(t.node) == [
        (':ARG0', 'a'),
        (':ARG1', 'b'),
    ]


def test_rearrange_handles_empty_branch_list():
    t = make_tree(('x', []))

    rearrange(t)

    assert branches(t.node) == []


def test_rearrange_is_stable_for_equal_keys():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':b', '2'),
            (':a', '1'),
            (':c', '3'),
        ],
    ))

    rearrange(t, key=lambda role: 0)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':b', '2'),
        (':a', '1'),
        (':c', '3'),
    ]


def test_rearrange_keeps_atomic_attributes_before_nested_edges_when_attributes_first():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG0', ('b', [(' /'.strip(), 'beta')])),
            (':name', 'val'),
            (':ARG1', ('c', [(' /'.strip(), 'gamma')])),
        ],
    ))

    rearrange(t, attributes_first=True)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':name', 'val'),
        (':ARG0', ('b', [('/', 'beta')])),
        (':ARG1', ('c', [('/', 'gamma')])),
    ]


def test_rearrange_nested_nodes_are_sorted_even_if_parent_branch_order_is_preserved():
    t = make_tree((
        'x',
        [
            ('/', 'alpha'),
            (':ARG0', (
                'b',
                [
                    ('/', 'beta'),
                    (':c', '3'),
                    (':a', '1'),
                ],
            )),
            (':ARG1', (
                'd',
                [
                    ('/', 'delta'),
                    (':b', '2'),
                    (':a', '1'),
                ],
            )),
        ],
    ))

    rearrange(t, key=lambda role: role)

    assert branches(t.node) == [
        ('/', 'alpha'),
        (':ARG0', ('b', [('/', 'beta'), (':a', '1'), (':c', '3')])),
        (':ARG1', ('d', [('/', 'delta'), (':a', '1'), (':b', '2')])),
    ]
