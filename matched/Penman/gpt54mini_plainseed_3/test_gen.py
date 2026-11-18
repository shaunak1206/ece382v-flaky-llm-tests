import copy

import pytest

from penman.tree import Tree
from penman import layout


def make_tree():
    return Tree(
        ('s', [
            ('/', 'see-01'),
            ('ARG1', ('c', [
                ('/', 'cat'),
                ('mod', 'small'),
            ])),
            ('ARG0', ('d', [
                ('/', 'dog'),
            ])),
            ('time', 'yesterday'),
        ])
    )


def roles(node):
    return [role for role, _ in node[1]]


def test_rearrange_keeps_instance_branch_first_even_with_key():
    t = make_tree()
    layout.rearrange(t, key=lambda r: 0)
    assert roles(t.node)[0] == '/'
    assert roles(t.node) == ['/', 'ARG1', 'ARG0', 'time']


def test_rearrange_sorts_top_level_branches_by_key():
    t = make_tree()
    layout.rearrange(t, key=lambda r: {'time': 0, 'ARG0': 1, 'ARG1': 2}.get(r, 99))
    assert roles(t.node) == ['/', 'time', 'ARG0', 'ARG1']


def test_rearrange_without_key_is_deterministic_and_preserves_instance_first():
    t = make_tree()
    layout.rearrange(t)
    assert roles(t.node) == ['/', 'ARG0', 'ARG1', 'time']


def test_rearrange_can_put_attributes_before_variable_edges():
    t = make_tree()
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', 'time', 'ARG0', 'ARG1']


def test_rearrange_attributes_first_only_affects_non_instance_branches():
    t = make_tree()
    layout.rearrange(t, attributes_first=True)
    arg0_node = t.node[1][2][1]
    assert roles(arg0_node) == ['/']
    arg1_node = t.node[1][3][1]
    assert roles(arg1_node) == ['/', 'mod']


def test_rearrange_recurses_into_nested_nodes():
    t = make_tree()
    layout.rearrange(t, key=lambda r: {'mod': 0}.get(r, 1))
    arg1_node = t.node[1][2][1]
    assert roles(arg1_node) == ['/', 'mod']


def test_rearrange_handles_atomic_non_variable_targets_as_attributes():
    t = Tree(('x', [
        ('/', 'example'),
        ('z', 'zzz'),
        ('a', 'aaa'),
        ('m', 'mmm'),
    ]))
    layout.rearrange(t, attributes_first=True)
    assert roles(t.node) == ['/', 'a', 'm', 'z']


def test_rearrange_handles_nested_variable_targets_as_edges():
    t = Tree(('x', [
        ('/', 'root'),
        ('rel2', ('b', [('/', 'bee')])),
        ('rel1', ('a', [('/', 'ayy')])),
    ]))
    layout.rearrange(t, key=lambda r: 0 if r == 'rel1' else 1)
    assert roles(t.node) == ['/', 'rel1', 'rel2']


def test_rearrange_on_tree_with_no_branches_is_noop():
    t = Tree(('x', []))
    layout.rearrange(t, attributes_first=True)
    assert t.node == ('x', [])


def test_rearrange_preserves_tree_metadata():
    t = Tree(('x', [('/', 'root'), ('b', 'b')]), metadata={'id': '1'})
    layout.rearrange(t)
    assert t.metadata == {'id': '1'}


def test_rearrange_modifies_tree_in_place():
    t = make_tree()
    original_id = id(t)
    layout.rearrange(t, key=lambda r: {'time': 0, 'ARG0': 1, 'ARG1': 2}.get(r, 99))
    assert id(t) == original_id
    assert roles(t.node) == ['/', 'time', 'ARG0', 'ARG1']


def test_rearrange_does_not_change_deepcopy_source():
    t = make_tree()
    t_copy = copy.deepcopy(t)
    layout.rearrange(t_copy, key=lambda r: {'time': 0, 'ARG0': 1, 'ARG1': 2}.get(r, 99))
    assert roles(t.node) == ['/', 'ARG1', 'ARG0', 'time']
    assert roles(t_copy.node) == ['/', 'time', 'ARG0', 'ARG1']


def test_rearrange_is_idempotent_when_called_twice():
    t = make_tree()
    layout.rearrange(t, key=lambda r: {'time': 0, 'ARG0': 1, 'ARG1': 2}.get(r, 99))
    first = repr(t)
    layout.rearrange(t, key=lambda r: {'time': 0, 'ARG0': 1, 'ARG1': 2}.get(r, 99))
    second = repr(t)
    assert first == second
