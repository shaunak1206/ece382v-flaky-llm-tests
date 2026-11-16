import pytest

from penman.tree import Tree
from penman.layout import rearrange
from penman.codec import PENMANCodec


def _roles(node):
    """Return the list of branch roles of a tree node."""
    return [role for role, _ in node[1]]


def _child(node, role):
    """Return the nested node reached by the first branch with *role*."""
    for r, target in node[1]:
        if r == role:
            return target
    raise KeyError(role)


def test_rearrange_returns_none():
    t = Tree(('s', [('/', 'see-01'), (':ARG0', 'd')]))
    assert rearrange(t) is None


def test_sorts_branches_by_key():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', 'd'),
    ]))
    rearrange(t, key=lambda role: role)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_no_key_keeps_original_order():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', 'd'),
    ]))
    rearrange(t)
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0']


def test_instance_branch_stays_first_even_if_key_would_move_it():
    # key puts '/' last if it were sorted with the other branches
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG0', 'd'),
        (':ARG1', 'c'),
    ]))
    rearrange(t, key=lambda role: role == '/')
    assert _roles(t.node)[0] == '/'


def test_instance_branch_is_excluded_from_key_calls():
    seen = []

    def key(role):
        seen.append(role)
        return role

    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', 'd'),
    ]))
    rearrange(t, key=key)
    assert '/' not in seen
    assert set(seen) == {':ARG0', ':ARG1'}


def test_key_receives_role_not_whole_branch():
    received = []

    def key(role):
        received.append(role)
        return role

    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', ('d', [('/', 'dog')])),
    ]))
    rearrange(t, key=key)
    assert all(isinstance(r, str) for r in received)


def test_reverse_order_by_key():
    priority = {':ARG0': 0, ':ARG1': 1, ':ARG2': 2}
    t = Tree(('s', [
        ('/', 'give-01'),
        (':ARG0', 'a'),
        (':ARG2', 'c'),
        (':ARG1', 'b'),
    ]))
    rearrange(t, key=lambda role: -priority[role])
    assert _roles(t.node) == ['/', ':ARG2', ':ARG1', ':ARG0']


def test_equal_keys_preserve_original_order():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', 'd'),
        (':mod', 'e'),
    ]))
    rearrange(t, key=lambda role: 0)
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0', ':mod']


def test_attributes_first_places_attributes_before_edges():
    # ':polarity' and ':mode' are attributes (atomic, non-variable values)
    # ':ARG1' is a re-entrancy to variable 'd', ':ARG0' is a nested node
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'd'),
        (':polarity', '-'),
        (':ARG0', ('d', [('/', 'dog')])),
        (':mode', 'interrogative'),
    ]))
    rearrange(t, key=lambda role: role, attributes_first=True)
    assert _roles(t.node) == ['/', ':mode', ':polarity', ':ARG0', ':ARG1']


def test_attributes_first_false_does_not_separate_attributes():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'd'),
        (':polarity', '-'),
        (':ARG0', ('d', [('/', 'dog')])),
        (':mode', 'interrogative'),
    ]))
    rearrange(t, key=lambda role: role, attributes_first=False)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1', ':mode', ':polarity']


def test_attributes_first_without_key_keeps_relative_order_within_groups():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', 'd'),
        (':polarity', '-'),
        (':ARG0', ('d', [('/', 'dog')])),
        (':mode', 'interrogative'),
    ]))
    rearrange(t, attributes_first=True)
    # attributes (False) first, edges (True) after, original order in each group
    assert _roles(t.node) == ['/', ':polarity', ':mode', ':ARG1', ':ARG0']


def test_nested_node_branches_are_rearranged():
    child = ('d', [
        ('/', 'dog'),
        (':quant', '2'),
        (':ARG1', 'c'),
        (':ARG0', 'e'),
    ])
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [('/', 'cat')])),
        (':ARG0', child),
    ]))
    rearrange(t, key=lambda role: role)
    assert _roles(_child(t.node, ':ARG0')) == ['/', ':ARG0', ':ARG1', ':quant']


def test_recursion_reaches_deeply_nested_nodes():
    grandchild = ('g', [
        ('/', 'gamma'),
        (':ARG1', 'x'),
        (':ARG0', 'y'),
    ])
    child = ('d', [
        ('/', 'dog'),
        (':ARG1', ('g', grandchild[1])),
    ])
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG0', child),
    ]))
    rearrange(t, key=lambda role: role)
    deep = _child(_child(t.node, ':ARG0'), ':ARG1')
    assert _roles(deep) == ['/', ':ARG0', ':ARG1']


def test_instance_remains_first_in_nested_nodes():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [
            ('/', 'cat'),
            (':mod', 'x'),
            (':ARG0', 'y'),
        ])),
    ]))
    rearrange(t, key=lambda role: role == '/')
    child = _child(t.node, ':ARG1')
    assert _roles(child)[0] == '/'


def test_empty_branches_are_handled():
    t = Tree(('s', []))
    rearrange(t, key=lambda role: role)
    assert t.node == ('s', [])


def test_only_instance_branch_is_unchanged():
    t = Tree(('x', [('/', 'thing')]))
    rearrange(t, key=lambda role: role)
    assert t.node == ('x', [('/', 'thing')])


def test_rearrange_modifies_branch_list_in_place():
    branches = [
        ('/', 'see-01'),
        (':ARG1', 'c'),
        (':ARG0', 'd'),
    ]
    node = ('s', branches)
    t = Tree(node)
    rearrange(t, key=lambda role: role)
    assert t.node[1] is branches
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_parsed_tree_integration():
    c = PENMANCodec()
    t = c.parse('(s / see-01 :ARG1 (c / cat) :ARG0 (d / dog))')
    rearrange(t, key=lambda role: role)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']
    assert _roles(_child(t.node, ':ARG0')) == ['/']
