import pytest

from penman.codec import PENMANCodec
from penman.layout import rearrange
from penman.tree import Tree


def _roles(node):
    """Return the roles of the branches of a node, in order."""
    return [role for role, _ in node[1]]


def _descending(role):
    """A key that orders roles in reverse lexicographic order."""
    return [-ord(c) for c in role]


def test_returns_none_and_sorts_in_place():
    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    result = rearrange(t, key=str)
    assert result is None
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_no_key_preserves_original_order():
    branches = [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]
    t = Tree(('s', list(branches)))
    rearrange(t)
    assert t.node[1] == branches


def test_key_sorts_edge_roles():
    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    rearrange(t, key=str)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_key_receives_roles_not_branches():
    seen = []

    def key(role):
        seen.append(role)
        return role

    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    rearrange(t, key=key)
    # the leading instance branch is never passed to the key
    assert set(seen) == {':ARG0', ':ARG1'}


def test_instance_branch_first_is_not_passed_to_key():
    def key(role):
        assert role != '/'
        return role

    t = Tree(('s', [('/', 'see-01'), (':ARG0', 'd'), (':ARG1', 'c')]))
    rearrange(t, key=key)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_instance_branch_not_first_is_sorted_like_other_branches():
    t = Tree(('s', [(':ARG0', 'd'), ('/', 'see-01')]))
    rearrange(t, key=lambda role: role == '/')
    assert _roles(t.node) == [':ARG0', '/']


def test_descending_key_reverses_edge_order():
    t = Tree(('s', [('/', 'see-01'), (':ARG0', 'd'), (':ARG1', 'c')]))
    rearrange(t, key=_descending)
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0']


def test_attributes_first_places_attributes_before_edges():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [('/', 'cat')])),
        (':polarity', '-'),
        (':ARG0', 'c'),
    ]))
    rearrange(t, key=str, attributes_first=True)
    assert _roles(t.node) == ['/', ':polarity', ':ARG0', ':ARG1']


def test_attributes_not_moved_without_attributes_first():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [('/', 'cat')])),
        (':polarity', '-'),
    ]))
    rearrange(t, key=str)
    assert _roles(t.node) == ['/', ':ARG1', ':polarity']


def test_undefined_variable_target_is_treated_as_attribute():
    # 'x' is not a node defined in the tree, so it is an attribute value
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG0', 'x'),
        (':ARG1', ('c', [('/', 'cat')])),
    ]))
    rearrange(t, key=_descending, attributes_first=True)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_attributes_first_keeps_instance_branch_first():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [('/', 'cat')])),
        (':name', 'x'),
    ]))
    rearrange(t, key=_descending, attributes_first=True)
    assert _roles(t.node) == ['/', ':name', ':ARG1']


def test_nested_nodes_are_sorted_recursively():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [
            ('/', 'beta'),
            (':ARG2', 'z'),
            (':ARG1', 'y'),
        ])),
    ]))
    rearrange(t, key=str)
    child = t.node[1][1][1]
    assert child[0] == 'b'
    assert _roles(child) == ['/', ':ARG1', ':ARG2']


def test_sibling_nested_nodes_keep_their_content():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [('/', 'cat')])),
        (':ARG0', ('d', [('/', 'dog')])),
    ]))
    rearrange(t, key=str)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']
    assert t.node[1][1] == (':ARG0', ('d', [('/', 'dog')]))
    assert t.node[1][2] == (':ARG1', ('c', [('/', 'cat')]))


def test_empty_branches_unchanged():
    t = Tree(('x', []))
    rearrange(t, key=str)
    assert t.node == ('x', [])


def test_only_instance_branch_unchanged():
    t = Tree(('x', [('/', 'thing')]))
    rearrange(t, key=str)
    assert t.node == ('x', [('/', 'thing')])


def test_branch_list_object_identity_preserved():
    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    branches = t.node[1]
    rearrange(t, key=str)
    assert t.node[1] is branches


def test_constant_key_preserves_order_stably():
    t = Tree(('s', [(':ARG1', 'c'), (':ARG0', 'd'), (':mod', 'e')]))
    rearrange(t, key=lambda role: 0)
    assert _roles(t.node) == [':ARG1', ':ARG0', ':mod']


def test_codec_parsed_tree_is_sorted_and_formats_in_order():
    c = PENMANCodec()
    t = c.parse('(s / see-01 :ARG1 (c / cat) :ARG0 (d / dog))')
    rearrange(t, key=str)
    text = c.format(t)
    assert text.index('ARG0') < text.index('ARG1')
    assert 'see-01' in text and 'dog' in text and 'cat' in text
