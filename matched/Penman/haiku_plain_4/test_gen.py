import pytest

from penman.tree import Tree
from penman.layout import rearrange
from penman.codec import PENMANCodec


def _roles(node):
    """Return the list of roles of the branches of *node*."""
    _, branches = node
    return [role for role, _ in branches]


def test_default_keeps_original_order_without_key():
    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    rearrange(t)
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0']


def test_sorts_edges_by_key():
    t = Tree(('s', [('/', 'see-01'),
                    (':mod', 'loudly'),
                    (':ARG1', 'c'),
                    (':ARG0', 'd')]))
    rearrange(t, key=lambda role: role)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1', ':mod']


def test_instance_branch_stays_first_despite_key():
    # the key reverses the order of the edge roles
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG0', 'd'),
                    (':ARG1', 'c')]))
    rearrange(t, key=lambda role: tuple(-ord(ch) for ch in role))
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0']


def test_key_is_not_called_for_instance_branch():
    seen = []

    def key(role):
        seen.append(role)
        return role

    t = Tree(('s', [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]))
    rearrange(t, key=key)
    assert '/' not in seen
    assert set(seen) == {':ARG0', ':ARG1'}


def test_key_receives_role_strings():
    seen = []

    def key(role):
        seen.append(role)
        return role

    t = Tree(('s', [('/', 'see-01'), (':mod', 'loudly')]))
    rearrange(t, key=key)
    assert all(isinstance(role, str) for role in seen)
    assert seen == [':mod']


def test_stable_for_equal_keys():
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG1', 'c'),
                    (':ARG0', 'd'),
                    (':mod', 'loudly')]))
    rearrange(t, key=lambda role: 0)
    assert _roles(t.node) == ['/', ':ARG1', ':ARG0', ':mod']


def test_attributes_first_false_keeps_attributes_interleaved():
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':polarity', '-'),
                    (':ARG1', ('c', [('/', 'cat')]))]))
    rearrange(t, attributes_first=False)
    assert _roles(t.node) == ['/', ':ARG0', ':polarity', ':ARG1']


def test_attributes_first_places_attributes_before_edges():
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':polarity', '-'),
                    (':ARG1', ('c', [('/', 'cat')]))]))
    rearrange(t, attributes_first=True)
    assert _roles(t.node) == ['/', ':polarity', ':ARG0', ':ARG1']


def test_attributes_first_sorts_within_groups():
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG1', ('c', [('/', 'cat')])),
                    (':time', '"now"'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':polarity', '-')]))
    rearrange(t, key=lambda role: role, attributes_first=True)
    assert _roles(t.node) == ['/', ':polarity', ':time', ':ARG0', ':ARG1']


def test_variable_defined_elsewhere_counts_as_edge_for_attributes_first():
    # ':ARG0 d' is an edge because 'd' is defined deeper in the tree
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG0', 'd'),
        (':mod', 'loud'),
        (':ARG1', ('c', [
            ('/', 'cat'),
            (':mod', ('d', [('/', 'dog')])),
        ])),
    ]))
    rearrange(t, attributes_first=True)
    assert _roles(t.node) == ['/', ':mod', ':ARG0', ':ARG1']


def test_nested_nodes_are_rearranged_recursively():
    t = Tree(('s', [
        ('/', 'see-01'),
        (':ARG1', ('c', [
            ('/', 'cat'),
            (':quant', '2'),
            (':mod', 'black'),
        ])),
    ]))
    rearrange(t, key=lambda role: role)
    _, s_branches = t.node
    nested = s_branches[1][1]
    assert _roles(nested) == ['/', ':mod', ':quant']


def test_deeply_nested_rearrangement():
    t = Tree(('a', [
        ('/', 'alpha'),
        (':ARG0', ('b', [
            ('/', 'beta'),
            (':mod', 'x'),
            (':ARG1', ('c', [
                ('/', 'gamma'),
                (':quant', '1'),
                (':ARG0', 'a'),
            ])),
        ])),
    ]))
    rearrange(t, key=lambda role: role)
    _, a_branches = t.node
    b_node = a_branches[1][1]
    c_node = b_node[1][2][1]
    assert _roles(t.node) == ['/', ':ARG0']
    assert _roles(b_node) == ['/', ':ARG1', ':mod']
    assert _roles(c_node) == ['/', ':ARG0', ':quant']


def test_empty_branches_is_noop():
    t = Tree(('x', []))
    rearrange(t)
    assert t.node == ('x', [])


def test_single_instance_branch_unchanged():
    t = Tree(('x', [('/', 'dog')]))
    rearrange(t, key=lambda role: role)
    assert t.node == ('x', [('/', 'dog')])


def test_returns_none():
    t = Tree(('s', [('/', 'see-01'), (':ARG0', 'd')]))
    assert rearrange(t) is None


def test_mutates_branch_list_in_place():
    branches = [('/', 'see-01'), (':ARG1', 'c'), (':ARG0', 'd')]
    t = Tree(('s', branches))
    rearrange(t, key=lambda role: role)
    assert t.node[1] is branches
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']


def test_rearrange_is_idempotent():
    t = Tree(('s', [('/', 'see-01'),
                    (':mod', 'loudly'),
                    (':ARG1', 'c'),
                    (':ARG0', 'd')]))
    rearrange(t, key=lambda role: role)
    first = _roles(t.node)
    rearrange(t, key=lambda role: role)
    assert _roles(t.node) == first


def test_rearrange_parsed_tree_from_codec():
    c = PENMANCodec()
    t = c.parse('(s / see-01 :ARG1 (c / cat) :ARG0 (d / dog))')
    rearrange(t, key=lambda role: role)
    assert _roles(t.node) == ['/', ':ARG0', ':ARG1']
