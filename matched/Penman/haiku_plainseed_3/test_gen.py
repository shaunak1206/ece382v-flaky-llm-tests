import pytest

from penman.tree import Tree
from penman.layout import rearrange
from penman.codec import PENMANCodec


def _roles(branches):
    return [role for role, _ in branches]


def test_rearrange_returns_none():
    t = Tree(('b', [('/', 'bark-01'), (':ARG0', 'd')]))
    assert rearrange(t) is None


def test_no_key_preserves_order():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG1', 'x'),
                    (':ARG0', 'y'),
                    (':mod', 'z')]))
    rearrange(t)
    assert _roles(t.node[1]) == ['/', ':ARG1', ':ARG0', ':mod']


def test_key_sorts_edges_by_role():
    t = Tree(('s', [('/', 'see-01'),
                    (':ARG1', ('c', [('/', 'cat')])),
                    (':ARG0', ('d', [('/', 'dog')]))]))
    rearrange(t, key=lambda r: r)
    assert _roles(t.node[1]) == ['/', ':ARG0', ':ARG1']


def test_instance_branch_stays_first_with_descending_key():
    # descending by last character: ':ARG1' before ':ARG0'
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG0', 'x'),
                    (':ARG1', 'y')]))
    rearrange(t, key=lambda r: -ord(r[-1]))
    assert _roles(t.node[1]) == ['/', ':ARG1', ':ARG0']


def test_key_called_only_on_non_instance_roles():
    calls = []

    def key(role):
        calls.append(role)
        return role

    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG1', 'x'),
                    (':ARG0', 'y')]))
    rearrange(t, key=key)
    assert calls
    assert '/' not in calls
    assert set(calls) == {':ARG0', ':ARG1'}


def test_sort_is_stable_for_equal_keys():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG2', 'x'),
                    (':ARG0', 'y'),
                    (':ARG1', 'z')]))
    rearrange(t, key=lambda r: 0)
    assert _roles(t.node[1]) == ['/', ':ARG2', ':ARG0', ':ARG1']


def test_empty_branches():
    t = Tree(('x', []))
    rearrange(t)
    assert t.node[1] == []


def test_only_instance_branch():
    t = Tree(('x', [('/', 'thing')]))
    rearrange(t, key=lambda r: r)
    assert _roles(t.node[1]) == ['/']


def test_no_instance_branch_sorts_all_branches():
    t = Tree(('x', [(':ARG1', 'a'), (':ARG0', 'b')]))
    rearrange(t, key=lambda r: r)
    assert _roles(t.node[1]) == [':ARG0', ':ARG1']


def test_nested_nodes_are_sorted_recursively():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG0', ('d', [('/', 'dog'),
                                     (':mod', 'big'),
                                     (':ARG1', 'x')])),
                    (':ARG1', 'y')]))
    rearrange(t, key=lambda r: r)
    assert _roles(t.node[1]) == ['/', ':ARG0', ':ARG1']
    d_node = t.node[1][1][1]
    assert _roles(d_node[1]) == ['/', ':ARG1', ':mod']


def test_deeply_nested_nodes_are_sorted():
    t = Tree(('a', [('/', 'x'),
                    (':ARG0', ('b', [('/', 'y'),
                                     (':ARG2', ('c', [('/', 'z'),
                                                      (':ARG1', 'q'),
                                                      (':ARG0', 'r')]))]))]))
    rearrange(t, key=lambda r: r)
    b_node = t.node[1][1][1]
    c_node = b_node[1][1][1]
    assert _roles(c_node[1]) == ['/', ':ARG0', ':ARG1']


def test_nested_instance_branch_stays_first():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG0', ('d', [('/', 'dog'),
                                     (':ARG1', 'x'),
                                     (':ARG0', 'y')]))]))
    rearrange(t, key=lambda r: -ord(r[-1]))
    d_node = t.node[1][1][1]
    assert _roles(d_node[1]) == ['/', ':ARG1', ':ARG0']


def test_attributes_first_groups_atomic_before_variables():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':ARG1', 'd'),
                    (':time', 'now')]))
    rearrange(t, attributes_first=True)
    assert _roles(t.node[1]) == ['/', ':time', ':ARG0', ':ARG1']


def test_attributes_first_false_ignores_variable_targets():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':ARG1', 'd'),
                    (':time', 'now')]))
    rearrange(t, attributes_first=False)
    assert _roles(t.node[1]) == ['/', ':ARG0', ':ARG1', ':time']


def test_attributes_first_with_key_sorts_within_groups():
    t = Tree(('b', [('/', 'bark-01'),
                    (':ARG1', 'd'),
                    (':mod', 'loud'),
                    (':ARG0', ('d', [('/', 'dog')])),
                    (':ARG2', 'x')]))
    rearrange(t, key=lambda r: r, attributes_first=True)
    assert _roles(t.node[1]) == ['/', ':ARG2', ':mod', ':ARG0', ':ARG1']


def test_attributes_first_detects_variables_nested_deeper():
    # 'c' is only defined inside the ARG1 subtree but is referenced
    # as an atomic target at the top level; it must count as a variable
    t = Tree(('a', [('/', 'alpha'),
                    (':ARG0', 'c'),
                    (':mod', 'x'),
                    (':ARG1', ('b', [('/', 'beta'),
                                     (':ARG0', ('c', [('/', 'gamma')]))]))]))
    rearrange(t, attributes_first=True)
    assert _roles(t.node[1]) == ['/', ':mod', ':ARG0', ':ARG1']


def test_branches_list_mutated_in_place():
    t = Tree(('x', [(':ARG1', 'a'), (':ARG0', 'b')]))
    before = t.node[1]
    rearrange(t, key=lambda r: r)
    assert t.node[1] is before
    assert before[0][0] == ':ARG0'
    assert before[1][0] == ':ARG1'


def test_codec_parse_rearrange_format():
    c = PENMANCodec()
    t = c.parse('(s / see-01 :ARG1 (c / cat) :ARG0 (d / dog))')
    rearrange(t, key=lambda r: r)
    s = c.format(t)
    assert s.startswith('(s / see-01')
    assert s.index(':ARG0') < s.index(':ARG1')
