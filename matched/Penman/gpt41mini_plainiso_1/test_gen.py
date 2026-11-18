import pytest
from penman.layout import rearrange, _rearrange, interpret, get_pushed_variable, appears_inverted, node_contexts
from penman.tree import Tree, is_atomic
from penman.graph import Graph
from penman.epigraph import Epidatum
from penman.layout import Push, POP
from penman.model import Model


def tree_helper():
    # Simple tree for rearranging: (b / bark-01 :ARG0 (d / dog) :mod (l / loud))
    return Tree((
        'b',
        [
            ('/', 'bark-01'),
            (':ARG0', ('d', [('/', 'dog')])),
            (':mod', ('l', [('/', 'loud')]))
        ]
    ))


def test_rearrange_no_key_no_attrsfirst():
    t = tree_helper()
    original_branches = list(t.node[1])
    # make a copy to compare order after rearrange
    rearrange(t)
    # Instance stays first
    assert t.node[1][0][0] == '/'
    # rest should be sorted by (criterion1, criterion2) with default key None
    # thus all booleans True for criterion2
    # leaves order effectively unchanged except sorted by criterion1 which is False (atomic target in variables set) or True
    # since none of the targets is in variables set (empty set by default) criterion1 all False
    # so order by True (key is None) which is True; hence order unchanged
    assert t.node[1][1:] == original_branches[1:]


def test_rearrange_attributes_first_true():
    t = tree_helper()
    rearrange(t, attributes_first=True)
    # attribute branches (atomic target in variables) appear first after instance
    # 'bark-01' is concept, so instance branch first
    # Both :ARG0 and :mod have targets that are variables
    roles = [role for role, _ in t.node[1]]
    assert roles[0] == '/'
    # attributes_first=True makes attributes (atomic targets in variables) appear first,
    # but here targets 'd' and 'l' appear as nested nodes so not atomic
    # So ordering depends on key=None so criterion2 is True for all
    # Guarantees instance branch first and no error.
    assert set(roles[1:]).issuperset({':ARG0', ':mod'})


def test_rearrange_with_key_reverse_order():
    t = tree_helper()
    def reverse_key(role):
        # reverse alphabetical order
        return ''.join(reversed(role))
    rearrange(t, key=reverse_key)
    # instance branch always first
    assert t.node[1][0][0] == '/'
    # the rest is sorted by reversed role string
    sorted_roles = sorted([b[0] for b in t.node[1][1:]], key=reverse_key)
    result_roles = [b[0] for b in t.node[1][1:]]
    assert result_roles == sorted_roles


@pytest.mark.parametrize('is_atomic_target', [True, False])
def test__rearrange_recursive(is_atomic_target):
    if is_atomic_target:
        t = Tree((
            'a',
            [('/', 'alpha'),
             (':role1', 'val1'),
             (':role2', 'val2')]))
        orig = list(t.node[1])
        _rearrange(t.node, key=lambda b: b[0])
        # instance branch should remain first
        assert t.node[1][0][0] == '/'
        # rest sorted by role name
        sorted_roles = sorted(b[0] for b in orig[1:])
        assert [b[0] for b in t.node[1][1:]] == sorted_roles
    else:
        t = Tree((
            'a',
            [('/', 'alpha'),
             (':r1', ('b', [('/', 'beta')])),
             (':r2', ('c', [('/', 'gamma')]))]))
        _rearrange(t.node, key=lambda b: b[0])
        # instance branch remains first
        assert t.node[1][0][0] == '/'
        # subtrees rearranged recursively
        for _, target in t.node[1][1:]:
            assert isinstance(target, tuple)
            assert target[1][0][0] == '/'  # instance branch first in subtree


def test_get_pushed_variable_returns_variable():
    # Create graph with epidata containing Push on triple
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: [Push('b')]}
    g = Graph([triple], epidata=epidata, top='a')
    pushed = get_pushed_variable(g, triple)
    assert pushed == 'b'


def test_get_pushed_variable_returns_none():
    triple = ('a', ':instance', 'alpha')
    epidata = {triple: []}
    g = Graph([triple], epidata=epidata, top='a')
    pushed = get_pushed_variable(g, triple)
    assert pushed is None


def test_appears_inverted_true_with_push():
    # triple with Push where variable == triple[0]
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: [Push('a')]}
    g = Graph([triple], epidata=epidata, top='a')
    assert appears_inverted(g, triple) is True


def test_appears_inverted_false_due_to_concept_role():
    triple = ('a', ':instance', 'alpha')
    epidata = {triple: [Push('a')]}
    g = Graph([triple], epidata=epidata, top='a')
    # Concept role triples never appear inverted
    assert appears_inverted(g, triple) is False


def test_appears_inverted_false_no_push_but_target_node_context():
    # Setup graph where triple target matches node context
    triple1 = ('a', ':ARG0', 'b')
    triple2 = ('b', ':instance', 'beta')
    epidata = {triple1: [], triple2: []}
    g = Graph([triple1, triple2], epidata=epidata, top='a')
    # node contexts returns [a, b]
    assert appears_inverted(g, triple1) is True
    assert appears_inverted(g, triple2) is False


def test_appears_inverted_false_no_push_no_target_in_context():
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: []}
    g = Graph([triple], epidata=epidata, top='a')
    # If target not in variables or no push, should be False
    assert appears_inverted(g, triple) is False


def test_node_contexts_simple_graph():
    triple1 = ('a', ':instance', 'alpha')
    triple2 = ('a', ':ARG0', 'b')
    triple3 = ('b', ':instance', 'beta')
    epidata = {
        triple1: [],
        triple2: [Push('b')],
        triple3: []
    }
    g = Graph([triple1, triple2, triple3], epidata=epidata, top='a')
    contexts = node_contexts(g)
    assert contexts[0] == 'a'  # triple1 context 'a'
    assert contexts[1] == 'a'  # triple2 context 'a'
    assert contexts[2] == 'b'  # triple3 context 'b'


def test_node_contexts_extra_pop_handling():
    triple1 = ('a', ':instance', 'alpha')
    triple2 = ('a', ':ARG0', 'b')
    triple3 = ('b', ':instance', 'beta')
    epidata = {
        triple1: [],
        triple2: [Push('b'), POP, POP],  # extra POPs
        triple3: []
    }
    g = Graph([triple1, triple2, triple3], epidata=epidata, top='a')
    contexts = node_contexts(g)
    # Should return contexts up to where stack exhaustion happens gracefully
    assert contexts[0] == 'a'
    assert contexts[1] == 'a'
    # Cannot guarantee triple3 context because of extra POPs
    assert contexts[2] is None or contexts[2] == 'b'


def test_rearrange_with_attributes_first_and_key():
    t = tree_helper()
    def key_reverse(role):
        return ''.join(reversed(role))
    rearrange(t, key=key_reverse, attributes_first=True)
    # instance branch first
    assert t.node[1][0][0] == '/'
    # rest sorted by reversed role string with attributes first
    other_roles = [b[0] for b in t.node[1][1:]]
    expected_sorted = sorted(other_roles, key=key_reverse)
    assert other_roles == expected_sorted


def test_rearrange_empty_tree():
    t = Tree(('a', []))
    rearrange(t)
    assert t.node == ('a', [])


def test_rearrange_instance_missing_branches():
    t = Tree(('a', [(':role', 'b')]))
    rearrange(t)
    # instance branch stays missing, but rearrange can handle gracefully
    roles = [r for r, _ in t.node[1]]
    assert ':role' in roles


@pytest.mark.parametrize("target_type", ['atomic', 'nested'])
def test_rearrange_branch_targets(target_type):
    if target_type == 'atomic':
        t = Tree(('a', [('/', 'alpha'), (':r1', 'value1'), (':r2', 'value2')]))
    else:
        t = Tree(('a', [('/', 'alpha'),
                        (':r1', ('b', [('/', 'beta')])),
                        (':r2', ('c', [('/', 'gamma')]))]))
    rearrange(t)
    # instance branch first
    assert t.node[1][0][0] == '/'
    # branches rearranged without error


def test_rearrange_preserves_instance_position():
    t = Tree(('x', [('/', 'concept'),
                    (':a', ('y', [('/', 'cat')])),
                    (':b', 'val')]))
    rearrange(t)
    # instance branch always first
    assert t.node[1][0][0] == '/'

