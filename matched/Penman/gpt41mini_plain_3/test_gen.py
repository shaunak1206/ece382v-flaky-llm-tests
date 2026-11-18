import pytest

from penman.layout import rearrange, _rearrange, interpret, get_pushed_variable, appears_inverted, node_contexts
from penman.layout import Push, POP
from penman.tree import Tree, is_atomic
from penman.graph import Graph
from penman.model import Model
from penman.exceptions import LayoutError


def make_simple_tree():
    """Simple tree: (b / bark-01 :ARG0 (d / dog))"""
    return Tree(
        ('b', [
            ('/', 'bark-01'),
            (':ARG0', ('d', [
                ('/', 'dog'),
            ])),
        ])
    )


def test_rearrange_simple_no_key_changes_order():
    t = make_simple_tree()
    original_order = [r[0] for r in t.node[1]]
    rearrange(t)
    # instance role is always first and no key -> order unchanged or stable
    reorder = [r[0] for r in t.node[1]]
    # Because no key, only attributes first behavior applies
    assert reorder[0] == '/'  # instance branch always first
    assert set(original_order) == set(reorder)


def test_rearrange_attributes_first_true_moves_attributes_first():
    t = make_simple_tree()
    # Add an attribute branch (atomic target)
    t.node[1].append((':mod', 'quickly'))
    rearrange(t, attributes_first=True)
    roles = [b[0] for b in t.node[1]]
    # Instance first, then attributes (:mod), then edges (:ARG0)
    assert roles[0] == '/'
    # :mod (attribute) comes before :ARG0 (edge)
    idx_mod = roles.index(':mod')
    idx_arg0 = roles.index(':ARG0')
    assert idx_mod < idx_arg0


def test_rearrange_with_key_sorts_edges():
    t = make_simple_tree()
    # Add another edge branch
    t.node[1].append((':ARG1', ('c', [('/', 'cat')])))
    def reverse_role_key(role):
        # Sort reversed alphabetically (ignores instance role)
        return role[::-1]
    rearrange(t, key=reverse_role_key)
    roles = [b[0] for b in t.node[1] if b[0] != '/']
    sorted_roles = sorted(roles, key=reverse_role_key)
    assert roles == sorted_roles


def test__rearrange_recurses_into_nested_nodes():
    # Tree with depth 2, 1 nested node to check recursion
    t = make_simple_tree()
    # Add attribute branch to nested node 'd'
    nested_node = t.node[1][1][1]
    nested_node[1].append((':mod', 'loyally'))
    calls = []
    def key(branch):
        calls.append(branch)
        # simple no change key returning role
        return branch[0]
    _rearrange(t.node, key=key)
    # Key should be called on branches at root and nested node
    assert any(k[0] == ':mod' for k in calls)


def test_get_pushed_variable_returns_variable_when_exists():
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: [Push('b')]}
    g = Graph([triple], epidata=epidata, top='a')
    var = get_pushed_variable(g, triple)
    assert var == 'b'


def test_get_pushed_variable_returns_none_when_no_push():
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: []}
    g = Graph([triple], epidata=epidata, top='a')
    assert get_pushed_variable(g, triple) is None


def test_appears_inverted_returns_false_for_attributes_and_instance():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':attr', 'value')]
    epidata = {triples[0]: [], triples[1]: []}
    g = Graph(triples, epidata=epidata, top='a')
    assert not appears_inverted(g, triples[0])
    assert not appears_inverted(g, triples[1])


def test_appears_inverted_true_when_source_pushed():
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: [Push('a')]}
    g = Graph([triple], epidata=epidata, top='a')
    assert appears_inverted(g, triple)


def test_appears_inverted_true_when_target_is_current_context():
    triples = [
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('b', ':ARG0', 'c'),
        ('c', ':instance', 'charlie'),
        ('c', ':ARG0-of', 'a')]
    epidata = {triple: [] for triple in triples}
    g = Graph(triples, epidata=epidata, top='a')
    inverted = appears_inverted(g, triples[-1])
    assert inverted


def test_appears_inverted_false_when_no_push_and_no_context_match():
    triple = ('a', ':ARG0', 'b')
    epidata = {triple: []}
    g = Graph([triple], epidata=epidata, top='a')
    assert not appears_inverted(g, triple)


def test_node_contexts_basic():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':attr', 'val'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('b', ':ARG0', 'g'),
        ('g', ':instance', 'gamma'),
        ('g', ':ARG0-of', 'a')]
    epidata = {triple: [] for triple in triples}
    g = Graph(triples, epidata=epidata, top='a')

    contexts = node_contexts(g)
    # First triple context must be top 'a'
    assert contexts[0] == 'a'
    # After pushing 'b', context switches on triples to 'b'
    assert contexts[3] == 'b'
    # After pushing 'g', context switches on triples to 'g'
    assert contexts[5] == 'g'


def test_node_contexts_with_push_and_pop():
    triples = [
        ('t', ':instance', 'try-01'),
        ('t', ':ARG0', 'd'),
        ('d', ':instance', 'dog'),
        ('t', ':ARG1', 'b'),
        ('b', ':instance', 'bark-01'),
        ('b', ':ARG0', 'd'),
    ]
    epidata = {
        triples[1]: [Push('d')],
        triples[2]: [POP],
        triples[3]: [Push('b')],
        triples[4]: [],
        triples[5]: [POP],
        triples[0]: [],
    }
    g = Graph(triples, epidata=epidata, top='t')
    contexts = node_contexts(g)
    expected_contexts = ['t', 't', 'd', 't', 'b', 'b']
    assert contexts == expected_contexts


def test_rearrange_maintains_instance_first_even_with_attributes_first():
    t = make_simple_tree()
    t.node[1].append((':mod', 'quickly'))
    rearrange(t, attributes_first=True)
    roles = [b[0] for b in t.node[1]]
    assert roles[0] == '/'


def test_rearrange_atomic_target_in_variables_attributes_first():
    t = make_simple_tree()
    vars_ = {node[0] for node in t.nodes()}
    # Add a target that is a variable for attribute first effect
    t.node[1].append((':mod', 'b'))  # 'b' is in vars_
    rearrange(t, attributes_first=True)
    roles = [b[0] for b in t.node[1]]
    assert roles[0] == '/'


def test_rearrange_handles_empty_tree():
    t = Tree(('a', []))
    rearrange(t)
    assert t.node[0] == 'a'
    assert t.node[1] == []


def test_rearrange_string_atomic_target_with_tilde():
    from penman.surface import Alignment
    t = Tree(('a', [
        (':attr', 'val~0:0'),
        (':mod', ('b', [('/', 'dog')])),
    ]))
    # forcibly add atomic target with alignment marker
    rearrange(t)
    # Should not raise and still keep alignment (indirect test)
    targets = [b[1] for b in t.node[1]]
    assert any(isinstance(target, str) for target in targets)


def test_interpret_returns_graph_with_expected_triples():
    t = make_simple_tree()
    g = interpret(t)
    # The triples should represent the tree
    triples_set = set(g.triples)
    expected = {
        ('b', ':instance', 'bark-01'),
        ('b', ':ARG0', 'd'),
        ('d', ':instance', 'dog'),
    }
    assert expected.issubset(triples_set)


def test_interpret_inverted_triple_handling_warns(monkeypatch):
    mn = Model()
    # Force is_role_inverted to True for role to test inversion path
    monkeypatch.setattr(mn, 'is_role_inverted', lambda role: True)
    monkeypatch.setattr(mn, 'invert', lambda triple: (triple[2], triple[1], triple[0]))
    t = Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', 'x')  # atomic target but x not in vars
        ])
    )
    with pytest.warns(UserWarning):
        g = interpret(t, model=mn)
    # triples should include the inverted triple for ':ARG0' but warn that cannot deinvert attribute


def test_rearrange_key_none_with_attributes_first_false_leaves_order():
    t = make_simple_tree()
    # add one attribute branch to nested node
    nested_node = t.node[1][1][1]
    nested_node[1].append((':mod', 'loyally'))
    # capture before order
    before = [
        (b[0], b[1] if is_atomic(b[1]) else b[1][0]) for b in t.node[1]
    ]
    rearrange(t, key=None, attributes_first=False)
    after = [
        (b[0], b[1] if is_atomic(b[1]) else b[1][0]) for b in t.node[1]
    ]
    # The order of root branches should be unchanged since key is None and attributes_first is False
    assert before == after

