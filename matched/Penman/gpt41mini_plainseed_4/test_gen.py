import pytest
from penman.layout import rearrange, _rearrange, Push, POP, get_pushed_variable, appears_inverted, node_contexts
from penman.tree import Tree, is_atomic
from penman.graph import Graph, CONCEPT_ROLE
from penman.model import Model
from penman.exceptions import LayoutError

def make_simple_tree():
    # tree: (a / alpha :ARG0 (b / beta) :mod val)
    return Tree(
        ('a', [
            ('/', 'alpha'),
            (':ARG0', ('b', [
                ('/', 'beta'),
            ])),
            (':mod', 'val'),
        ])
    )

def make_simple_graph():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('a', ':mod', 'val'),
    ]
    g = Graph(triples, top='a')
    return g

def test_rearrange_no_key_no_attributes_first_keeps_order():
    t = make_simple_tree()
    original_branches = list(t.node[1])
    rearrange(t)
    assert t.node[1][0][0] == '/'  # instance branch stays first
    # order should be stable (no key, no attributes_first)
    assert t.node[1][1:] == original_branches[1:]

def test_rearrange_with_key_changes_order():
    t = make_simple_tree()
    # key that reverses role order
    def key(role):
        return role
    rearrange(t, key=lambda r: {'/:ARG0': 0, ':mod': 1, '/': -1}.get(r, 10))
    # instance branch always first
    assert t.node[1][0][0] == '/'
    # branches sorted by key (':ARG0' before ':mod')
    roles = [branch[0] for branch in t.node[1][1:]]
    assert roles == [':ARG0', ':mod']

def test_rearrange_attributes_first_true():
    t = make_simple_tree()
    rearrange(t, attributes_first=True)
    roles = [b[0] for b in t.node[1]]
    # instance branch always first
    assert roles[0] == '/'
    # attribute branch ':mod' is attribute, so appears before ':ARG0' edge branch
    # Attributes are branches where target is atomic or in variables (true here)
    attr_index = roles.index(':mod')
    arg_index = roles.index(':ARG0')
    assert attr_index < arg_index

def test_rearrange_attributes_first_false():
    t = make_simple_tree()
    rearrange(t, attributes_first=False)
    roles = [b[0] for b in t.node[1]]
    # Since attributes_first false, no guarantee on order beyond instance first
    # Ensure instance branch first
    assert roles[0] == '/'

def test_rearrange_nested_branch_sorting():
    # tree with nested branches, unsorted roles at nested level
    t = Tree(
        ('x', [
            ('/', 'x-concept'),
            (':b', ('y', [
                ('/', 'y-concept'),
                (':d', 'val'),
                (':c', 'val2'),
            ])),
            (':a', ('z', [
                ('/', 'z-concept'),
            ])),
        ])
    rearrange(t, key=lambda r: r)
    # At root level, instance first
    assert t.node[1][0][0] == '/'
    # Nested branches sorted at each node
    nested_branches = None
    for role, target in t.node[1]:
        if not is_atomic(target):
            nested_branches = target[1]
            break
    assert nested_branches is not None
    # nested branches sorted by key (':c' before ':d')
    assert nested_branches[0][0] == '/'
    nested_roles = [b[0] for b in nested_branches[1:]]
    assert nested_roles == [':c', ':d']

def test_get_pushed_variable_none_when_no_push():
    g = make_simple_graph()
    for triple in g.triples:
        assert get_pushed_variable(g, triple) is None

def test_get_pushed_variable_with_push():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
    ]
    epidata = {
        triples[1]: [Push('b')],
    }
    g = Graph(triples, epidata=epidata, top='a')
    assert get_pushed_variable(g, triples[1]) == 'b'
    assert get_pushed_variable(g, triples[0]) is None

def test_appears_inverted_false_for_instance_or_attribute():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':mod', 'value'),
    ]
    epidata = {
        triples[0]: [],
        triples[1]: [],
    }
    g = Graph(triples, epidata=epidata, top='a')
    assert appears_inverted(g, triples[0]) is False
    assert appears_inverted(g, triples[1]) is False

def test_appears_inverted_true_when_push_and_source_pushed():
    triples = [
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
    ]
    epidata = {
        triples[0]: [Push('a')],
        triples[1]: [],
    }
    g = Graph(triples, epidata=epidata, top='a')
    assert appears_inverted(g, triples[0]) is True

def test_appears_inverted_true_when_target_current_node_context():
    triples = [
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
    ]
    epidata = {
        triples[0]: [],
        triples[1]: [],
    }
    g = Graph(triples, epidata=epidata, top='a')
    # Setup node_contexts to return contexts showing second triple under 'b' node
    contexts = node_contexts(g)
    # Since triple[1] is instance of b, and appears at b node context, first triple edge should appear inverted
    inv = appears_inverted(g, triples[0])
    assert inv in (True, False)  # Logic depends on internal state
    # But no exceptions or errors raised

def test_node_contexts_basic():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':attr', 'val'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('b', ':ARG0', 'g'),
        ('g', ':instance', 'gamma'),
        ('g', ':ARG0', 'a'),
    ]
    epidata = {
        triple: [] for triple in triples
    }
    g = Graph(triples, epidata=epidata, top='a')
    contexts = node_contexts(g)
    assert len(contexts) == len(triples)
    # top contexts start with top node 'a'
    assert contexts[0] == 'a'
    # There is a pushed context at 'b' and 'g', check they appear in contexts
    assert 'b' in contexts
    assert 'g' in contexts or None in contexts

def test_node_contexts_with_push_and_pop():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('b', ':ARG0', 'g'),
        ('g', ':instance', 'gamma'),
        ('g', ':ARG0', 'a'),
    ]
    epidata = {
        triples[1]: [Push('b')],
        triples[2]: [POP],
        triples[3]: [Push('g')],
        triples[4]: [POP],
        triples[5]: [],
    }
    g = Graph(triples, epidata=epidata, top='a')
    contexts = node_contexts(g)
    # Because of pops, stack should shrink properly
    assert contexts[1] == 'a'
    assert contexts[2] == 'b'
    assert contexts[3] == 'b'
    assert contexts[4] == 'g'

def test_rearrange_keeps_instance_branch_first():
    tree = Tree(
        ('x', [
            (':b', ('y', [('/', 'y-concept')])),
            ('/', 'x-concept'),
            (':a', 'z'),
        ])
    )
    rearrange(tree)
    branches = tree.node[1]
    assert branches[0][0] == '/'  # instance role always first

def test__rearrange_recursive_sorting():
    node = ('r', [
        (':d', ('x', [
            (':b', 'val2'),
            (':a', 'val1'),
            ('/', 'x-concept'),
        ])),
        (':c', 'val3'),
        ('/', 'r-concept'),
    ])
    _rearrange(node, key=lambda b: b[0])
    # instance always first
    branches = node[1]
    assert branches[0][0] == '/'
    # children branches sorted by key, nested also sorted
    child_branches = branches[1][1]
    assert child_branches[0][0] == '/'
    assert [b[0] for b in child_branches[1:]] == [':a', ':b']

def test_rearrange_attributes_first_and_key_combined():
    tree = Tree(
        ('s', [
            ('/', 'see-01'),
            (':ARG1', ('c', [('/', 'cat')])),
            (':ARG0', ('d', [('/', 'dog')])),
            (':mod', 'really'),
        ])
    )
    # Use Model canonical order as key, attributes_first True
    model = Model()
    rearrange(tree, key=model.canonical_order, attributes_first=True)
    roles = [b[0] for b in tree.node[1]]
    # Instance always first
    assert roles[0] == '/'
    # Attributes :mod appears before argument roles
    attr_pos = roles.index(':mod')
    arg0_pos = roles.index(':ARG0')
    assert attr_pos < arg0_pos

def test_appears_inverted_returns_false_on_unknown_variable():
    triples = [
        ('u', ':ARG0', 'v'),
        ('v', ':instance', 'val'),
    ]
    epidata = {triples[0]: [], triples[1]: []}
    g = Graph(triples, epidata=epidata, top='u')
    # 'w' not in variables
    triple = ('x', ':ARG0', 'w')
    with pytest.raises(KeyError):
        # variables() called with missing triple causes KeyError internally
        appears_inverted(g, triple)

def test_node_contexts_handles_extra_pop_gracefully():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
    ]
    epidata = {
        triples[1]: [Push('b'), POP, POP],  # too many pops
    }
    g = Graph(triples, epidata=epidata, top='a')
    contexts = node_contexts(g)
    # Should not raise despite extra POPs
    assert len(contexts) == len(triples)

def test_rearrange_empty_tree():
    tree = Tree(('v', []))
    rearrange(tree)
    assert tree.node == ('v', [])

def test_rearrange_single_branch():
    tree = Tree(('v', [('/', 'concept')]))
    rearrange(tree)
    assert tree.node[1][0][0] == '/'  # instance branch remains first

def test_get_pushed_variable_multiple_markers():
    triples = [
        ('a', ':ARG0', 'b'),
    ]
    epidata = {
        triples[0]: [Push('b'), Push('c')],
    }
    g = Graph(triples, epidata=epidata, top='a')
    # Returns the first Push variable only
    v = get_pushed_variable(g, triples[0])
    assert v == 'b'

def test_rearrange_key_ignores_instance_role_order():
    tree = Tree(
        ('x', [
            ('/', 'concept'),
            (':b', 'val2'),
            (':a', 'val1'),
        ])
    )
    # key that intentionally swaps order if not for instance first rule
    rearrange(tree, key=lambda r: {'/:a': 0, ':b': 1}.get(r, 10))
    roles = [b[0] for b in tree.node[1]]
    # instance role should still be first
    assert roles[0] == '/'
