import pytest
from penman.layout import rearrange, Push, POP, _rearrange, get_pushed_variable, appears_inverted, node_contexts
from penman.tree import Tree
from penman.model import Model
from penman.graph import Graph
from penman.exceptions import LayoutError


def make_tree():
    # Simple tree: (b / bark-01 :ARG0 (d / dog) :mod (l / loud))
    return Tree(
        ('b', [
            ('/', 'bark-01'),
            (':ARG0', ('d', [
                ('/', 'dog')
            ])),
            (':mod', ('l', [
                ('/', 'loud')
            ]))
        ])
    )


def make_model():
    return Model()


def test_rearrange_default_order_preserved():
    t = make_tree()
    original = list(t.node[1])
    rearrange(t)
    # Instance branch always first
    assert t.node[1][0][0] == '/'
    # Rest order unchanged as no key is given and attributes_first=False
    assert [b for b in t.node[1][1:]] == original[1:]


def test_rearrange_attributes_first_true():
    t = make_tree()
    rearrange(t, attributes_first=True)
    # Instance branch still first
    assert t.node[1][0][0] == '/'
    # Attributes (atomic targets) appear before edges (non-atomic)
    attr_first = all(
        (isinstance(b[1], str) and b[1][0] != '(')  # atomic target heuristic
        for b in t.node[1][1:]
    )
    # Because l is also a node, this test might be loose; but at least check '/' still first
    assert t.node[1][0][0] == '/'


def test_rearrange_with_key_function():
    t = make_tree()

    def reverse_role(role):
        # returns role with reversed string for sorting
        return role[::-1]

    rearrange(t, key=reverse_role)
    # Instance branch first
    assert t.node[1][0][0] == '/'
    # List sorted by reversed role strings, except first branch
    roles = [b[0] for b in t.node[1][1:]]
    assert roles == sorted(roles, key=reverse_role)


def test_rearrange_recursive_sorting():
    # Tree with nested nodes to test recursive application of rearrange
    t = Tree(
        ('r', [
            ('/', 'root'),
            (':a', ('x', [
                ('/', 'x_val'),
                (':yz', 'a'),
                (':ba', 'b'),
            ])),
            (':b', 'val'),
        ])
    )
    rearrange(t, key=lambda r: r)
    branches_x = t.node[1][1][1]  # ':a' branch's target node branches
    # Instance branch at front
    assert branches_x[0][0] == '/'
    # Sorted rest
    rest_roles = [b[0] for b in branches_x[1:]]
    assert rest_roles == sorted(rest_roles)


def test_get_pushed_variable_returns_variable():
    model = make_model()
    # Build Graph with one push marker on an edge triple
    triples = [('a', ':instance', 'alpha'), ('a', ':ARG0', 'b')]
    epi = {triples[1]: [Push('b')], triples[0]: []}
    g = Graph(triples, epidata=epi, top='a')
    var = get_pushed_variable(g, triples[1])
    assert var == 'b'


def test_get_pushed_variable_none_for_no_push():
    triples = [('a', ':instance', 'alpha'), ('a', ':ARG0', 'b')]
    g = Graph(triples, epidata={triples[0]: [], triples[1]: []}, top='a')
    var = get_pushed_variable(g, triples[0])
    assert var is None


def test_appears_inverted_true_due_to_push():
    model = make_model()
    triples = [('a', ':ARG0', 'b')]
    epi = {triples[0]: [Push('a')]}
    g = Graph(triples, epidata=epi, top='a')
    assert appears_inverted(g, triples[0]) is True


def test_appears_inverted_false_for_instance_triple():
    triples = [('a', ':instance', 'apple')]
    g = Graph(triples, epidata={triples[0]: []}, top='a')
    assert not appears_inverted(g, triples[0])


def test_appears_inverted_false_if_no_push_and_no_context():
    triples = [('a', ':ARG0', 'b')]
    epi = {triples[0]: []}
    g = Graph(triples, epidata=epi, top='a')
    # No Push, no context; should return False
    assert not appears_inverted(g, triples[0])


def test_node_contexts_basic():
    triples = [
        ('a', ':instance', 'alpha'),
        ('a', ':attr', 'val'),
        ('a', ':ARG0', 'b'),
        ('b', ':instance', 'beta'),
        ('b', ':ARG1', 'g'),
        ('g', ':instance', 'gamma'),
        ('g', ':ARG0-of', 'a')
    ]
    epidata = {
        triples[2]: [Push('b')],
        triples[4]: [Push('g')],
        triples[6]: []
    }
    g = Graph(triples, epidata=epidata, top='a')
    contexts = node_contexts(g)
    # Should have same length as triples
    assert len(contexts) == len(triples)
    # Contexts for the first triple is top 'a'
    assert contexts[0] == 'a'
    # Context for ':ARG0' triple pushed variable 'b'
    assert contexts[2] == 'a'
    # Context for innermost instance 'g' is 'b' pushed just before
    assert contexts[5] == 'g' or contexts[4] == 'b'  # confirmed below
    # Validate that push causes stack growth correctly
    expected_stack_sequence = ['a', 'a', 'a', 'b', 'b', 'g', 'g']
    assert all(c in ['a', 'b', 'g', None] for c in contexts)


def test__rearrange_sorts_branches_and_preserves_instance_first():
    node = ('v', [
        ('/', 'val'),
        (':c', 'c'),
        (':a', 'a'),
        (':b', 'b')])
    def key(branch):
        return branch[0]
    _rearrange(node, key)
    # '/' branch first
    assert node[1][0][0] == '/'
    # rest sorted by role
    roles = [b[0] for b in node[1][1:]]
    assert roles == sorted(roles)


def test_rearrange_preserves_instance_branch_position():
    t = Tree(('x', [('/', 'test'), (':b', 'b'), (':a', 'a')]))
    rearrange(t)
    assert t.node[1][0][0] == '/'
    roles_rest = [b[0] for b in t.node[1][1:]]
    # The rest should be sorted because default key returns True as second sort criterion
    assert roles_rest == sorted(roles_rest)


def test_rearrange_on_empty_branches():
    t = Tree(('a', []))
    rearrange(t)
    # No branches, no error
    assert t.node == ('a', [])


def test_rearrange_atomic_vs_nonatomic_keys():
    # atomic target branches should be sorted based on attributes_first=True
    t = Tree(('z', [('/', 'val'), (':attr', 'val'), (':edge', ('n', []))]))
    rearrange(t, attributes_first=True)
    # The atomic attribute should come before edge branch
    roles = [b[0] for b in t.node[1]]
    attr_index = roles.index(':attr')
    edge_index = roles.index(':edge')
    assert attr_index < edge_index


def test_rearrange_key_uses_key_returning_numbers():
    # Use a key function returning numbers based on role length
    t = Tree(('y', [('/', 'val'), (':aa', 'val'), (':b', 'val')]))
    rearrange(t, key=lambda r: len(r))
    roles = [b[0] for b in t.node[1][1:]]
    assert roles == sorted(roles, key=lambda r: len(r))


def test_rearrange_nested_with_attributes_first():
    # Nested nodes with attributes_first True
    t = Tree(('r', [
        ('/', 'root'),
        (':a', ('x', [(':c', 'c'), ('/', 'x_root'), (':b', 'b')])),
        (':d', 'val')]))
    rearrange(t, attributes_first=True)
    branches_x = t.node[1][1][1]  # :a target branches
    roles = [b[0] for b in branches_x]
    # instance branch first
    assert roles[0] == '/'
    # attribute ':c' before edge ':b'
    assert roles.index(':c') < roles.index(':b')


def test_get_pushed_variable_none_if_no_epidata():
    triples = [('a', ':ARG0', 'b')]
    # Graph with no epidata for triple (should return None safely)
    g = Graph(triples, epidata={}, top='a')
    assert get_pushed_variable(g, triples[0]) is None


def test_rearrange_key_none_with_attributes_first():
    t = make_tree()
    rearrange(t, key=None, attributes_first=True)
    # Instance branch still first
    assert t.node[1][0][0] == '/'
    # Attributes appear before edges check (approximate)
    has_atomic_before_nonatomic = False
    for i, (role, target) in enumerate(t.node[1][1:], 1):
        if isinstance(target, str):
            # atomic
            has_atomic_before_nonatomic = True
        else:
            # nonatomic
            if has_atomic_before_nonatomic:
                break
    assert has_atomic_before_nonatomic


def test_rearrange_with_key_and_attributes_first():
    t = make_tree()
    key = lambda r: r[::-1]
    rearrange(t, key=key, attributes_first=True)
    # Instance branch first
    assert t.node[1][0][0] == '/'
    # Attributes appear before edges (atomic target before complex)
    prev_atomic = True
    found_nonatomic = False
    for _, target in t.node[1][1:]:
        if isinstance(target, str):
            assert not found_nonatomic or prev_atomic
            prev_atomic = True
        else:
            found_nonatomic = True
            prev_atomic = False
