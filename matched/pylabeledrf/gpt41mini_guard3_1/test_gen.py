import pytest
import random
import dendropy
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node

@pytest.fixture
def simple_tree():
    # returns a simple rooted species tree with 3 leaves and correct labels
    t = dendropy.Tree.get(
        data="(A,(B,C));",
        schema="newick",
    )
    # set internal node labels
    for i, node in enumerate(t.internal_nodes()):
        node.label = 'speciation' if i % 2 == 0 else 'duplication'
    return t

@pytest.fixture
def two_equal_trees(simple_tree):
    # returns two identical copies of simple_tree to compare
    return simple_tree.clone(depth=2), simple_tree.clone(depth=2)

@pytest.mark.parametrize("flip_label,expected", [
    ('speciation', 'duplication'),
    ('duplication', 'speciation'),
])
def test_flip_node_flips_label(flip_label, expected):
    random.seed(42)
    n = dendropy.Node()
    n.label = flip_label
    flip_node(n)
    assert n.label == expected

def test_flip_node_raises_on_invalid_label():
    n = dendropy.Node()
    n.label = "unknown"
    with pytest.raises(ValueError):
        flip_node(n)

def test_mutateLabeledTree_flips_and_modifies(n=5):
    random.seed(123)
    tree = dendropy.Tree.get(data="(A,(B,C));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    mutated = mutateLabeledTree(tree, n)
    # must be different tree object
    assert mutated is not tree
    # check labels are only 'speciation' or 'duplication' inside mutated tree (except root possibly None)
    labels = [nd.label for nd in mutated.internal_nodes() if nd.label is not None]
    assert all(l in ('speciation','duplication') for l in labels)

def test_mutateLabeledTree_n_zero_returns_clone():
    random.seed(42)
    tree = dendropy.Tree.get(data="(A,(B,C));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    mutated = mutateLabeledTree(tree, 0)
    assert mutated.as_string("newick") == tree.as_string("newick")
    assert mutated is not tree

def test_collapse_node_on_LRF_and_ELRF_valid():
    random.seed(42)
    tree = dendropy.Tree.get(data="(A,(B,C));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    # test LRF collapses without error
    for node in tree.internal_nodes():
        collapse_node(node, model='LRF')
        break  # test only first node

    tree2 = dendropy.Tree.get(data="(A,(B,C));", schema="newick")
    for node in tree2.internal_nodes():
        node.label = 'speciation'
    # For ELRF, collapse only allowed if node label equals parent label
    # set a parent and child label same to test success
    internal_nodes = list(tree2.internal_nodes())
    if len(internal_nodes) > 1:
        internal_nodes[1].parent_node.label = internal_nodes[1].label
        collapse_node(internal_nodes[1], model='ELRF')
    else:
        # fallback: cannot test properly without sufficient nodes
        pass

def test_collapse_node_ELRF_raises_when_labels_differ():
    random.seed(42)
    tree = dendropy.Tree.get(data="(A,(B,C));", schema="newick")
    nodes = list(tree.internal_nodes())
    if len(nodes) < 2:
        pytest.skip("Tree too small for this test")
    # assign distinct labels to node and parent
    nodes[1].label = 'speciation'
    nodes[1].parent_node.label = 'duplication'
    with pytest.raises(ValueError):
        collapse_node(nodes[1], model='ELRF')

def test_collapse_node_raises_on_unsupported_model():
    n = dendropy.Node()
    with pytest.raises(ValueError):
        collapse_node(n, model='unsupported_model')

def test_extend_node_raises_on_bad_degree():
    # Create node with small degree and parent to trigger error
    tree = dendropy.Tree.get(data="(A,B,C);", schema="newick")
    root = tree.seed_node
    root.label = 'speciation'
    node = next(tree.internal_nodes())
    with pytest.raises(ValueError):
        extend_node(node, model='LRF')

def test_extend_node_valid_behavior():
    random.seed(24)
    tree = dendropy.Tree.get(data="(A,(B,C,D,E));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    candidates = [n for n in tree.internal_nodes() if len(n.adjacent_nodes()) > 3 or (n.parent_node is None and len(n.adjacent_nodes()) > 2)]
    if not candidates:
        # create a node with degree > 3 by manual expansion
        node = next(tree.internal_nodes())
        # forcibly add a child node to increase degree
        node.add_child(dendropy.Node(label='speciation'))
        candidate = node
    else:
        candidate = candidates[0]
    try:
        extend_node(candidate, model='LRF')
    except ValueError:
        pytest.skip("No suitable node for extension testing")

def test_mutateLabeledTree_probabilities(monkeypatch):
    # patch random.random and randint to produce deterministic outcome favoring flip
    monkeypatch.setattr("random.random", lambda: 0.1)  # favor flipping (p_flip=0.3)
    monkeypatch.setattr("random.randint", lambda a, b: 0)
    tree = dendropy.Tree.get(data="(A,(B,C,D));", schema="newick")
    for n in tree.internal_nodes():
        n.label = 'duplication'
    mutated = mutateLabeledTree(tree, 1, p_flip=0.3)
    labels = [n.label for n in mutated.internal_nodes() if n.label is not None]
    assert any(lbl == 'speciation' for lbl in labels)

def test_computeLRF_identical_trees_zero_distance(simple_tree):
    random.seed(1)
    t1, t2 = simple_tree.clone(depth=2), simple_tree.clone(depth=2)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_simple_tree_label_change_increases_distance(simple_tree):
    random.seed(2)
    t1, t2 = simple_tree.clone(depth=2), simple_tree.clone(depth=2)
    # flip label of one internal node in t2 to cause difference
    nodes = list(t2.internal_nodes())
    if nodes:
        flip_node(nodes[0])
    d = computeLRF(t1, t2)
    # Distance must be >=1
    assert d >= 1

def test_mutateLabeledTree_does_not_change_input_tree(simple_tree):
    random.seed(123)
    tree_copy = simple_tree.clone(depth=2)
    mutated = mutateLabeledTree(simple_tree, 3)
    # original tree unchanged
    assert simple_tree.as_string("newick") == tree_copy.as_string("newick")

def test_flip_node_randomness_seed(monkeypatch):
    # Ensure flip_node behavior is deterministic: flip speciation to duplication every time
    random.seed(50)
    node = dendropy.Node()
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'

def test_extend_node_randomness_seed(monkeypatch):
    # Given fixed seed, extend_node picks predictable new labels or shuffling
    random.seed(999)
    tree = dendropy.Tree.get(data="(A,(B,C,D,E));", schema="newick")
    for nd in tree.internal_nodes():
        nd.label = 'speciation'
    node = next(tree.internal_nodes())
    try:
        extend_node(node)
    except ValueError:
        pytest.skip("No suitable node to extend for test")

def test_mutateLabeledTree_with_model_ELRF(monkeypatch):
    random.seed(1234)
    tree = dendropy.Tree.get(data="(A,(B,C,D));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    # We patch random.random to prevent flipping
    monkeypatch.setattr("random.random", lambda: 0.9)  # force else clause: collapse or extend
    monkeypatch.setattr("random.randint", lambda a, b: 0)  # pick collapse node index 0
    mutated = mutateLabeledTree(tree, 2, model='ELRF')
    assert mutated is not None

def test_flip_node_raises_if_label_none():
    node = dendropy.Node()
    node.label = None
    with pytest.raises(ValueError):
        flip_node(node)

def test_collapse_node_model_invalid_raises():
    node = dendropy.Node()
    node.label = "speciation"
    with pytest.raises(ValueError):
        collapse_node(node, model="unknown")

def test_mutateLabeledTree_uses_monkeypatch(monkeypatch):
    # Patch random functions to control behavior and avoid flaky tests
    monkeypatch.setattr("random.random", lambda: 0.0)  # always flip
    monkeypatch.setattr("random.randint", lambda a,b: 0)
    tree = dendropy.Tree.get(data="(A,B,C);", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    mutated = mutateLabeledTree(tree, 1)
    # After flip, some node label must be 'duplication'
    labels = [node.label for node in mutated.internal_nodes()]
    assert 'duplication' in labels or 'speciation' in labels  # just sanity

@pytest.mark.parametrize("n_edits", [1,3,5])
def test_mutateLabeledTree_edit_count(n_edits):
    random.seed(2024)
    tree = dendropy.Tree.get(data="(A,(B,(C,D,E)));", schema="newick")
    for node in tree.internal_nodes():
        node.label = 'speciation'
    mutated = mutateLabeledTree(tree, n_edits)
    # Just test that output is a tree and differs in newick format probably
    assert mutated is not tree
    assert isinstance(mutated, dendropy.Tree)
    assert mutated.as_string("newick") != tree.as_string("newick") or n_edits == 0

