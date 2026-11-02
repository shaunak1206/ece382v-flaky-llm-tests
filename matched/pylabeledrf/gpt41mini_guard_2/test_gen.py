import pytest
import dendropy
import random
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node

# Helpers to create simple labeled trees for testing
def make_tree_newick(newick, labels):
    """
    Create a dendropy tree from newick string and assign internal node labels according to labels dict.
    :param newick: Newick string
    :param labels: dict mapping node splits (frozenset of leaf labels) to label ('speciation' or 'duplication')
    :return: dendropy.Tree with labeled internal nodes
    """
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)

    tree.encode_bipartitions()

    # Assign labels to internal nodes based on leaf descendant sets matching keys in labels
    label_map = {}
    for node in tree.internal_nodes():
        leaves = frozenset(leaf.taxon.label for leaf in node.leaf_nodes())
        label_map[leaves] = node

    for leafset, label in labels.items():
        node = label_map.get(frozenset(leafset))
        if node is not None:
            node.label = label

    # Ensure root label is None (consistent with the code)
    tree.seed_node.label = None

    return tree

def clone_tree_with_labels(tree):
    """Return a clone of the tree with the same internal node labels"""
    t = tree.clone(depth=1)
    return t

# Test trees definitions

@pytest.fixture
def simple_tree_1():
    # Tree: (A,(B,C));
    # Label the single internal node (B,C) as 'speciation'
    newick = "(A,(B,C));"
    labels = {('B', 'C'): 'speciation'}
    return make_tree_newick(newick, labels)

@pytest.fixture
def simple_tree_2():
    # Tree: ((A,B),C);
    # Label internal node (A,B) as 'duplication'
    newick = "((A,B),C);"
    labels = {('A', 'B'): 'duplication'}
    return make_tree_newick(newick, labels)

@pytest.fixture
def identical_tree():
    # Tree: ((A,B),C);
    # Label internal node (A,B) as 'speciation'
    newick = "((A,B),C);"
    labels = {('A', 'B'): 'speciation'}
    return make_tree_newick(newick, labels)

@pytest.fixture
def two_level_tree():
    # Tree: ((A,B),(C,D));
    # Label (A,B) and (C,D) nodes as 'duplication', root as None
    newick = "((A,B),(C,D));"
    labels = {('A', 'B'): 'duplication', ('C', 'D'): 'duplication'}
    return make_tree_newick(newick, labels)

# Tests for computeLRF

def test_computeLRF_identical(simple_tree_1):
    # Distance between identical trees must be 0
    t1 = simple_tree_1
    t2 = simple_tree_1.clone(depth=1)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_topology_difference(simple_tree_1, simple_tree_2):
    # Different topology and labels, distance should be >0
    dist = computeLRF(simple_tree_1, simple_tree_2)
    assert dist > 0

def test_computeLRF_label_difference(identical_tree):
    # Same topology, but label difference increases distance
    t1 = identical_tree
    t2 = identical_tree.clone(depth=1)
    # Flip label on internal node
    for n in t2.internal_nodes():
        n.label = 'duplication' if n.label == 'speciation' else 'speciation'
    dist = computeLRF(t1, t2)
    # Distance at least 1 because of label difference
    assert dist >= 1

def test_computeLRF_unlabeled_internal_node():
    # Tree with an internal node unlabeled (label None) compared to one with label
    newick = "((A,B),C);"
    taxa = dendropy.TaxonNamespace()
    t1 = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    t2 = t1.clone(depth=1)
    # Assign all internal labels in t2
    for node in t2.internal_nodes():
        node.label = 'speciation'
    # t1 has unlabeled internal nodes (label None)
    dist = computeLRF(t1, t2)
    # Distance should be >= 1 due to label differences
    assert dist >= 1

def test_computeLRF_island_mismatch_raises():
    # Build two trees with different number of islands (should raise ValueError)
    # Construct trees manually to force different islands
    
    # Tree 1: ((A,B),(C,D));
    newick1 = "((A,B),(C,D));"
    labels1 = {('A', 'B'): 'speciation', ('C', 'D'): 'duplication'}
    t1 = make_tree_newick(newick1, labels1)
    
    # Tree 2: (A,(B,(C,D)));
    newick2 = "(A,(B,(C,D)));"
    labels2 = {('C', 'D'): 'speciation', ('B', 'C', 'D'): 'duplication'}
    t2 = make_tree_newick(newick2, labels2)

    # Intentionally corrupt islands by removing some bipartitions from t2 before call,
    # but since computeLRF clones, we cannot mutate bipartition sets.
    # Instead monkeypatch identify_labelled_bipartitions to return sets of different sizes.
    import pylabeledrf.computeLRF as computeLRFmod

    original_identify = computeLRFmod.identify_labelled_bipartitions

    def fake_identify(t):
        b, bTn = original_identify(t)
        if t == t2:
            # Remove one bipartition to create mismatch in islands count
            if b:
                b.pop()
        return b,bTn

    computeLRFmod.identify_labelled_bipartitions = fake_identify

    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF(t1, t2)

    computeLRFmod.identify_labelled_bipartitions = original_identify


# Tests for mutateLabeledTree

def test_mutateLabeledTree_flips_only():
    # Seed random to have reproducible flips
    random.seed(42)

    # Prepare a tree with internal labels
    newick = "((A,B),(C,D));"
    labels = {('A', 'B'): 'speciation', ('C', 'D'): 'duplication'}
    tree = make_tree_newick(newick, labels)

    # Call mutateLabeledTree with p_flip=1 so only flips occur
    mutated = mutateLabeledTree(tree, n=5, p_flip=1.0, model='LRF')

    # Check that tree remains same shape
    assert mutated.taxon_namespace.labels() == tree.taxon_namespace.labels()

    # Check that at least one internal node label changed
    original_labels = {frozenset(n.leaf_nodes()[i].taxon.label for i in range(len(n.leaf_nodes()))): n.label 
                       for n in tree.internal_nodes()}
    mutated_labels = {frozenset(n.leaf_nodes()[i].taxon.label for i in range(len(n.leaf_nodes()))): n.label 
                      for n in mutated.internal_nodes()}

    differences = sum(1 for k in original_labels if original_labels[k] != mutated_labels.get(k))
    assert differences >= 1

def test_mutateLabeledTree_collapse_or_extend():
    random.seed(123)

    newick = "(((A,B),C),(D,E));"
    labels = {('A', 'B'): 'speciation', ('A', 'B', 'C'): 'duplication', ('D', 'E'): 'speciation'}
    tree = make_tree_newick(newick, labels)

    mutated = mutateLabeledTree(tree, n=3, p_flip=0.0, model='LRF')

    # Ensure number of leaves/stats unchanged (topology can change internally)
    assert set(leaf.taxon.label for leaf in mutated.leaf_nodes()) == {'A','B','C','D','E'}

def test_mutateLabeledTree_model_ELRF_vs_LRF():
    # Check that collapse_node raises ValueError with ELRF for invalid collapse edge
    
    # Create a small tree with labels, parent and child differ
    newick = "((A,B),C);"
    labels = {('A', 'B'): 'speciation'}
    tree = make_tree_newick(newick, labels)
    t = tree.clone(depth=1)

    # Pick a node whose parent label is different to its label to test collapse_node with model ELRF
    node = None
    for n in t.internal_nodes():
        if n.parent_node and n.label != n.parent_node.label and n.label is not None:
            node = n
            break
    if node is None:
        # forcibly set label difference to test
        n = next(iter(t.internal_nodes()))
        n.label = "speciation"
        n.parent_node.label = "duplication"
        node = n

    # collapse_node with model='ELRF' should raise
    import pytest
    with pytest.raises(ValueError):
        collapse_node(node, model='ELRF')

def test_flip_node_changes_label():
    # Setup node object with mutable label attribute
    class DummyNode:
        def __init__(self, label):
            self.label = label

    n_spec = DummyNode("speciation")
    flip_node(n_spec)
    assert n_spec.label == "duplication"

    n_dup = DummyNode("duplication")
    flip_node(n_dup)
    assert n_dup.label == "speciation"

    n_bad = DummyNode("other")
    with pytest.raises(ValueError):
        flip_node(n_bad)

def test_extend_node_invalid_degree_raises():
    random.seed(1)

    # Node with adjacents <= 3 and parent_node != None should raise
    class DummyNode:
        def __init__(self, label):
            self.label = label
            self._children = [self] * 2  # just 2 children
            self.parent_node = object()
        def adjacent_nodes(self):
            return [1,2]  # length 2 <= 3
        def incident_edges(self):
            return [1,2]
        def child_nodes(self):
            return self._children
        def clear_child_nodes(self):
            self._children.clear()
        def add_child(self, node):
            self._children.append(node)

    node = DummyNode("speciation")
    with pytest.raises(ValueError):
        extend_node(node, model='LRF')

def test_mutateLabeledTree_seed_p_flip_half():
    # Test reproducibility of mutateLabeledTree with fixed seed and p_flip=0.5

    random.seed(2023)
    newick = "((A,B),(C,D));"
    labels = {('A', 'B'): 'speciation', ('C', 'D'): 'duplication'}
    tree = make_tree_newick(newick, labels)

    mutated1 = mutateLabeledTree(tree, n=4, p_flip=0.5)
    random.seed(2023)
    mutated2 = mutateLabeledTree(tree, n=4, p_flip=0.5)

    # The two mutated trees should be identical (labels and structure)
    for n1, n2 in zip(mutated1.internal_nodes(), mutated2.internal_nodes()):
        assert n1.label == n2.label

def test_computeLRF_flower_tree_label_difference():
    # More complex tree with multiple internal nodes and labels, testing label difference influence

    newick = "((A,(B,C)),(D,E));"
    labels1 = {('B','C'): 'speciation', ('A','B','C'): 'duplication', ('D','E'): 'speciation'}
    labels2 = {('B','C'): 'duplication', ('A','B','C'): 'duplication', ('D','E'): 'speciation'}

    t1 = make_tree_newick(newick, labels1)
    t2 = make_tree_newick(newick, labels2)

    dist = computeLRF(t1, t2)
    # Distance should be at least 1 due to label difference in ('B','C')
    assert dist >= 1

def test_computeLRF_zero_if_same_tree_and_labels(two_level_tree):
    dist = computeLRF(two_level_tree, two_level_tree.clone(depth=1))
    assert dist == 0

def test_mutateLabeledTree_invalid_model_raises():
    newick = "(A,B,C);"
    tree = make_tree_newick(newick, {})

    with pytest.raises(ValueError):
        mutateLabeledTree(tree, n=1, model='INVALID')

def test_collapse_node_invalid_model_raises(simple_tree_1):
    node = next(simple_tree_1.internal_nodes())
    with pytest.raises(ValueError):
        collapse_node(node, model='INVALID')

def test_mutateLabeledTree_does_not_modify_input(simple_tree_2):
    tree_copy = simple_tree_2.clone(depth=1)
    mutated = mutateLabeledTree(simple_tree_2, n=2)
    # original input unchanged
    orig_labels = [n.label for n in simple_tree_2.internal_nodes()]
    copy_labels = [n.label for n in tree_copy.internal_nodes()]
    assert orig_labels == copy_labels

def test_extend_node_modified_children(monkeypatch):
    random.seed(5)

    # Build a tree with 4 children for a node
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data="(((A,B),(C,D)),(E,F));", schema="newick", taxon_namespace=taxa)
    tree.encode_bipartitions()

    # pick a node with >= 3 children, artificially set labels etc
    node = None
    for n in tree.internal_nodes():
        if len(n.child_nodes()) >= 3:
            node = n
            node.label = 'speciation'
            break

    # patch random.shuffle to track if it was called
    calls = []
    def fake_shuffle(x):
        calls.append(list(x))
        # do a no-op shuffle for test reproducibility
    import pylabeledrf.computeLRF as computeLRFmod
    monkeypatch.setattr(computeLRFmod.random, "shuffle", fake_shuffle)
    # patch random.randint to fixed value for reproducibility
    monkeypatch.setattr(computeLRFmod.random, "randint", lambda a,b: 2)

    extend_node(node, model='LRF')

    # We expect exactly one call to shuffle (children list)
    assert len(calls) == 1
    # After extension, node should have two children and one new child with some children
    assert len(node.child_nodes()) >= 2

def test_flip_node_invalid_value_raises():
    class FakeNode:
        def __init__(self):
            self.label = "invalid_label"

    node = FakeNode()
    with pytest.raises(ValueError):
        flip_node(node)

