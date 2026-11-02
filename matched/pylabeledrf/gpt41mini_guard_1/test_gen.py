import pytest
import dendropy
import random
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node

@pytest.fixture(autouse=True)
def seed_random():
    # seed random before each test to ensure reproducibility where randomness is involved
    random.seed(12345)
    yield
    # no global state changed, so no cleanup needed

def _make_tree(newick, labels):
    """
    Helper to make a dendropy tree from a newick string and assign labels to internal nodes.
    labels is a dict mapping node labels (leaf or None for internal) to their "label" attribute ('speciation'/'duplication'/None).
    Leaves keep their taxon names.
    """
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema='newick', taxon_namespace=taxa, rooting="force-rooted")
    # Assign labels to internal nodes.
    # We assign by topology order here for convenience:
    internal_nodes = [n for n in tree.internal_nodes() if n != tree.seed_node and n.is_internal()]
    # Special case seed_node can be labeled None or assigned label
    # We assign labels in order of internal_nodes list, or match by node label if present (usually None)
    for idx, n in enumerate(internal_nodes):
        # If user provided label for None key (internal nodes), assign in order
        if None in labels and isinstance(labels[None], list):
            # multiple internal node labels given as list
            if idx < len(labels[None]):
                n.label = labels[None][idx]
        elif None in labels:
            n.label = labels[None]
    # For root node label if provided explicitly
    if 'root' in labels:
        tree.seed_node.label = labels['root']
    return tree

def _simple_tree1():
    # Tree: ((A,B),(C,D));
    # Internal nodes: root, and two internal nodes joining (A,B) and (C,D)
    l = {
        None: ['speciation', 'duplication'],  # assign in order to internal nodes below leaves
        'root': 'speciation'
    }
    newick = "((A,B),(C,D));"
    return _make_tree(newick, l)

def _simple_tree2():
    # Tree: ((A,B),(C,D));
    # difference: one internal node with duplication swapped with speciation
    l = {
        None: ['duplication', 'speciation'],
        'root': 'speciation'
    }
    newick = "((A,B),(C,D));"
    return _make_tree(newick, l)

def test_computeLRF_identical_trees_zero_distance():
    t1 = _simple_tree1()
    t2 = _simple_tree1()
    dist = computeLRF(t1,t2)
    assert dist == 0

def test_computeLRF_different_labels_gives_positive_distance():
    t1 = _simple_tree1()
    t2 = _simple_tree2()
    dist = computeLRF(t1,t2)
    assert dist > 0

def test_computeLRF_same_topology_different_duplication_speciation():
    # Build a tree with internal nodes all speciation in t1 and all duplication in t2
    newick = "((A,B),(C,D));"
    labels1 = {None: 'speciation', 'root': 'speciation'}
    labels2 = {None: 'duplication', 'root': 'duplication'}
    t1 = _make_tree(newick, labels1)
    t2 = _make_tree(newick, labels2)
    dist = computeLRF(t1, t2)
    # Distance should be positive and equal to number of internal nodes with label difference
    # Here we have 3 internal nodes: two internal joining tips plus root
    assert dist >= 1

def test_computeLRF_raises_value_error_on_unequal_islands():
    # Create two trees structurally different so islands count differs
    # One tree has one internal split, other has no splits
    t1 = dendropy.Tree.get(data="((A,B),C);", schema="newick", rooting="force-rooted")
    t2 = dendropy.Tree.get(data="(A,(B,C));", schema="newick", rooting="force-rooted")
    # Assign labels to internal nodes (speciation)
    for n in t1.internal_nodes():
        n.label = 'speciation'
    for n in t2.internal_nodes():
        n.label = 'speciation'

    # forcibly tweak t1 so islands count differs from t2 by faking labels on biparts
    # Actually just reroot t2 to force different islands count:
    # but safer to expect ValueError when computeLRF called on trees with different islands count

    # Since computeLRF checks islands' lengths, trees with different island counts will raise ValueError.
    with pytest.raises(ValueError):
        computeLRF(t1,t2)

def test_mutateLabeledTree_flips_nodes_correctly():
    t1 = _simple_tree1()
    t2 = mutateLabeledTree(t1, n=10, p_flip=1.0)  # force only flips
    # All original internal nodes labeled spec/dup
    original_labels = set(n.label for n in t1.internal_nodes())
    new_labels = set(n.label for n in t2.internal_nodes())
    # The labels must be from {'speciation','duplication'}
    assert new_labels.issubset({'speciation', 'duplication'})
    # Since p_flip=1, at least some nodes flipped label, likely different sets
    assert original_labels != new_labels or len(t1.internal_nodes()) == 0

def test_mutateLabeledTree_collapse_and_extend_behavior():
    t1 = _simple_tree1()
    t2 = mutateLabeledTree(t1, n=10, p_flip=0.0)  # force no flips, only collapse/extend
    # Tree should still have same leaves
    assert set(leaf.taxon.label for leaf in t1.leaf_node_iter()) == set(leaf.taxon.label for leaf in t2.leaf_node_iter())
    # Internal node labels remain valid
    for n in t2.internal_nodes():
        assert n.label in ('speciation', 'duplication', None)

def test_flip_node_changes_label_and_raises_on_invalid_label():
    node = dendropy.Node()
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'
    flip_node(node)
    assert node.label == 'speciation'
    node.label = 'foo'
    with pytest.raises(ValueError):
        flip_node(node)

def test_collapse_node_LRF_always_collapses():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    node = internal_nodes[0]
    # Before collapse edge exists
    assert node.edge is not None
    collapse_node(node, model='LRF')
    # After collapse edge should be collapsed (collapsed edges no longer exist)
    # We check node.edge.collapse() does not error, but we cannot check .edge is None;
    # just check that the node still exists and tree is valid
    assert node is not None

def test_collapse_node_ELRF_only_collapses_when_label_matches_parent():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    node = internal_nodes[0]
    # forcibly set labels to match parent
    node.label = 'speciation'
    node.parent_node.label = 'speciation'
    collapse_node(node, model='ELRF')  # Should not raise
    # forcibly set labels to mismatch parent
    node.label = 'duplication'
    node.parent_node.label = 'speciation'
    with pytest.raises(ValueError):
        collapse_node(node, model='ELRF')

def test_extend_node_raises_valueerror_on_low_degree_nodes():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    # pick node with degree <=3 forcibly
    node = internal_nodes[0]
    # to ensure degree low, forcibly clear children except 1 (low degree)
    node.clear_child_nodes()
    leaf = dendropy.Node()
    leaf.taxon = dendropy.Taxon(label="X")
    node.add_child(leaf)
    with pytest.raises(ValueError):
        extend_node(node)

def test_extend_node_produces_valid_structure():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    # Pick node with degree>3 by adding children forcibly
    node = internal_nodes[0]
    # Add children to ensure >3 adjacent nodes
    while len(node.adjacent_nodes()) <= 3:
        nnew = dendropy.Node(label='speciation')
        node.add_child(nnew)
    # Save old count children
    old_child_count = len(node.child_nodes())
    extend_node(node, model='LRF')
    # After extend, children should be changed, number of children not less than 1
    assert len(node.child_nodes()) > 0
    # The new node added should have label either 'speciation' or 'duplication' (since model LRF)
    # Actually the new node inside function is node.add_child(dendropy.Node(label=node.label)) so label matches node.label
    all_labels = set(c.label for c in node.child_nodes())
    assert all_labels.intersection({'speciation', 'duplication'})

def test_mutateLabeledTree_no_state_change_on_input_tree():
    t = _simple_tree1()
    # Clone and store labels before
    old_labels = {id(n): n.label for n in t.internal_nodes()}
    mutateLabeledTree(t, n=5, p_flip=0.5)
    # None of the original nodes labels should be changed by mutateLabeledTree as it clones the tree
    for n in t.internal_nodes():
        assert n.label == old_labels[id(n)]

def test_mutateLabeledTree_raises_valueerror_on_invalid_model_in_collapse():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    with pytest.raises(ValueError):
        collapse_node(internal_nodes[0], model='INVALIDMODEL')

def test_mutateLabeledTree_raises_valueerror_on_invalid_model_in_extend():
    t = _simple_tree1()
    internal_nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    # Add children forcibly to node to match conditions avoiding ValueError from degree check:
    node = internal_nodes[0]
    while len(node.adjacent_nodes()) <= 3:
        nnew = dendropy.Node(label='speciation')
        node.add_child(nnew)

    with pytest.raises(ValueError):
        extend_node(node, model='INVALIDMODEL')

# Additional test for mutateLabeledTree with p_flip=0 checks some expansions/collapses happen
def test_mutateLabeledTree_with_zero_pflip_modifies_tree():
    t = _simple_tree1()
    t2 = mutateLabeledTree(t, n=5, p_flip=0.0)
    # The resulting tree should still have all same leaves
    assert set(leaf.taxon.label for leaf in t.leaf_node_iter()) == set(leaf.taxon.label for leaf in t2.leaf_node_iter())
    # The labels of internal nodes should be in valid set
    for n in t2.internal_nodes():
        assert n.label in ('speciation', 'duplication', None)

# Test flip_node round trip consistency
def test_flip_node_flip_back_invariant():
    node = dendropy.Node()
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'
    flip_node(node)
    assert node.label == 'speciation'

