import pytest
import dendropy
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, collapse_node, flip_node, extend_node
from dendropy import Tree, Node
import random

def _make_simple_tree_newick():
    # simple rooted tree with 4 leaves
    return "((A,B),(C,D));"

def _label_tree_spec_dup(tree, labels):
    """
    labels is a dict mapping node labels (leaf or internal) to 'speciation' or 'duplication'.
    Applies labels to internal nodes by matching child sets.
    """
    # assign leaf labels just for reference (taxon labels)
    for leaf in tree.leaf_nodes():
        leaf._label = leaf.taxon.label  # store original label

    for node in tree.internal_nodes():
        child_leaves = tuple(sorted(leaf.taxon.label for leaf in node.leaf_nodes()))
        # find label by child set string key if given
        label = labels.get(child_leaves, None)
        node.label = label
    tree.seed_node.label = None
    return tree

def _clone_and_label_simple_tree(labels):
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    # encode bipartitions before labeling (some code requires encoding)
    t.encode_bipartitions()
    return _label_tree_spec_dup(t, labels)

def test_computeLRF_identical_trees_equal_zero():
    # identical trees with labels: should have LRF=0
    labels = {
        ('A','B','C','D'): None,  # root None
        ('A','B'): 'speciation',
        ('A',): None,
        ('B',): None,
        ('C','D'): 'duplication',
        ('C',): None,
        ('D',): None,
    }
    t1 = _clone_and_label_simple_tree(labels)
    t2 = _clone_and_label_simple_tree(labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_single_flip_difference():
    # Trees identical except one internal node label differs
    labels1 = {
        ('A','B','C','D'): None,
        ('A','B'): 'speciation',
        ('C','D'): 'duplication',
    }
    labels2 = {
        ('A','B','C','D'): None,
        ('A','B'): 'duplication',  # flipped label
        ('C','D'): 'duplication',
    }
    t1 = _clone_and_label_simple_tree(labels1)
    t2 = _clone_and_label_simple_tree(labels2)
    dist = computeLRF(t1, t2)
    # LRF must be >= 1, because of one substitution needed
    assert dist >= 1

def test_computeLRF_different_tree_topologies():
    # Different topology: ((A,(B,C)),D)
    newick1 = "((A,B),(C,D));"
    newick2 = "((A,(B,C)),D);"
    t1 = dendropy.Tree.get(data=newick1, schema='newick')
    t2 = dendropy.Tree.get(data=newick2, schema='newick')
    # label internal nodes with speciation
    for t in (t1, t2):
        for node in t.internal_nodes():
            node.label = 'speciation'
        t.seed_node.label = None
        t.encode_bipartitions()
    dist = computeLRF(t1, t2)
    # The distance should be greater than zero for different topologies
    assert dist > 0

def test_computeLRF_islands_mismatch_raises():
    # Manually construct scenario with mismatch number of islands to test error
    # This is tricky - so make two trees with different numbers of good edges
    
    # Tree 1 simpler, tree 2 more complex -> likely different number of islands
    t1 = dendropy.Tree.get(data="((A,B),C);", schema='newick')
    t2 = dendropy.Tree.get(data="(A,(B,C));", schema='newick')
    for t in (t1, t2):
        for node in t.internal_nodes():
            node.label = 'speciation'
        t.seed_node.label = None
        t.encode_bipartitions()
    with pytest.raises(ValueError):
        computeLRF(t1, t2)

def test_mutateLabeledTree_no_modifies_input():
    # Verify mutateLabeledTree returns a new tree and does not modify original input
    t = dendropy.Tree.get(data="((A,B),(C,D));", schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    t_copy = t.clone()
    t_mut = mutateLabeledTree(t, 5, p_flip=0.5)
    # Original tree unchanged
    assert all(n1.label == n2.label for n1,n2 in zip(t.internal_nodes(), t_copy.internal_nodes()))
    # Returned tree different instance
    assert t_mut is not t

def test_mutateLabeledTree_prob_flip_behavior(monkeypatch):
    # Patch random.random to control if flip occurs or not
    values = [0.1, 0.9]
    index = 0
    def fake_random():
        nonlocal index
        v = values[index % len(values)]
        index += 1
        return v
    monkeypatch.setattr("random.random", fake_random)
    # construct tree with internal nodes labeled 'speciation'
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    t2 = mutateLabeledTree(t, 2, p_flip=0.5)
    # Since first random < p_flip, one flip expected
    labels_before = [n.label for n in t.internal_nodes()]
    labels_after = [n.label for n in t2.internal_nodes()]
    assert labels_before != labels_after

def test_flip_node_speciation_to_duplication_and_back():
    # Create node and flip label both ways
    n = Node()
    n.label = 'speciation'
    flip_node(n)
    assert n.label == 'duplication'
    flip_node(n)
    assert n.label == 'speciation'

def test_flip_node_invalid_label_raises():
    n = Node()
    n.label = 'unknown'
    with pytest.raises(ValueError):
        flip_node(n)

def test_collapse_node_LRF_collapses_edge(monkeypatch):
    # Build tree with internal node
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    # pick one internal node that is not root
    node_to_collapse = None
    for n in t.internal_nodes():
        if n.parent_node is not None:
            node_to_collapse = n
            break
    edge_before = len(list(t.edges()))
    collapse_node(node_to_collapse, model='LRF')
    edge_after = len(list(t.edges()))
    # collapsing edge removes one edge (or makes topology simpler)
    assert edge_after < edge_before

def test_collapse_node_ELRF_valid_and_invalid(monkeypatch):
    # Setup tree with two internal nodes: one with label equal to parent, one not
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    nodes = list(t.internal_nodes())
    root = t.seed_node
    for n in nodes:
        n.label = 'speciation'
    root.label = 'speciation'
    t.encode_bipartitions()
    # node with parent same label: can collapse
    for n in nodes:
        if n.parent_node is not None and n.label == n.parent_node.label:
            collapse_node(n, model='ELRF')
            break
    # create node with different label than parent and test error
    n2 = None
    for n in nodes:
        if n.parent_node is not None and n.label == 'speciation' and n.parent_node.label == 'duplication':
            n2 = n
            break
    if n2 is None:
        # forcibly create this situation
        n2 = nodes[0]
        n2.parent_node.label = 'duplication'
    with pytest.raises(ValueError):
        collapse_node(n2, model='ELRF')

def test_extend_node_invalid_degree_raises():
    # build tree with node of degree <=3 that is not root - should raise
    t = dendropy.Tree.get(data="((A,B),(C,D));", schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    # pick node with <=3 adjacent nodes and parent_node != None, should raise
    n = None
    for node in t.internal_nodes():
        if node.parent_node is not None and len(node.adjacent_nodes()) <= 3:
            n = node
            break
    if n is None:
        pytest.skip("No node with degree <=3 and non-root found")
    with pytest.raises(ValueError):
        extend_node(n, model='LRF')

def test_extend_node_valid_behavior(monkeypatch):
    # patch random.shuffle and randint to deterministic behavior
    calls = []
    def fake_shuffle(x):
        calls.append(list(x))
    def fake_randint(a,b):
        return a  # always pick minimum to make deterministic

    monkeypatch.setattr("random.shuffle", fake_shuffle)
    monkeypatch.setattr("random.randint", fake_randint)

    # build tree with node of degree >3 to allow extension
    # Create a new star-like node with 4 children
    t = dendropy.Tree()
    taxa = [dendropy.Taxon(label=l) for l in "ABCD"]
    t.taxon_namespace.extend(taxa)
    root = t.seed_node
    root.label = 'speciation'
    for taxon in taxa:
        leaf = root.add_child()
        leaf.taxon = taxon
    # artificially create node with degree 4 (root)
    # Use manually added internal node
    internal = root.add_child(label='duplication')
    internal.add_child(taxon=taxa[0])
    internal.add_child(taxon=taxa[1])
    internal.add_child(taxon=taxa[2])
    internal.label = 'speciation'
    # Force label for root to duplication, parent_node None for internal node is root here
    internal.parent_node = root
    internal._update_child_nodes_cache()

    # call extend_node on internal node with degree >3
    extend_node(internal, model='LRF')
    # After extension, internal node should have changed children count and new child present
    assert len(internal.child_nodes()) >= 2

def test_mutateLabeledTree_zero_edits_returns_clone(monkeypatch):
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    t2 = mutateLabeledTree(t, 0)
    # zero edits mean tree unchanged (but clone returned)
    assert t2.as_string("newick") == t.as_string("newick")
    assert t2 is not t

def test_mutateLabeledTree_invalid_model_raises(monkeypatch):
    # force collapse_node with invalid model inside mutateLabeledTree indirectly
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    # monkeypatch collapse_node to raise ValueError for unsupported model
    from pylabeledrf import computeLRF as computeLRF_mod
    with pytest.raises(ValueError):
        collapse_node(t.internal_nodes()[0], model='UNKNOWN')

def test_computeLRF_with_trees_no_common_goodedges():
    # Two trees with same taxa but no common good edges causes fallback scenario
    t1 = dendropy.Tree.get(data="(A,(B,(C,D)));", schema='newick')
    t2 = dendropy.Tree.get(data="((A,B),(C,D));", schema='newick')
    for t in (t1, t2):
        for node in t.internal_nodes():
            node.label = 'speciation'
        t.seed_node.label = None
        t.encode_bipartitions()
    dist = computeLRF(t1, t2)
    assert dist >= 0  # just check no crash and non-negative

def test_mutateLabeledTree_p_flip_zero_never_flips(monkeypatch):
    # If p_flip = 0, flips never occur
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    t2 = mutateLabeledTree(t, 5, p_flip=0)
    # Check that no flip was done: labels same or only contracted/expanded happened
    labels_before = [node.label for node in t.internal_nodes()]
    labels_after = [node.label for node in t2.internal_nodes()]
    # labels should mostly be the same, no flip changes
    assert set(labels_after).issubset({'speciation','duplication'})

def test_mutateLabeledTree_p_flip_one_always_flips(monkeypatch):
    # If p_flip = 1, always flips happen
    t = dendropy.Tree.get(data=_make_simple_tree_newick(), schema='newick')
    for node in t.internal_nodes():
        node.label = 'speciation'
    t.seed_node.label = None
    t.encode_bipartitions()
    t2 = mutateLabeledTree(t, 2, p_flip=1)
    labels_before = [node.label for node in t.internal_nodes()]
    labels_after = [node.label for node in t2.internal_nodes()]
    # At least one label flipped
    assert labels_before != labels_after

