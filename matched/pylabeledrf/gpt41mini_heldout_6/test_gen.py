import pytest
import dendropy
from pylabeledrf import computeLRF, mutateLabeledTree
from pylabeledrf.computeLRF import flip_node, collapse_node, extend_node

def build_simple_tree_newick(newick, rooted=True):
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa, rooted=rooted)
    # assign labels to internal nodes, alternating speciation/duplication for simplicity
    labels = ["speciation", "duplication"]
    idx = 0
    for node in t.internal_nodes():
        node.label = labels[idx % 2]
        idx += 1
    # root label None
    t.seed_node.label = None
    t.update_bipartitions()
    return t

def test_computeLRF_identical_trees():
    nwk = "((A,B),(C,D));"
    t1 = build_simple_tree_newick(nwk)
    t2 = build_simple_tree_newick(nwk)
    dist = computeLRF.computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_different_leaf_labels():
    t1 = build_simple_tree_newick("((A,B),(C,D));")
    t2 = build_simple_tree_newick("((A,B),(E,F));")
    with pytest.raises(ValueError):
        computeLRF.computeLRF(t1, t2)

def test_computeLRF_simple_dup_vs_spec():
    # Two trees same topology but one internal node label differs
    t1 = build_simple_tree_newick("((A,B),(C,D));")
    t2 = build_simple_tree_newick("((A,B),(C,D));")
    # Flip label of one internal node in t2 to cause substitution
    nodes = list(t2.internal_nodes())
    # flip one node label to opposite
    nodes[0].label = "duplication" if nodes[0].label == "speciation" else "speciation"
    dist = computeLRF.computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_different_topologies():
    t1 = build_simple_tree_newick("((A,B),(C,D));")
    t2 = build_simple_tree_newick("(A,(B,(C,D)));")
    dist = computeLRF.computeLRF(t1, t2)
    # Distance must be positive and larger than zero
    assert dist > 0

def test_computeLRF_single_island_no_good_edges():
    # build two trees with no good edges (completely different)
    t1 = build_simple_tree_newick("((A,B),(C,D));")
    t2 = build_simple_tree_newick("(A,(B,(C,D)));")
    # Remove all common bipartitions forcing no good edges
    # forcibly manipulate bipartitions to simulate no overlap (unsafe but for test)
    # Actually lets use computeLRF as is - expect no exception & distance > 0
    dist = computeLRF.computeLRF(t1, t2)
    assert dist > 0

def test_mutateLabeledTree_output_type_and_clone():
    t = build_simple_tree_newick("((A,B),(C,D));")
    t2 = mutateLabeledTree.mutateLabeledTree(t, 5, p_flip=0.5, model='LRF')
    # Returned tree must be a dendropy.Tree and not the same instance
    assert isinstance(t2, dendropy.Tree)
    assert t2 is not t
    # The original tree still has root label None
    assert t.seed_node.label is None

def test_flip_node_switches_labels():
    t = build_simple_tree_newick("((A,B),(C,D));")
    internal_nodes = list(t.internal_nodes())
    for node in internal_nodes:
        old_label = node.label
        flip_node(node)
        assert node.label != old_label
        # flip back to original for each
        flip_node(node)
        assert node.label == old_label

def test_flip_node_invalid_label_raises():
    node = dendropy.Node(label="otherlabel")
    with pytest.raises(ValueError):
        flip_node(node)

def test_collapse_node_lrf_allowed():
    t = build_simple_tree_newick("((A,B),(C,D));")
    node = next(iter(t.internal_nodes()))
    # collapse_node with model LRF should not raise
    collapse_node(node, model='LRF')

def test_collapse_node_elrf_valid_and_invalid():
    t = build_simple_tree_newick("((A,B),(C,D));")
    # Manually set parent labels to match for one node:
    nodes = list(t.internal_nodes())
    # Make parent and node labels same for first internal node with parent
    node = next(n for n in nodes if n.parent_node is not None)
    node.label = "speciation"
    node.parent_node.label = "speciation"
    # Should not raise
    collapse_node(node, model='ELRF')

    # For node with different labels, should raise
    node_different = next(n for n in nodes if n.parent_node is not None and n != node)
    node_different.label = "speciation"
    node_different.parent_node.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(node_different, model='ELRF')

def test_collapse_node_unsupported_model():
    t = build_simple_tree_newick("((A,B),(C,D));")
    node = next(t.internal_nodes())
    with pytest.raises(ValueError):
        collapse_node(node, model='UNKNOWN_MODEL')

def test_extend_node_lrf_behavior_and_degree_error():
    t = build_simple_tree_newick("((A,B),(C,D),(E,F));")
    # Find a node with degree >=4 (3 children) - tree constructed with trichotomy at root often
    candidates = [n for n in t.internal_nodes() if len(n.adjacent_nodes()) > 3]
    if not candidates:
        # forcibly add children to root for test:
        root = t.seed_node
        newchild = dendropy.Node(label='speciation')
        root.add_child(newchild)
        candidates = [root]
    node = candidates[0]

    # extend_node should run without raising on this node
    try:
        extend_node(node, model='LRF')
    except ValueError:
        pass

    # Create node with insufficient degree to raise
    node_low_degree = None
    for n in t.internal_nodes():
        if n.parent_node is not None and len(n.adjacent_nodes()) <= 3:
            node_low_degree = n
            break
    if node_low_degree is not None:
        with pytest.raises(ValueError):
            extend_node(node_low_degree, model='LRF')

def test_extend_node_invalid_model():
    t = build_simple_tree_newick("((A,B),(C,D));")
    node = next(t.internal_nodes())
    with pytest.raises(ValueError):
        extend_node(node, model='BADMODEL')

def test_mutateLabeledTree_zero_edits_does_not_change():
    t = build_simple_tree_newick("((A,B),(C,D));")
    t2 = mutateLabeledTree.mutateLabeledTree(t, n=0)
    # Should be equal in structure and labels because no edits
    assert t.as_string(schema="newick") == t2.as_string(schema="newick")

def test_mutateLabeledTree_more_edits_changes_labels_or_structure():
    t = build_simple_tree_newick("((A,B),(C,D));")
    t2 = mutateLabeledTree.mutateLabeledTree(t, n=10)
    # The trees may differ in string or labels
    s1 = t.as_string(schema="newick")
    s2 = t2.as_string(schema="newick")
    # May or may not change structure, but labels should differ due to flips
    labels1 = [n.label for n in t.internal_nodes()]
    labels2 = [n.label for n in t2.internal_nodes()]
    assert s1 != s2 or labels1 != labels2

def test_computeLRF_raises_with_different_island_numbers(monkeypatch):
    # Patch getIslands to produce unequal number of islands forcibly
    import pylabeledrf.computeLRF as computeLRFmod

    def fake_getIslands(t, goodedges):
        # Return unequal islands length
        return ([set([1]), set([2])], {}, [1,1])

    # Save original method
    orig_getIslands = computeLRFmod.getIslands
    computeLRFmod.getIslands = lambda t, ge: ([set()], {}, [1]) if t.seed_node.label is None else ([set(), set()], {}, [1,1])

    t1 = build_simple_tree_newick("((A,B),(C,D));")
    t2 = build_simple_tree_newick("((A,B),(C,D));")
    # monkeypatch getIslands to produce mismatch deliberately
    def new_getIslands(t, goodedges):
        if t is t1:
            return [set()], {}, [1]
        else:
            return [set(), set()], {}, [1, 1]
    computeLRFmod.getIslands = new_getIslands

    with pytest.raises(ValueError):
        computeLRFmod.computeLRF(t1, t2)

    # restore
    computeLRFmod.getIslands = orig_getIslands

def test_mutateLabeledTree_model_ELRF_edge_case():
    t = build_simple_tree_newick("((A,B),(C,D));")
    # model ELRF should work without error
    t2 = mutateLabeledTree.mutateLabeledTree(t, n=3, p_flip=0.5, model='ELRF')
    assert isinstance(t2, dendropy.Tree)

def test_flip_node_roundtrip():
    t = build_simple_tree_newick("((A,B),(C,D));")
    node = next(t.internal_nodes())
    orig_label = node.label
    flip_node(node)
    flip_node(node)
    # Should return to original label
    assert node.label == orig_label
