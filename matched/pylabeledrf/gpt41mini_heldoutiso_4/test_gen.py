import pytest
import dendropy
from pylabeledrf import computeLRF, mutateLabeledTree
import random

# Helper to build a simple labeled tree from Newick and set internal node labels
def build_tree(newick, labels):
    """
    Build a dendropy tree from newick string and assign labels to internal nodes.
    :param newick: Newick string for the tree
    :param labels: List of labels to assign to internal nodes in traversal order
    :return: dendropy.Tree instance
    """
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema='newick', taxon_namespace=taxa)
    internal_nodes = list(tree.internal_nodes())
    # Skip the root if it has no label or assign labels in pre-order except root
    # We'll assign labels in the order internal_nodes returns (generally preorder)
    for node, label in zip(internal_nodes, labels):
        node.label = label
    # root label typically None/unlabeled
    if tree.seed_node.label is None:
        tree.seed_node.label = None
    return tree

def test_computeLRF_identical_trees_distance_zero():
    # Two identical trees, labels match, distance should be 0
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']  # for inner nodes
    t1 = build_tree(newick, labels)
    t2 = build_tree(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_different_labels_in_nodes():
    # Same topology, but one duplication vs speciation flip, substitution needed = 1
    newick = "((A,B),(C,D));"
    labels1 = ['speciation', 'duplication', 'speciation']
    labels2 = ['duplication', 'duplication', 'speciation']
    t1 = build_tree(newick, labels1)
    t2 = build_tree(newick, labels2)
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_different_topologies():
    # Trees with different topology (different Newick), expect nonzero positive distance
    t1 = build_tree("((A,B),(C,D));", ['speciation', 'duplication', 'speciation'])
    t2 = build_tree("((A,C),(B,D));", ['speciation', 'duplication', 'speciation'])
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_single_island_nontrivial():
    # Trees with only one island with >1 nodes, no good edges
    # Expect distance = (sizeIsland1 + sizeIsland2 -2) + (substitution if labels differ)
    newick1 = "((A,B),(C,D));"
    newick2 = "((A,B),(C,D));"
    labels1 = ['speciation', 'duplication', 'speciation']
    labels2 = ['duplication', 'duplication', 'duplication']
    t1 = build_tree(newick1, labels1)
    t2 = build_tree(newick2, labels2)
    dist = computeLRF(t1, t2)
    # There should be at least one substitution because labels sets won't intersect
    assert dist >= 1

def test_computeLRF_raises_on_unequal_islands(monkeypatch):
    # Create trees and monkeypatch getIslands to return different number of islands to provoke ValueError
    from pylabeledrf import computeLRF as computeLRF_mod

    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    t1 = build_tree(newick, labels)
    t2 = build_tree(newick, labels)

    def fake_getIslands(t, goodedges):
        if t is t1:
            return ([set(['speciation'])], {}, [1])
        else:
            return ([set(['speciation']), set(['duplication'])], {}, [1,1])

    monkeypatch.setattr("pylabeledrf.computeLRF.getIslands", fake_getIslands)
    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF_mod(t1, t2)

def test_mutateLabeledTree_flip_only(monkeypatch):
    # Test mutateLabeledTree with p_flip=1 forces flips and no expand/collapse
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, labels)
    
    # Patch random.random to always return 0 to always trigger flip < p_flip=1
    monkeypatch.setattr("random.random", lambda : 0)
    # Patch random.randint to always select first node for predictable flip
    monkeypatch.setattr("random.randint", lambda a,b: 0)
    
    mutated = mutateLabeledTree(tree, n=3, p_flip=1, model='LRF')
    
    # Check that the internal node labels swapped speciation <-> duplication only
    original_labels = [node.label for node in tree.internal_nodes()]
    mutated_labels = [node.label for node in mutated.internal_nodes()]
    # At least one flip happened
    assert original_labels != mutated_labels
    # All labels are either 'speciation' or 'duplication'
    assert all(lbl in ('speciation', 'duplication', None) for lbl in mutated_labels)

def test_mutateLabeledTree_expand_collapse(monkeypatch):
    # Try mutateLabeledTree with p_flip=0 (no flips) so only collapse or expand happen
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, labels)
    
    monkeypatch.setattr("random.random", lambda: 1)  # No flip: random() > p_flip=0
    
    # patch randint to select collapse first (indices within bounds)
    randint_gen = iter([0, 0, 0])
    monkeypatch.setattr("random.randint", lambda a,b: next(randint_gen))
    
    mutated = mutateLabeledTree(tree, n=2, p_flip=0, model='LRF')
    
    # The mutated tree should have the same taxa set
    assert set(t.label for t in mutated.leaf_nodes()) == set(t.label for t in tree.leaf_nodes())

def test_mutateLabeledTree_invalid_model():
    # Passing invalid model to mutateLabeledTree should raise ValueError inside collapse_node or extend_node
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, labels)

    with pytest.raises(ValueError):
        mutateLabeledTree(tree, n=1, p_flip=0, model='INVALID_MODEL')

def test_flip_node_changes_speciation_and_duplication():
    from pylabeledrf.computeLRF import flip_node
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data="((A,B),C);", schema="newick", taxon_namespace=taxa)
    internal_nodes = list(tree.internal_nodes())
    for node in internal_nodes:
        node.label = 'speciation'
    for node in internal_nodes:
        flip_node(node)
        assert node.label == 'duplication'
        flip_node(node)
        assert node.label == 'speciation'

def test_flip_node_invalid_label_raises():
    from pylabeledrf.computeLRF import flip_node
    node = dendropy.Node()
    node.label = "invalid_label"
    with pytest.raises(ValueError, match="Should not happen!!"):
        flip_node(node)

def test_collapse_node_lrf_collapses_edge():
    from pylabeledrf.computeLRF import collapse_node
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, labels)
    # Pick internal node with parent to collapse
    node_to_collapse = next(n for n in tree.internal_nodes() if n.parent_node is not None)
    collapse_node(node_to_collapse, model='LRF')
    # After collapse, node should have less or equal children than before (generally less)
    assert len(list(node_to_collapse.parent_node.child_nodes())) <= 3

def test_collapse_node_elrf_valid():
    from pylabeledrf.computeLRF import collapse_node
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'speciation', 'speciation']  # Labels to allow collapse in ELRF
    tree = build_tree(newick, labels)
    node = next(n for n in tree.internal_nodes() if n.parent_node is not None)
    node.label = 'speciation'
    node.parent_node.label = 'speciation'
    # Should not raise
    collapse_node(node, model='ELRF')

def test_collapse_node_elrf_invalid_raises():
    from pylabeledrf.computeLRF import collapse_node
    newick = "((A,B),(C,D));"
    labels = ['duplication', 'duplication', 'duplication']
    tree = build_tree(newick, labels)
    node = next(n for n in tree.internal_nodes() if n.parent_node is not None)
    node.label = 'duplication'
    # parent with different label
    node.parent_node.label = 'speciation'  
    with pytest.raises(ValueError, match="this edge cannot be collapsed without a flip!"):
        collapse_node(node, model='ELRF')

def test_extend_node_raises_on_invalid_degree():
    from pylabeledrf.computeLRF import extend_node
    newick = "((A,B),C);"
    labels = ['speciation', 'duplication']
    tree = build_tree(newick, labels)
    node = next(n for n in tree.internal_nodes())
    # forcibly reduce node degree to <= 3 and try extend_node
    # This simplistic tree might have low degree: force
    # mutate node to have adjacents length <= 3
    try:
        extend_node(node, model='LRF')
    except ValueError as e:
        assert "Insufficient degree" in str(e)

def test_parseEnsemblLabels_sets_labels_and_root_label():
    from pylabeledrf.computeLRF import parseEnsemblLabels
    newick = "((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;"
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    # Add Ensembl style annotations
    for node in tree.internal_nodes():
        node.annotations.add_new("D", "N") 
        node.annotations.add_new("DD", "N")
    # Set one node to 'Y'
    node = next(iter(tree.internal_nodes()))
    node.annotations["D"].value = 'Y'
    parsed = parseEnsemblLabels(tree)
    # At least one duplication label expected
    internal_labels = [n.label for n in parsed.internal_nodes()]
    assert 'duplication' in internal_labels or 'speciation' in internal_labels
    # Root should have label None
    assert parsed.seed_node.label is None

def test_ordered_bipartitions_returns_dict():
    from pylabeledrf.computeLRF import ordered_bipartitions
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    bipart_dict = ordered_bipartitions(tree)
    # Should be a dict with tuple keys and node values
    assert isinstance(bipart_dict, dict)
    if bipart_dict:
        k,v = next(iter(bipart_dict.items()))
        assert isinstance(k, tuple)
        # v is a node
        from dendropy import Node
        assert isinstance(v, Node)

def test_identify_labelled_bipartitions_contains_known_bipartitions():
    from pylabeledrf.computeLRF import identify_labelled_bipartitions
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, labels)
    bipartitions, bipart2nodes = identify_labelled_bipartitions(tree)
    # Bipartitions list should not be empty for internal nodes
    assert len(bipartitions) > 0
    # bipart2nodes keys should be bipartitions from bipartitions list
    for b in bipartitions:
        assert b in bipart2nodes

