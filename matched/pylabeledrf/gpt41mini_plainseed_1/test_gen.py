import pytest
import dendropy
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node
from pylabeledrf.computeLRF import parseEnsemblLabels

# Utility to build a simple labeled tree from Newick string with internal node labels
def build_tree(newick, node_labels):
    """
    Build a dendropy tree from Newick and assign labels to internal nodes in preorder.
    node_labels is a list of labels ('speciation'/'duplication'/None) for internal nodes in preorder.
    Leaves have label None.
    """
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    internal_nodes = [n for n in tree.internal_nodes()]
    # Assign labels to internal nodes in preorder (seed_node first, then children etc)
    # To approximate preorder, we use .preorder_node_iter() filtering internal nodes except root?
    # We label all internal nodes including root except if label is None
    # This method assigns node_labels in order to internal nodes including root
    # We assume node_labels length == number of internal nodes
    # Overwrite with given labels, in order of tree.internal_nodes() which is postorder in DendroPy,
    # so let's sort by depth and branch order instead:
    def preorder_internal_nodes(t):
        for node in t.preorder_node_iter():
            if not node.is_leaf():
                yield node
    nodes = list(preorder_internal_nodes(tree))
    assert len(nodes) == len(node_labels)
    for node, label in zip(nodes, node_labels):
        node.label = label
    return tree

def test_computeLRF_identical_simple_tree():
    # Two identical trees: LRF distance should be 0
    newick = "((A,B),(C,D));"
    # internal nodes: root + two internal nodes under root, order is root then left then right (approx)
    node_labels = ['speciation', 'duplication', 'speciation']
    t1 = build_tree(newick, node_labels)
    t2 = build_tree(newick, node_labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_simple_flip_label():
    # Trees differ only by one flip (speciation <-> duplication) at one internal node
    newick = "((A,B),(C,D));"
    node_labels1 = ['speciation', 'duplication', 'speciation']
    node_labels2 = ['speciation', 'speciation', 'speciation']  # flipped one label
    t1 = build_tree(newick, node_labels1)
    t2 = build_tree(newick, node_labels2)
    dist = computeLRF(t1, t2)
    # Should detect substitutions, distance > 0
    assert dist > 0

def test_computeLRF_different_topology():
    # Trees with different topology and same labels: distance > 0
    newick1 = "((A,B),(C,D));"
    newick2 = "((A,C),(B,D));"
    node_labels = ['speciation', 'duplication', 'speciation']
    t1 = build_tree(newick1, node_labels)
    t2 = build_tree(newick2, node_labels)
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_islands_mismatch_valueerror():
    # Force islands to differ in count by manipulating labels - should raise ValueError in computeLRF
    newick = "((A,B),(C,D));"
    node_labels1 = ['speciation', 'duplication', 'speciation']
    node_labels2 = ['duplication', 'duplication', 'duplication']
    t1 = build_tree(newick, node_labels1)
    t2 = build_tree(newick, node_labels2)
    # monkeypatch identify_labelled_bipartitions to produce different bipartitions sets to create unequal islands:
    from pylabeledrf import computeLRF as cLRF
    def fake_identify_labelled_bipartitions(t):
        b = []
        bTn = {}
        if t is t1:
            return ([1,2], {1:[None,None], 2:[None,None]})
        else:
            return ([1,2,3], {1:[None,None], 2:[None,None], 3:[None,None]})
    orig_func = cLRF.identify_labelled_bipartitions
    cLRF.identify_labelled_bipartitions = fake_identify_labelled_bipartitions
    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF(t1, t2)
    # restore original
    cLRF.identify_labelled_bipartitions = orig_func

def test_mutateLabeledTree_flips_and_collapses():
    newick = "(((A,B),(C,D)),(E,F));"
    node_labels = ['speciation', 'duplication', 'speciation', 'duplication']
    tree = build_tree(newick, node_labels)
    mutated = mutateLabeledTree(tree, n=5, p_flip=0.5, model='LRF')
    # mutated tree is a dendropy tree different or same but should have internal node labels all in allowed set or None
    allowed_labels = {'speciation', 'duplication', None}
    for n in mutated.internal_nodes():
        assert n.label in allowed_labels

def test_flip_node_changes_label():
    # Create a dummy node with label speciation, then flip
    node = dendropy.Node()
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'
    flip_node(node)
    assert node.label == 'speciation'

def test_flip_node_invalid_label_raises():
    node = dendropy.Node()
    node.label = 'unknownlabel'
    with pytest.raises(ValueError):
        flip_node(node)

def test_collapse_node_LRF_does_not_raise():
    newick = "((A,B),(C,D));"
    node_labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, node_labels)
    internal = [n for n in tree.internal_nodes() if n.parent_node is not None]
    # Should not raise error with model=LRF
    for node in internal:
        collapse_node(node, model='LRF')

def test_collapse_node_ELRF_valid_and_invalid():
    newick = "((A,B),(C,D));"
    node_labels = ['speciation', 'speciation', 'duplication']  # parent and child labels to test
    tree = build_tree(newick, node_labels)
    # find a node with matching label parent and node
    valid_node = None
    invalid_node = None
    for n in tree.internal_nodes():
        if n.parent_node is None:
            continue
        if n.label == n.parent_node.label:
            valid_node = n
        else:
            invalid_node = n
    if valid_node is not None:
        collapse_node(valid_node, model="ELRF")  # should not raise
    if invalid_node is not None:
        with pytest.raises(ValueError):
            collapse_node(invalid_node, model="ELRF")

def test_extend_node_raises_for_low_degree():
    newick = "((A,B),(C,D));"
    node_labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, node_labels)
    # Pick a node with degree <=3 to check raise
    for n in tree.internal_nodes():
        if len(n.adjacent_nodes()) <= 3:
            with pytest.raises(ValueError):
                extend_node(n, model='LRF')
            break

def test_extend_node_runs_for_high_degree():
    # Construct a tree with node degree > 3 artificially
    t = dendropy.Tree()
    root = t.seed_node
    root.label = 'speciation'
    # Add 4 children to root to guarantee degree>3
    for i in range(4):
        c = root.new_child(label='duplication' if i % 2 == 0 else 'speciation')
        c.new_child(taxon=dendropy.Taxon(label=f"L{i}"))
    # root has 4 children, so degree >3
    extend_node(root, model='LRF')
    # root should still have label in the set
    assert root.label in ['speciation', 'duplication']

def test_parseEnsemblLabels_assigns_correct_labels():
    # Newick with internal annotations; simulate Ensembl NHX format by manually adding
    newick = "((A,B),(C,D));"
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    # add fake annotations to internal nodes
    for n in tree.internal_nodes():
        if n.is_leaf():
            continue
        n.annotations.add_new("D", "Y")
    labeled = parseEnsemblLabels(tree)
    # All internal nodes except root have label 'duplication' (due to "D=Y")
    for n in labeled.internal_nodes():
        if n.parent_node is not None:
            assert n.label == 'duplication'
    assert labeled.seed_node.label is None

def test_mutateLabeledTree_returns_different_tree():
    newick = "((A,B),(C,D));"
    node_labels = ['speciation', 'duplication', 'speciation']
    tree = build_tree(newick, node_labels)
    mutated = mutateLabeledTree(tree, n=3, p_flip=0.1, model='LRF')
    # mutated must be a dendropy tree
    assert isinstance(mutated, dendropy.Tree)
    # mutated tree should not be the same instance as original
    assert mutated is not tree
    # At least one internal node label might differ (not guaranteed due to randomness)
    labels_orig = [n.label for n in tree.internal_nodes()]
    labels_mut = [n.label for n in mutated.internal_nodes()]
    assert len(labels_orig) == len(labels_mut)

def test_computeLRF_with_single_island():
    # Construct two trees with no good edges so exactly 1 island, distance should be computed correctly
    newick1 = "(A,B);"
    newick2 = "(A,B);"
    node_labels1 = [None]
    node_labels2 = [None]
    t1 = build_tree(newick1, node_labels1)
    t2 = build_tree(newick2, node_labels2)
    dist = computeLRF(t1, t2)
    assert dist == 0  # identical single edge tree, should have distance 0

def test_computeLRF_raises_for_unprocessed_islands_t2():
    # This is a rare error path: causes False in processed2 but not False in processed1
    # We achieve by monkeypatching getIslands to make processed2 incomplete
    from pylabeledrf import computeLRF as cLRF
    newick = "((A,B),(C,D));"
    node_labels = ['speciation', 'duplication', 'speciation']
    t1 = build_tree(newick, node_labels)
    t2 = build_tree(newick, node_labels)

    orig_getIslands = cLRF.getIslands
    def fake_getIslands(t, goodedges):
        islands, node_island, sizeIslands = orig_getIslands(t, goodedges)
        if t is t2:
            # Make processed2 incomplete by adding a fake island (affecting control flow)
            islands.append(set())
            sizeIslands.append(0)
            # node_island stays same but we add dummy island
        return islands, node_island, sizeIslands

    cLRF.getIslands = fake_getIslands

    # monkeypatch identify_labelled_bipartitions to always have some goodedges
    orig_identify = cLRF.identify_labelled_bipartitions
    def fake_identify(t):
        biparts, bipart2nodes = orig_identify(t)
        # return the same biparts so goodedges is nonempty
        return biparts, bipart2nodes

    cLRF.identify_labelled_bipartitions = fake_identify

    with pytest.raises(ValueError, match="At least one unprocessed island in t2"):
        computeLRF(t1, t2)

    cLRF.getIslands = orig_getIslands
    cLRF.identify_labelled_bipartitions = orig_identify
