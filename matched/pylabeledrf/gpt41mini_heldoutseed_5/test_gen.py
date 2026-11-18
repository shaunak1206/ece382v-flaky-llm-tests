import pytest
import dendropy
from pylabeledrf import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node

# Utilities for test setup

def tree_from_newick_label(newick, labels):
    """
    Given a Newick string and a dict mapping node labels or node indices (int) to 
    'speciation' or 'duplication', assign labels to internal nodes in tree.
    Leaves get no labels (None).
    """
    tree = dendropy.Tree.get(data=newick, schema="newick")
    # label nodes in preorder traversal (skipping leaves)
    internal_nodes = [n for n in tree.internal_nodes() if n != tree.seed_node]
    for i, node in enumerate(internal_nodes):
        if isinstance(labels, dict) and i in labels:
            node.label = labels[i]
        elif isinstance(labels, dict) and node.taxon is not None and node.taxon.label in labels:
            node.label = labels[node.taxon.label]
        else:
            # default label if none provided
            node.label = 'speciation'
    # root label is None
    tree.seed_node.label = None
    return tree

def label_all_internal_nodes(tree, label='speciation'):
    for n in tree.internal_nodes():
        n.label = label
    tree.seed_node.label = None
    return tree


# --------------- Tests for computeLRF -------------------

def test_computeLRF_identical_simple_tree():
    # two identical trees, same labels, distance should be zero
    newick = "((A,B),(C,D));"
    labels = {0:'speciation',1:'speciation'}
    t1 = tree_from_newick_label(newick, labels)
    t2 = tree_from_newick_label(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_different_labels():
    # same topology but one node label differs
    newick = "((A,B),(C,D));"
    labels1 = {0:'speciation', 1:'speciation'}
    labels2 = {0:'duplication', 1:'speciation'}
    t1 = tree_from_newick_label(newick, labels1)
    t2 = tree_from_newick_label(newick, labels2)
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_different_topology():
    # Trees with different topology, distance > 0
    newick1 = "((A,B),(C,D));"
    newick2 = "((A,C),(B,D));"
    labels = {0:'speciation',1:'speciation'}
    t1 = tree_from_newick_label(newick1, labels)
    t2 = tree_from_newick_label(newick2, labels)
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_unequal_islands_Error():
    # Construct weird trees to trigger unequal islands and ValueError
    # This is tough to create manually, so forcibly patch getIslands to return unequal count
    import pylabeledrf.computeLRF as clrf
    
    newick = "((A,B),(C,D));"
    labels = {0:'speciation',1:'speciation'}
    t1 = tree_from_newick_label(newick, labels)
    t2 = tree_from_newick_label(newick, labels)
    
    def fake_getIslands(t, goodedges):
        if t == t1:
            return [{1}, {}, {}], {}, [1,2,3]
        else:
            return [{1}], {}, [1]
    
    original_getIslands = clrf.getIslands
    clrf.getIslands = fake_getIslands
    
    with pytest.raises(ValueError):
        computeLRF(t1,t2)
    clrf.getIslands = original_getIslands

def test_computeLRF_label_flip_substitution():
    # Two identical trees, but node labels flipped on one island => substitution counted
    newick = "((A,B),(C,D));"
    # assign all speciation for t1
    labels1 = {0:'speciation',1:'speciation'}
    # assign duplication on one island for t2
    labels2 = {0:'duplication',1:'speciation'}
    t1 = tree_from_newick_label(newick, labels1)
    t2 = tree_from_newick_label(newick, labels2)
    
    dist = computeLRF(t1,t2)
    # Expect at least substitution counted as 1
    assert dist >= 1

def test_computeLRF_no_good_edges_single_island():
    # Construct two trees with no shared bipartitions -> only one island each
    # so that False in processed1 evaluates True branch
    newick1 = "(A,B,C,D);"
    newick2 = "(A,B,C,D);"
    t1 = dendropy.Tree.get_from_path(None, 'newick', data=newick1)
    t2 = dendropy.Tree.get_from_path(None, 'newick', data=newick2)
    label_all_internal_nodes(t1, 'speciation')
    label_all_internal_nodes(t2, 'duplication')
    
    # forcibly patch identify_labelled_bipartitions to return empty goodedges
    import pylabeledrf.computeLRF as clrf
    orig_ident = clrf.identify_labelled_bipartitions
    def fake_ident(t):
        return [],{}
    clrf.identify_labelled_bipartitions = fake_ident

    dist = computeLRF(t1,t2)
    assert dist >= 0

    clrf.identify_labelled_bipartitions = orig_ident

def test_computeLRF_processed2_unprocessed_Error():
    # Patch processed2 to contain False after loop to trigger error
    import pylabeledrf.computeLRF as clrf
    newick = "((A,B),(C,D));"
    labels = {0:'speciation',1:'speciation'}
    t1 = tree_from_newick_label(newick, labels)
    t2 = tree_from_newick_label(newick, labels)
    
    # Patch identify_labelled_bipartitions to produce a goodedge
    orig_ident = clrf.identify_labelled_bipartitions
    def fake_ident(t):
        # create a fake bipartition object that can be a key with some child and parent nodes
        class FakeBipart:
            leafset_bitmask = 7
            def __hash__(self):
                return 42
            def __eq__(self, other):
                return True
        fb = FakeBipart()
        node = next(n for n in t.internal_nodes() if n.label is not None)
        return [fb], {fb:[(node,node.parent_node) if node.parent_node is not None else (node,node) for _ in range(2)]}
    
    clrf.identify_labelled_bipartitions = fake_ident
    
    # Patch getIslands to yield multiple islands
    orig_islands = clrf.getIslands
    def fake_getIslands(t, goodedges):
        islands = [set(['speciation']), set(['duplication'])]
        node_island = {}
        size = [1,1]
        # dummy map so that processed triggered incorrectly
        return islands, node_island, size
    clrf.getIslands = fake_getIslands

    with pytest.raises(ValueError):
        computeLRF(t1,t2)
    clrf.identify_labelled_bipartitions = orig_ident
    clrf.getIslands = orig_islands

def test_mutateLabeledTree_flip_only():
    # patch random.random to be always less than p_flip to flip labels only
    import random as pyrandom

    original_random = pyrandom.random
    original_randint = pyrandom.randint

    calls = {"flip":0}

    def fake_random():
        return 0.1  # Always cause flip branch in mutateLabeledTree

    def fake_randint(a,b):
        # always pick first node available
        return 0

    pyrandom.random = fake_random
    pyrandom.randint = fake_randint
    
    newick = "((A,B),(C,D));"
    labels = {0:'speciation', 1:'duplication'}
    t = tree_from_newick_label(newick, labels)
    mtree = mutateLabeledTree(t, n=5, p_flip=0.3, model='LRF')

    # Some node labels should be flipped (some changed)
    original_labels = set(n.label for n in t.internal_nodes())
    new_labels = set(n.label for n in mtree.internal_nodes())
    # There must be some difference
    assert new_labels != original_labels

    pyrandom.random = original_random
    pyrandom.randint = original_randint

def test_mutateLabeledTree_collapse_and_extend_mixed():
    # patch random.random to be more than p_flip to test collapse/extend branch
    import random as pyrandom

    original_random = pyrandom.random
    original_randint = pyrandom.randint
    original_shuffle = pyrandom.shuffle

    def fake_random():
        return 0.9  # high to choose collapse/extend branch

    # Cycle through choosing collapse first, then extend next call
    choices = [0, 100]  # 0 triggers collapse, else extend
    iter_choices = iter(choices * 10)
    def fake_randint(a,b):
        return next(iter_choices)

    def fake_shuffle(x):
        # don't reorder for determinism
        pass

    pyrandom.random = fake_random
    pyrandom.randint = fake_randint
    pyrandom.shuffle = fake_shuffle

    newick = "(((A,B),C),D);"
    # create a node with 3+ children to enable extend_node calls
    tree = dendropy.Tree.get(data=newick, schema='newick')
    # label all internal nodes as speciation
    for n in tree.internal_nodes():
        n.label = 'speciation'
    tree.seed_node.label = None

    mutated = mutateLabeledTree(tree, n=2, p_flip=0.3, model='LRF')

    # Should still be a DendroPy tree and have same leaf set
    orig_leafset = {l.taxon.label for l in tree.leaf_nodes()}
    mut_leafset = {l.taxon.label for l in mutated.leaf_nodes()}
    assert orig_leafset == mut_leafset

    pyrandom.random = original_random
    pyrandom.randint = original_randint
    pyrandom.shuffle = original_shuffle

def test_collapse_node_LRF_and_ELRF_behavior():
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    # Label all nodes speciation
    for n in tree.internal_nodes():
        n.label = "speciation"
    tree.seed_node.label = None

    node = next(n for n in tree.internal_nodes() if n.parent_node is not None)

    # In LRF, collapse_node should always succeed
    collapse_node(node, model='LRF')
    # In ELRF, if label equals parent's label, collapse allowed, else exception
    node2 = next(n for n in tree.internal_nodes() if n != node and n.parent_node is not None)
    node2.label = node2.parent_node.label
    collapse_node(node2, model='ELRF')
    
    node3 = next(n for n in tree.internal_nodes() if n.label != n.parent_node.label and n.parent_node is not None)
    with pytest.raises(ValueError):
        collapse_node(node3, model='ELRF')

def test_flip_node_correct_label_flip():
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    for n in tree.internal_nodes():
        n.label = "speciation"
    tree.seed_node.label = None
    node = next(tree.internal_nodes())

    original_label = node.label
    flip_node(node)
    assert node.label != original_label
    # flip back
    flip_node(node)
    assert node.label == original_label

def test_flip_node_invalid_label_raises():
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    for n in tree.internal_nodes():
        n.label = None
    tree.seed_node.label = None
    node = next(tree.internal_nodes())
    with pytest.raises(ValueError):
        flip_node(node)

def test_extend_node_invalid_degree_raises():
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    # Select a node with degree <=3
    node = next(n for n in tree.internal_nodes() if len(n.adjacent_nodes()) <= 3)
    node.label = 'speciation'
    tree.seed_node.label = None
    with pytest.raises(ValueError):
        extend_node(node, model='LRF')
    with pytest.raises(ValueError):
        extend_node(node, model='ELRF')

def test_extend_node_modifies_tree_structure():
    newick = "((((A,B),C),D),E);"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    # Ensure there is at least one internal node with degree > 3
    for n in tree.internal_nodes():
        n.label = 'speciation'
    tree.seed_node.label = None

    # find a node with degree > 3
    candidates = [n for n in tree.internal_nodes() if len(n.adjacent_nodes()) > 3]
    if not candidates:
        pytest.skip("No internal node with degree >3 for extend_node test")
    node = candidates[0]

    # count original children
    orig_children = list(node.child_nodes())
    extend_node(node, model='LRF')
    # after extension, node should have more children than before or structure changed
    new_children = list(node.child_nodes())
    assert len(new_children) >= 2
    # The children moved to a new node, so the new node added is one of the children
    new_nodes_labels = [child.label for child in new_children]
    assert node.label in new_nodes_labels or any(n.label == node.label for n in new_children)


def test_computeLRF_handle_non_rooted_tree_with_label():
    # Construct unrooted tree with root node degree 2 and label
    newick = "(A,B,C);"
    tree = dendropy.Tree.get(data=newick, schema='newick', rooting='force-unrooted')
    for n in tree.internal_nodes():
        n.label = 'speciation'
    tree.seed_node.label = 'duplication'
    # Should not raise warning/error during getIslands call
    from pylabeledrf.computeLRF import getIslands
    goodedges = set()
    islands, node_island, size = getIslands(tree, goodedges)
    assert isinstance(islands, list)
    assert isinstance(size, list)

def test_identify_labelled_bipartitions_output_valid():
    newick = "((A,B),(C,D));"
    tree = dendropy.Tree.get(data=newick, schema='newick')
    for n in tree.internal_nodes():
        n.label = "speciation"
    biparts, bip2nodes = computeLRF.identify_labelled_bipartitions(tree)
    # biparts is list of bipartitions; bip2nodes dict keys match biparts
    assert isinstance(biparts, list)
    assert all(b in bip2nodes for b in biparts)

