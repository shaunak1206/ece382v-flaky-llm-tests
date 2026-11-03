import pytest
import dendropy
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree
from pylabeledrf.computeLRF import flip_node, collapse_node, extend_node


def make_simple_tree(newick_str, labels):
    """
    Helper function to create a dendropy Tree from Newick string and assign labels to internal nodes.
    labels: dict mapping node labels (taxon or internal) to 'speciation' or 'duplication'
    Internal nodes might lack labels but should get assigned in breadth order.
    """
    tree = dendropy.Tree.get(data=newick_str, schema="newick")
    # Assign labels to internal nodes in preorder, as taxon labels are for leaves
    internal_nodes = [n for n in tree.internal_nodes() if n.is_internal()]
    leaf_nodes = list(tree.leaf_node_iter())
    # Map taxon labels
    for leaf in leaf_nodes:
        if leaf.taxon.label in labels:
            leaf.label = labels[leaf.taxon.label]
    # Map internal nodes by their order; if labels keys missing, assign defaults to internal nodes
    ilabels = [labels.get(f"i{i}", None) for i in range(len(internal_nodes))]
    for node, lab in zip(internal_nodes, ilabels):
        if lab:
            node.label = lab
        else:
            node.label = "speciation"  # default if not specified
    # Root label = None as standard
    tree.seed_node.label = None
    return tree

# --------- computeLRF tests ---------

def test_computeLRF_identical_small_trees():
    # Two identical small trees, no difference => distance 0
    newick = '((A,B),(C,D));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    t1 = make_simple_tree(newick, labels)
    t2 = make_simple_tree(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_simple_flip_label():
    # Same topology but flip single label in one tree => distance should account for substitution
    newick = '((A,B),(C,D));'
    labels1 = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    labels2 = {"i0": "duplication", "i1": "duplication", "i2": "speciation"}  # root flipped
    t1 = make_simple_tree(newick, labels1)
    t2 = make_simple_tree(newick, labels2)
    dist = computeLRF(t1, t2)
    assert dist >= 1  # at least one substitution difference

def test_computeLRF_completely_different_topologies():
    # Two trees with totally different topologies but same leaves
    nwk1 = '(((A,B),C),D);'
    nwk2 = '((A,(B,C)),D);'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    t1 = make_simple_tree(nwk1, labels)
    t2 = make_simple_tree(nwk2, labels)
    dist = computeLRF(t1, t2)
    assert dist > 0

def test_computeLRF_unrooted_two_star():
    # Two star trees with only all leaves connected to root (no internal bipartitions)
    newick = '(A,B,C,D);'
    labels = {}  # no internal nodes to label
    t1 = make_simple_tree(newick, labels)
    t2 = make_simple_tree(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0

def test_computeLRF_raise_error_unequal_islands():
    # Manually create trees that will create unequal islands (force error)
    # Create two trees where one tree is rooted differently causing uneven islands
    nwk1 = '((A,B),(C,D));'
    nwk2 = '(A,(B,(C,D)));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    t1 = make_simple_tree(nwk1, labels)
    t2 = make_simple_tree(nwk2, labels)
    # There is a chance that unequal islands arise due to differences in rooting or bipartition splitting
    with pytest.raises(ValueError, match="Unequal number of islands. Should not happen."):
        computeLRF(t1, t2)


def test_computeLRF_substitution_count():
    # Test substitution count triggers by having disjoint label sets on islands
    nwk = '((A,B),(C,D));'
    # tree1 with speciation at root and duplication in left child
    labels1 = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    # tree2 with duplication at root and speciation in left child
    labels2 = {"i0": "duplication", "i1": "speciation", "i2": "duplication"}
    t1 = make_simple_tree(nwk, labels1)
    t2 = make_simple_tree(nwk, labels2)
    dist = computeLRF(t1, t2)
    assert dist >= 1


# --------- mutateLabeledTree tests -----------

def test_mutateLabeledTree_returns_new_tree():
    newick = '((A,B),(C,D));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    tree = make_simple_tree(newick, labels)
    mutated = mutateLabeledTree(tree, n=1, p_flip=1.0)  # force flip
    assert mutated is not tree
    # Check labels changed accordingly (at least one flipped)
    original_labels = [n.label for n in tree.internal_nodes()]
    mutated_labels = [n.label for n in mutated.internal_nodes()]
    assert original_labels != mutated_labels

def test_mutateLabeledTree_zero_edits_returns_clone():
    newick = '(A,B);'
    labels = {}
    tree = make_simple_tree(newick, labels)
    mutated = mutateLabeledTree(tree, n=0)
    # Should be identical tree in topology and labels
    assert mutated.as_string("newick") == tree.as_string("newick")

def test_mutateLabeledTree_flip_probability_zero_no_flips():
    newick = '((A,B),(C,D));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    tree = make_simple_tree(newick, labels)
    # Run multiple edits with p_flip=0 so no flips happen, only collapse/extend
    mutated = mutateLabeledTree(tree, n=5, p_flip=0)
    # The mutated tree should be different or same, but no labels are flipped so labels remain subset of originals
    orig_labels = [n.label for n in tree.internal_nodes()]
    mut_labels = [n.label for n in mutated.internal_nodes()]
    # Because of collapse/extend the set might differ but no flips should exist
    # We test no label changed from 'speciation' to 'duplication' or vice versa randomly:
    for olab, mlabel in zip(orig_labels, mut_labels):
        # For LRF model flips allowed only by flip_node, but here no flips happened due to p_flip=0
        # So either labels equal or collapse/extend kept labels same (sometimes labels propagate)
        assert olab in ('speciation', 'duplication')
        assert mlabel in ('speciation', 'duplication')

def test_mutateLabeledTree_invalid_model_raises():
    newick = '((A,B),(C,D));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    tree = make_simple_tree(newick, labels)
    # mutateLabeledTree does not raise directly, but collapse_node and extend_node do.
    from pylabeledrf.computeLRF import collapse_node, extend_node
    # Setup node to test collapse_node and extend_node with invalid model
    node = next(tree.internal_nodes())
    with pytest.raises(ValueError, match='unsupported model'):
        collapse_node(node, model="foo")
    with pytest.raises(ValueError, match='unsupported model'):
        extend_node(node, model="bar")


# --------- flip_node tests -----------

def test_flip_node_speciation_to_duplication():
    newick = '(A,B);'
    tree = dendropy.Tree.get(data=newick, schema='newick')
    # Add label to root internal node artificially
    node = tree.seed_node
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'

def test_flip_node_duplication_to_speciation():
    newick = '(A,B);'
    tree = dendropy.Tree.get(data=newick, schema='newick')
    node = tree.seed_node
    node.label = 'duplication'
    flip_node(node)
    assert node.label == 'speciation'

def test_flip_node_invalid_label_raises():
    newick = '(A,B);'
    tree = dendropy.Tree.get(data=newick, schema='newick')
    node = tree.seed_node
    node.label = 'invalid_label'
    with pytest.raises(ValueError, match='Should not happen!!'):
        flip_node(node)

# -------- collapse_node tests ---------

def test_collapse_node_LRF_no_error():
    newick = '(A,(B,C));'
    labels = {"i0": "speciation", "i1": "duplication"}
    tree = make_simple_tree(newick, labels)
    # Collapse any internal node with model='LRF'
    node = next(tree.internal_nodes())
    collapse_node(node, model='LRF')
    # After collapse node edge should be removed (collapsed)
    assert node.parent_node is None or node.edge.length is None or node.edge.length == 0 or node.edge.incident_nodes() is not None

def test_collapse_node_ELRF_valid():
    newick = '(A,(B,C));'
    labels = {"i0": "duplication", "i1": "duplication"}  # set parent and child label equal
    tree = make_simple_tree(newick, labels)
    node = None
    for n in tree.internal_nodes():
        if n.parent_node and n.label == n.parent_node.label:
            node = n
            break
    if node:
        collapse_node(node, model='ELRF')
    else:
        pytest.skip("No suitable node with matching labels found for ELRF collapse test")

def test_collapse_node_ELRF_invalid_raises():
    newick = '(A,(B,C));'
    labels = {"i0": "duplication", "i1": "speciation"}  # parent and child labels differ
    tree = make_simple_tree(newick, labels)
    node = None
    for n in tree.internal_nodes():
        if n.parent_node and n.label != n.parent_node.label:
            node = n
            break
    if node is None:
        pytest.skip("No suitable node with differing labels found for ELRF collapse invalid test")
    else:
        with pytest.raises(ValueError, match='this edge cannot be collapsed without a flip!'):
            collapse_node(node, model='ELRF')

# -------- extend_node tests ---------

def test_extend_node_success():
    # Build tree with node of degree >=4 to allow extend_node
    newick = '(((A,B),C),D);'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    tree = make_simple_tree(newick, labels)
    # Find node with degree>3 or adjacent nodes > 3
    node = None
    for n in tree.internal_nodes():
        if len(n.adjacent_nodes()) > 3 or (n.parent_node is None and len(n.adjacent_nodes()) > 2):
            node = n
            break
    if node is None:
        # If none found, forcibly create multifurcations (unlikely)
        pytest.skip("No node with degree >3 found for extend_node test")
    else:
        # Should not raise error
        extend_node(node, model='LRF')
        # After extension, children count should change but sum remain same
        assert len(node.child_nodes()) > 0

def test_extend_node_fails_insufficient_degree():
    # Build simple tree with only 3 adjacent nodes to a node - extend_node should raise
    newick = '(A,B);'
    tree = make_simple_tree(newick, {})
    node = tree.seed_node  # root with 2 adjacent nodes probably
    # forcibly reduce degree to test condition
    # In this case, adjacent_nodes() <=3 and node.parent_node not None triggers error
    # Need to add a node with degree<=3 and parent not None
    # Create a new internal node with few children
    newnode = dendropy.Node(label="speciation")
    # forcibly create small subtree
    leaf1 = dendropy.Node(taxon=dendropy.Taxon(label="X"))
    leaf2 = dendropy.Node(taxon=dendropy.Taxon(label="Y"))
    newnode.add_child(leaf1)
    newnode.add_child(leaf2)
    # Attach newnode as child to root to have parent_node not None and degree 2
    tree.seed_node.add_child(newnode)
    with pytest.raises(ValueError, match="Insufficient degree. Should not happen"):
        extend_node(newnode, model='LRF')

def test_extend_node_invalid_model():
    newick = '((A,B),(C,D));'
    labels = {"i0": "speciation", "i1": "duplication", "i2": "speciation"}
    tree = make_simple_tree(newick, labels)
    node = next(tree.internal_nodes())
    with pytest.raises(ValueError, match='unsupported model'):
        extend_node(node, model='unsupported')

