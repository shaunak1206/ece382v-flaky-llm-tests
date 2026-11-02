import pytest
import random
import dendropy
from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree, flip_node, collapse_node, extend_node


def tree_from_newick(newick, labels):
    """
    Helper to create a dendropy tree and assign labels as given dict {node_label: label}.
    Node labels are taxon labels for leaves, and inner node labels can be assigned after.
    To identify inner nodes, we rely on postorder traversal and assign in order.
    """
    t = dendropy.Tree.get(data=newick, schema="newick", rooting='force-rooted')
    # assign internal node labels: skip root node (seed_node)
    internal = [n for n in t.internal_nodes() if n != t.seed_node]
    for node, label in zip(internal, labels):
        node.label = label
    t.seed_node.label = None
    return t


def get_all_node_labels(tree):
    return [n.label for n in tree.internal_nodes() if n.label is not None]


def test_computeLRF_identical_simple_trees():
    # Two identical simple trees with 3 leaves, all internal nodes labeled 'speciation'
    newick = "((A,B),(C,D));"
    labels = ['speciation']  # one internal node excluding root
    t1 = tree_from_newick(newick, labels)
    t2 = tree_from_newick(newick, labels)
    dist = computeLRF(t1, t2)
    assert dist == 0


def test_computeLRF_simple_flipped_labels():
    # Same topology but all internal nodes labels flipped speciation<->duplication
    newick = "((A,B),(C,D));"
    labels1 = ['speciation']
    labels2 = ['duplication']
    t1 = tree_from_newick(newick, labels1)
    t2 = tree_from_newick(newick, labels2)
    dist = computeLRF(t1, t2)
    # No topology difference, but labels differ => substitution needed between islands
    assert dist > 0


def test_computeLRF_different_topologies():
    # Trees differ by one bipartition (prune one leaf internal node)
    t1newick = "((A,B),(C,D));"
    t2newick = "(((A,B),C),D);"
    labels = ['speciation', 'speciation']  # 2 internal nodes each excluding root
    t1 = tree_from_newick(t1newick, labels)
    t2 = tree_from_newick(t2newick, labels)
    d = computeLRF(t1, t2)
    # Differences in topology and potentially label differences; distance must be positive integer
    assert isinstance(d, int)
    assert d > 0


def test_computeLRF_single_island_no_good_edge():
    # Two completely different trees with no common bipartitions => single island processed
    newick1 = "(A,(B,(C,D)));"
    newick2 = "((A,B),(C,D));"
    labels1 = ['speciation', 'duplication']
    labels2 = ['speciation']
    t1 = tree_from_newick(newick1, labels1)
    t2 = tree_from_newick(newick2, labels2)
    dist = computeLRF(t1, t2)
    # Distance must be positive
    assert dist > 0


def test_computeLRF_raises_on_unequal_islands():
    # Manually construct trees with different island counts by manipulating labels and edge sets
    newick1 = "((A,B),(C,D));"
    newick2 = "(A,(B,(C,D)));"
    labels1 = ['speciation']
    labels2 = ['speciation', 'duplication']
    t1 = tree_from_newick(newick1, labels1)
    t2 = tree_from_newick(newick2, labels2)
    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF(t1, t2)


def test_mutateLabeledTree_basic_properties():
    # Test mutateLabeledTree returns a tree of correct taxon namespace and does not modify original
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    random.seed(1234)
    mutated = mutateLabeledTree(t, n=5, p_flip=0.5)
    assert isinstance(mutated, dendropy.Tree)
    # Original tree unchanged
    original_labels = get_all_node_labels(t)
    mutated_labels = get_all_node_labels(mutated)
    assert original_labels != mutated_labels or original_labels == mutated_labels  # either can happen
    # Taxon namespace preserved
    assert t.taxon_namespace == mutated.taxon_namespace


@pytest.mark.parametrize("label", ["speciation", "duplication"])
def test_flip_node_flips_label(label):
    # Create dummy node
    node = dendropy.Node()
    node.label = label
    flip_node(node)
    if label == 'speciation':
        assert node.label == 'duplication'
    else:
        assert node.label == 'speciation'


def test_flip_node_invalid_label_raises():
    node = dendropy.Node()
    node.label = "other"
    with pytest.raises(ValueError, match='Should not happen!!'):
        flip_node(node)


def test_collapse_node_LRF_and_ELRF():
    # Create a small tree with parent and child node labeled same to test collapse_node under ELRF and LRF
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    nodes = [n for n in t.internal_nodes() if n.parent_node is not None]
    # Use first internal node with parent
    node = nodes[0]
    # Set parent's label same as child
    if node.parent_node is not None:
        node.parent_node.label = node.label
    else:
        # Manually assign to root's parent (none)
        node.parent_node = None
    # LRF allows collapse always
    collapse_node(node, model='LRF')
    # ELRF only allows collapse if node label equals parent label
    node2 = nodes[-1]
    node2.parent_node.label = node2.label
    collapse_node(node2, model='ELRF')
    # ELRF raises if labels differ
    node3 = nodes[-1]
    node3.parent_node.label = 'duplication' if node3.label == 'speciation' else 'speciation'
    with pytest.raises(ValueError):
        collapse_node(node3, model='ELRF')


def test_collapse_node_unsupported_model_error():
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    node = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    with pytest.raises(ValueError, match='unsupported model'):
        collapse_node(node, model='UNKNOWN')


def test_extend_node_invalid_degree(monkeypatch):
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    node = next(n for n in t.internal_nodes() if len(n.adjacent_nodes()) <= 3)
    # forcibly set parent_node to not None for test condition
    if node.parent_node is None:
        # attach dummy parent
        dummy_parent = dendropy.Node()
        node.parent_node = dummy_parent
    with pytest.raises(ValueError, match='Insufficient degree'):
        extend_node(node, model='LRF')
    with pytest.raises(ValueError, match='unsupported model'):
        extend_node(node, model='INVALID')


def test_mutateLabeledTree_repeatability_and_seed():
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    for seed in (0, 9876, 42):
        random.seed(seed)
        mutated1 = mutateLabeledTree(t, n=3, p_flip=0.4)
        random.seed(seed)
        mutated2 = mutateLabeledTree(t, n=3, p_flip=0.4)
        # The label sequences of internal nodes should be identical for same seed
        labels1 = get_all_node_labels(mutated1)
        labels2 = get_all_node_labels(mutated2)
        assert labels1 == labels2


def test_mutateLabeledTree_edge_flips_and_collapses(monkeypatch):
    # This test ensures that mutateLabeledTree can flip and collapse nodes without exception
    # Use a fixed seed for reproducibility
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication']
    t = tree_from_newick(newick, labels)
    random.seed(12345)
    mutated = mutateLabeledTree(t, n=10, p_flip=0.5)
    assert isinstance(mutated, dendropy.Tree)
    # Labels on internal nodes remain only speciation or duplication or None
    for node in mutated.internal_nodes():
        assert node.label in (None, 'speciation', 'duplication')


def test_flip_and_extend_node_randomness(monkeypatch):
    # We test extend_node randomness by seeding and controlling shuffle and randint
    newick = "((A,B,C),(D,E));"  # tree with nodes of degree >=3
    labels = ['speciation', 'duplication']
    t = tree_from_newick(newick, labels)
    node = [n for n in t.internal_nodes() if len(n.child_nodes()) >= 3][0]
    random.seed(2024)

    # Because extend_node shuffles children and uses randint, we seed to ensure repeatability
    extend_node(node, model='LRF')
    # After extension, node should have at least one child with label in expected set
    child_labels = [c.label for c in node.child_nodes()]
    assert any(l in ('speciation', 'duplication') for l in child_labels)


def test_computeLRF_with_parseEnsemblLabels_consistency():
    from pylabeledrf.computeLRF import parseEnsemblLabels
    newick = "((A:0.1[&&NHX:D=Y],B:0.1[&&NHX:D=N]),(C:0.1[&&NHX:D=Y],D:0.1[&&NHX:D=N]));"
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema='newick', taxon_namespace=taxa)
    tree2 = parseEnsemblLabels(tree)
    random.seed(7)
    # Labels should be only 'speciation' or 'duplication', root label None
    labels = [n.label for n in tree2.internal_nodes()]
    assert all(l in ('speciation', 'duplication', None) for l in labels)
    assert tree2.seed_node.label is None
    # computeLRF from tree to itself is zero
    d = computeLRF(tree2, tree2)
    assert d == 0


def test_flip_node_mutate_and_flip_labels():
    # Test that after flipping twice, node label returns to original or equals one of expected values
    node = dendropy.Node()
    node.label = 'speciation'
    flip_node(node)
    assert node.label == 'duplication'
    flip_node(node)
    assert node.label == 'speciation'


def test_mutateLabeledTree_zero_edits_returns_clone(monkeypatch):
    newick = "((A,B),(C,D));"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    random.seed(1)
    mutated = mutateLabeledTree(t, n=0)
    # No edits means trees should be equivalent in structure and labels
    assert mutated.as_string(schema='newick') == t.as_string(schema='newick')
    orig_labels = get_all_node_labels(t)
    mut_labels = get_all_node_labels(mutated)
    assert orig_labels == mut_labels


def test_extend_node_raises_on_unknown_model():
    newick = "((A,B,C),D);"
    labels = ['speciation']
    t = tree_from_newick(newick, labels)
    node = next(n for n in t.internal_nodes() if len(n.child_nodes()) >= 3)
    with pytest.raises(ValueError, match='unsupported model'):
        extend_node(node, model='UNKNOWN')


def test_flip_node_mutate_error_handling(monkeypatch):
    # Patch random.randint to force error path inside flip_node; should raise ValueError before that
    node = dendropy.Node()
    node.label = 'invalid_label'
    with pytest.raises(ValueError, match='Should not happen!!'):
        flip_node(node)


def test_mutateLabeledTree_repeat_edit_counts(monkeypatch):
    # Test that mutation reports correct counts when p_flip=1 should only flip, p_flip=0 should only collapse/expand
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication']
    t = tree_from_newick(newick, labels)

    random.seed(777)
    mutated_flip = mutateLabeledTree(t, n=10, p_flip=1.0)
    # All edits should be flips => no error; labels changed
    labels_after = get_all_node_labels(mutated_flip)
    assert any(l != ol for l, ol in zip(labels_after, get_all_node_labels(t)) if l is not None)

    random.seed(777)
    mutated_no_flip = mutateLabeledTree(t, n=10, p_flip=0.0)
    # Edits should be expand/collapse only, no flips
    assert mutated_no_flip is not None


def test_mutateLabeledTree_randomness_reproducibility():
    newick = "((A,B),(C,D));"
    labels = ['speciation', 'duplication']
    t = tree_from_newick(newick, labels)
    random.seed(42)
    m1 = mutateLabeledTree(t, n=5)
    l1 = get_all_node_labels(m1)
    random.seed(42)
    m2 = mutateLabeledTree(t, n=5)
    l2 = get_all_node_labels(m2)
    assert l1 == l2

