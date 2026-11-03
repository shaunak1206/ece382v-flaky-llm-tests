import pytest
import dendropy
import random
import pylabeledrf.computeLRF as lrf


def create_basic_tree_newick():
    # Simple rooted tree Newick string with 4 leaves
    return "((A,B),(C,D));"


def assign_labels_simple(tree, labels):
    """
    Assign labels ('speciation' or 'duplication') to internal nodes in preorder.
    Leaves are unlabeled.
    labels: list of labels for internal nodes in traversal order.
    """
    internal_nodes = list(tree.internal_nodes())
    assert len(labels) == len(internal_nodes)
    for node, label in zip(internal_nodes, labels):
        node.label = label
    tree.seed_node.label = None


def test_computeLRF_identical_trees_same_labels():
    # Seed random used in code to avoid side effects
    random.seed(42)

    # Create two identical trees with identical labels
    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    t1 = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    t2 = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    # Label both trees same way
    labels = ['speciation'] * len(list(t1.internal_nodes()))
    assign_labels_simple(t1, labels)
    assign_labels_simple(t2, labels)

    dist = lrf.computeLRF(t1, t2)
    # For identical trees and labels, distance should be zero
    assert dist == 0


def test_computeLRF_identical_trees_different_labels():
    random.seed(42)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    t1 = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    t2 = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    labels1 = ['speciation'] * len(list(t1.internal_nodes()))
    labels2 = ['duplication'] * len(list(t2.internal_nodes()))

    assign_labels_simple(t1, labels1)
    assign_labels_simple(t2, labels2)

    dist = lrf.computeLRF(t1, t2)
    # Trees same topology, labels swapped: expect substitutions = number internal nodes
    # rf includes sizeIsland sums; verify distance > 0, substitutions >=1
    assert dist > 0


def test_computeLRF_different_topology_same_labels():
    random.seed(42)

    taxa = dendropy.TaxonNamespace()
    # Tree1
    nwk1 = "((A,B),(C,D));"
    t1 = dendropy.Tree.get(data=nwk1, schema="newick", taxon_namespace=taxa)
    # Tree2: different topology
    nwk2 = "(((A,B),C),D);"
    t2 = dendropy.Tree.get(data=nwk2, schema="newick", taxon_namespace=taxa)

    labels1 = ['speciation'] * len(list(t1.internal_nodes()))
    labels2 = ['speciation'] * len(list(t2.internal_nodes()))

    assign_labels_simple(t1, labels1)
    assign_labels_simple(t2, labels2)

    dist = lrf.computeLRF(t1, t2)
    # Different topologies but all labels same: distance > 0
    assert dist > 0


def test_computeLRF_raises_value_error_on_unequal_islands():
    # Create two trees purposely with different number of islands to trigger ValueError directly

    random.seed(42)

    taxa = dendropy.TaxonNamespace()
    nwk1 = "((A,B),(C,D));"
    nwk2 = "((A,B,C),D);"

    t1 = dendropy.Tree.get(data=nwk1, schema="newick", taxon_namespace=taxa)
    t2 = dendropy.Tree.get(data=nwk2, schema="newick", taxon_namespace=taxa)

    # Label all internal nodes as 'speciation'
    labels1 = ['speciation'] * len(list(t1.internal_nodes()))
    labels2 = ['speciation'] * len(list(t2.internal_nodes()))
    assign_labels_simple(t1, labels1)
    assign_labels_simple(t2, labels2)

    with pytest.raises(ValueError, match="Unequal number of islands"):
        lrf.computeLRF(t1, t2)


def test_mutateLabeledTree_flips_only_labels(monkeypatch):
    # Seed random used in mutateLabeledTree code and shuffle
    random.seed(12345)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    # Assign mixture of labels so flip possible
    labels = ['speciation', 'duplication']
    assign_labels_simple(tree, labels)

    # Use p_flip=1 to force flip
    mutated = lrf.mutateLabeledTree(tree, n=5, p_flip=1.0, model='LRF')

    # Check that only existing labels are flipped (spec <-> dup) and no other changes in taxa count or leaf names
    original_leaves = {leaf.taxon.label for leaf in tree.leaf_nodes()}
    mutated_leaves = {leaf.taxon.label for leaf in mutated.leaf_nodes()}
    assert original_leaves == mutated_leaves
    # At least one label flipped compared to original to demonstrate flip effect
    original_labels = [node.label for node in tree.internal_nodes()]
    mutated_labels = [node.label for node in mutated.internal_nodes()]
    # There must be differences due to flips
    assert any(o != m for o, m in zip(original_labels, mutated_labels))


def test_mutateLabeledTree_collapses_only(monkeypatch):
    # Seed random used in mutateLabeledTree and shuffle
    random.seed(2020)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    # Assign all labels identical to allow collapses in LRF model
    labels = ['speciation'] * len(list(tree.internal_nodes()))
    assign_labels_simple(tree, labels)

    # Force p_flip=0, so mutations only collapse or extend, here we expect collapses or extends
    mutated = lrf.mutateLabeledTree(tree, n=3, p_flip=0.0, model='LRF')

    # Check taxa preserved
    original_leaves = {leaf.taxon.label for leaf in tree.leaf_nodes()}
    mutated_leaves = {leaf.taxon.label for leaf in mutated.leaf_nodes()}
    assert original_leaves == mutated_leaves


def test_flip_node_changes_label_and_raises(monkeypatch):
    # Seed random for possible fallback in flip_node (not reachable in given code, but test coverage)
    random.seed(10)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    assign_labels_simple(tree, ['speciation'] * len(list(tree.internal_nodes())))

    node = next(tree.internal_nodes())
    old_label = node.label
    lrf.flip_node(node)
    assert node.label != old_label
    assert node.label in ('speciation', 'duplication')

    # Test flip_node on invalid label triggers ValueError
    node.label = 'invalid_label'
    with pytest.raises(ValueError, match="Should not happen!!"):
        lrf.flip_node(node)


def test_collapse_node_LRF_and_ELRF(monkeypatch):
    random.seed(42)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree_LRF = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    tree_ELRF = tree_LRF.clone(depth=1)

    assign_labels_simple(tree_LRF, ['speciation'] * len(list(tree_LRF.internal_nodes())))
    assign_labels_simple(tree_ELRF, ['speciation'] * len(list(tree_ELRF.internal_nodes())))

    # Pick an internal node with parent for collapse test
    node_LRF = [n for n in tree_LRF.internal_nodes() if n.parent_node is not None][0]
    node_ELRF = [n for n in tree_ELRF.internal_nodes() if n.parent_node is not None][0]

    # LRF collapse should succeed regardless
    lrf.collapse_node(node_LRF, model='LRF')
    # ELRF collapse should succeed only if labels same, here they are 'speciation'
    lrf.collapse_node(node_ELRF, model='ELRF')

    # ELRF collapse with differing labels should raise ValueError
    node_ELRF.label = 'duplication'
    node_ELRF.parent_node.label = 'speciation'
    with pytest.raises(ValueError, match="this edge cannot be collapsed without a flip!"):
        lrf.collapse_node(node_ELRF, model='ELRF')

    # Unsupported model raises ValueError
    with pytest.raises(ValueError, match="unsupported model"):
        lrf.collapse_node(node_LRF, model='UNKNOWN')


def test_extend_node_degree_checks(monkeypatch):
    random.seed(123)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    assign_labels_simple(tree, ['speciation'] * len(list(tree.internal_nodes())))

    # Pick internal node with degree <= 3 to test error raised
    node = [n for n in tree.internal_nodes() if len(n.adjacent_nodes()) <= 3 and n.parent_node is not None][0]

    with pytest.raises(ValueError, match="Insufficient degree. Should not happen "):
        lrf.extend_node(node, model='LRF')

    # Create a node with degree > 3 by adding children manually for a passing test
    node = [n for n in tree.internal_nodes() if n.parent_node is not None][0]
    # forcibly add children until degree > 3
    while len(node.adjacent_nodes()) <= 3:
        node.add_child(dendropy.Node(label='duplication'))

    # Should not raise ValueError now
    # To be sure random.shuffle is deterministic, patch random.shuffle
    def fake_shuffle(x):
        pass  # do nothing, keep order
    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    monkeypatch.setattr(random, "choice", lambda x: 'speciation')
    monkeypatch.setattr(random, "randint", lambda a, b: 2)

    lrf.extend_node(node, model='LRF')


def test_parseEnsemblLabels_basic_conversion():
    random.seed(99)
    # Build a tree with annotations for D and DD
    nwk = "((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;"
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    # Manually annotate internal nodes with D and DD values
    for node in t.internal_nodes():
        node.annotations.add_new("D", "Y")
        node.annotations.add_new("DD", "N")

    t2 = lrf.parseEnsemblLabels(t)
    # root should have label None
    assert t2.seed_node.label is None
    for n in t2.internal_nodes():
        assert n.label == 'duplication'  # due to annotation D=Y


def test_randomLabels_probabilities(monkeypatch):
    # Patch random.random to always return 0.5 so labels alternate deterministically
    monkeypatch.setattr(random, "random", lambda: 0.5)

    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)

    # p_speciation=0.7 means 0.5 < 0.7 => node labeled speciation always
    t2 = lrf.randomLabels(tree, p_speciation=0.7)
    for node in t2.internal_nodes():
        assert node.label == "speciation" or node.label == "duplication"
    # seed_node.label None preserved
    assert t2.seed_node.label is None


def test_ordered_bipartitions_returns_correct_keys():
    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    obs = lrf.ordered_bipartitions(t)
    # keys are tuples of sorted leaf labels
    for key in obs.keys():
        assert isinstance(key, tuple)
        for leaf_label in key:
            assert isinstance(leaf_label, str)
    # values are nodes
    for v in obs.values():
        assert hasattr(v, "label")


def test_maxpath_comparisons():
    p1 = lrf.maxpath(length=3)
    p2 = lrf.maxpath(length=5)
    assert p1 < p2
    assert (p1 == lrf.maxpath(length=3))
    assert (p2 != p1)
    assert str(p1).startswith("Path length:")


# Minimal test for cloneAndReroot does not raise and produces a tree with same leaf labels
def test_cloneAndReroot_basic():
    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    t2 = lrf.cloneAndReroot(t)
    leaf_labels_orig = {leaf.taxon.label for leaf in t.leaf_nodes()}
    leaf_labels_new = {leaf.taxon.label for leaf in t2.leaf_nodes()}
    assert leaf_labels_orig == leaf_labels_new
    # Should be a new tree object (clone)
    assert t is not t2


def test_mutateLabeledTree_invalid_model_raises():
    random.seed(42)
    nwk = create_basic_tree_newick()
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=taxa)
    labels = ['speciation'] * len(list(tree.internal_nodes()))
    assign_labels_simple(tree, labels)

    with pytest.raises(ValueError, match="unsupported model"):
        lrf.mutateLabeledTree(tree, n=1, p_flip=0.5, model="BADMODEL")

