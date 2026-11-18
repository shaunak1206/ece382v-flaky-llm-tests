import pytest
import dendropy

from pylabeledrf.computeLRF import (
    computeLRF,
    mutateLabeledTree,
    collapse_node,
    extend_node,
    flip_node,
    randomLabels,
)


def make_tree(newick, taxon_namespace=None):
    if taxon_namespace is None:
        taxon_namespace = dendropy.TaxonNamespace()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxon_namespace,
        preserve_underscores=True,
    )


def set_internal_labels(tree, labels):
    internal_nodes = list(tree.internal_nodes())
    # Keep the root/seed node unlabeled unless explicitly included by caller.
    for node, label in zip(internal_nodes, labels):
        node.label = label
    return tree


def labeled_binary_tree1():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    # internal_nodes() typically includes seed_node first; label all internal nodes.
    set_internal_labels(t, [None, "speciation", "duplication"])
    return t


def labeled_binary_tree2_same():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "speciation", "duplication"])
    return t


def labeled_binary_tree2_diff_label():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "duplication", "duplication"])
    return t


def labeled_binary_tree_different_topology():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,C),(B,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "speciation", "duplication"])
    return t


def test_computeLRF_identical_trees_zero_distance():
    t1 = labeled_binary_tree1()
    t2 = labeled_binary_tree2_same()
    assert computeLRF(t1, t2) == 0


def test_computeLRF_is_symmetric():
    t1 = labeled_binary_tree1()
    t2 = labeled_binary_tree_different_topology()
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_computeLRF_same_topology_different_labels_positive():
    t1 = labeled_binary_tree1()
    t2 = labeled_binary_tree2_diff_label()
    assert computeLRF(t1, t2) > 0


def test_computeLRF_different_topology_positive():
    t1 = labeled_binary_tree1()
    t2 = labeled_binary_tree_different_topology()
    assert computeLRF(t1, t2) > 0


def test_computeLRF_does_not_mutate_input_trees():
    t1 = labeled_binary_tree1()
    t2 = labeled_binary_tree2_same()
    before1 = [n.label for n in t1.internal_nodes()]
    before2 = [n.label for n in t2.internal_nodes()]

    _ = computeLRF(t1, t2)

    after1 = [n.label for n in t1.internal_nodes()]
    after2 = [n.label for n in t2.internal_nodes()]
    assert after1 == before1
    assert after2 == before2


def test_computeLRF_star_trees_with_mismatched_labels():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("(A,B,C,D);", taxon_namespace=taxa)
    t2 = make_tree("(A,B,C,D);", taxon_namespace=taxa)
    t1.seed_node.label = "speciation"
    t2.seed_node.label = "duplication"
    assert computeLRF(t1, t2) == 1


def test_randomLabels_returns_clone_and_sets_seed_none():
    t = labeled_binary_tree1()
    out = randomLabels(t, p_speciation=1.0)

    assert out is not t
    assert t.seed_node.label is None or t.seed_node.label == None
    assert out.seed_node.label is None
    assert all(n.label == "speciation" or n.is_leaf() or n is out.seed_node for n in out.internal_nodes())


def test_mutateLabeledTree_returns_new_tree_and_leaves_original_unchanged(monkeypatch):
    t = labeled_binary_tree1()
    before = [n.label for n in t.internal_nodes()]

    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=1.0, model="LRF")

    assert out is not t
    assert [n.label for n in t.internal_nodes()] == before
    assert [n.label for n in out.internal_nodes()] != before


def test_mutateLabeledTree_flip_path_flips_one_label(monkeypatch):
    t = labeled_binary_tree1()
    labels_before = [n.label for n in t.internal_nodes()]

    seq = iter([0.0, 0])  # flip branch, pick first flip candidate
    monkeypatch.setattr("random.random", lambda: next(seq))
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=1.0, model="LRF")
    labels_after = [n.label for n in out.internal_nodes()]

    assert labels_after != labels_before
    assert sorted(labels_after.count(x) for x in ["speciation", "duplication", None]) != sorted(
        labels_before.count(x) for x in ["speciation", "duplication", None]
    )


def test_collapse_node_lrf_reduces_internal_nodes():
    t = labeled_binary_tree1()
    target = next(n for n in t.internal_nodes() if n.parent_node is not None and not n.is_leaf())
    before = len(list(t.internal_nodes()))

    collapse_node(target, model="LRF")

    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_collapse_node_elrf_requires_same_label():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "speciation", "duplication"])
    target = next(n for n in t.internal_nodes() if n.parent_node is not None and not n.is_leaf())

    with pytest.raises(ValueError, match="cannot be collapsed"):
        collapse_node(target, model="ELRF")


def test_extend_node_lrf_increases_internal_node_count(monkeypatch):
    taxa = dendropy.TaxonNamespace()
    t = make_tree("(A,B,C,D);", taxon_namespace=taxa)
    t.seed_node.label = "speciation"
    before = len(list(t.internal_nodes()))

    monkeypatch.setattr("random.shuffle", lambda x: None)
    monkeypatch.setattr("random.choice", lambda seq: "duplication")
    monkeypatch.setattr("random.randint", lambda a, b: 2)

    extend_node(t.seed_node, model="LRF")

    after = len(list(t.internal_nodes()))
    assert after == before + 1
    assert len(t.seed_node.child_nodes()) >= 2


def test_extend_node_invalid_model_raises():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("(A,B,C,D);", taxon_namespace=taxa)
    t.seed_node.label = "speciation"

    with pytest.raises(ValueError, match="unsupported model"):
        extend_node(t.seed_node, model="BAD")


def test_flip_node_swaps_labels():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "speciation", "duplication"])
    target = next(n for n in t.internal_nodes() if n.label == "speciation")

    flip_node(target)
    assert target.label == "duplication"


def test_flip_node_invalid_label_raises():
    taxa = dendropy.TaxonNamespace()
    t = make_tree("((A,B),(C,D));", taxon_namespace=taxa)
    set_internal_labels(t, [None, "unknown", "duplication"])
    target = next(n for n in t.internal_nodes() if n.label == "unknown")

    with pytest.raises(ValueError, match="Should not happen"):
        flip_node(target)
