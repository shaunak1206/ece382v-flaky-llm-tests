import pytest
import dendropy

from pylabeledrf.computeLRF import (
    computeLRF,
    mutateLabeledTree,
    collapse_node,
    flip_node,
    extend_node,
)


def make_tree(newick, labels=None, rooted=True):
    taxa = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "default-unrooted",
    )
    if labels is not None:
        internal_nodes = [n for n in tree.preorder_node_iter() if not n.is_leaf()]
        assert len(labels) == len(internal_nodes)
        for node, label in zip(internal_nodes, labels):
            node.label = label
    return tree


def internal_labels(tree):
    return [n.label for n in tree.preorder_node_iter() if not n.is_leaf()]


def test_compute_lrf_identical_trees_is_zero():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    t2 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    t2 = make_tree("((A,C),(B,D));", labels=[None, "speciation", "duplication"])
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_compute_lrf_same_topology_different_labels_counts_substitution():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    t2 = make_tree("((A,B),(C,D));", labels=[None, "duplication", "duplication"])
    assert computeLRF(t1, t2) == 1


def test_compute_lrf_different_topology_positive_distance():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "speciation"])
    t2 = make_tree("((A,C),(B,D));", labels=[None, "speciation", "speciation"])
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_does_not_mutate_inputs():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    t2 = make_tree("((A,C),(B,D));", labels=[None, "duplication", "speciation"])
    before1 = internal_labels(t1)
    before2 = internal_labels(t2)

    _ = computeLRF(t1, t2)

    assert internal_labels(t1) == before1
    assert internal_labels(t2) == before2


def test_compute_lrf_single_leaf_trees_zero():
    t1 = make_tree("(A);", labels=[None])
    t2 = make_tree("(A);", labels=[None])
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_raises_on_different_taxa_sets():
    t1 = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    t2 = make_tree("((A,B),(C,E));", labels=[None, "speciation", "duplication"])
    with pytest.raises(Exception):
        computeLRF(t1, t2)


def test_mutate_labeled_tree_flip_only(monkeypatch):
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=1.0, model="LRF")

    assert out is not t
    assert internal_labels(out) == [None, "duplication", "duplication"]
    assert internal_labels(t) == [None, "speciation", "duplication"]


def test_mutate_labeled_tree_collapse_only_lrf(monkeypatch):
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "speciation"])

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 1.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert out is not t
    assert len([n for n in out.internal_nodes() if not n.is_leaf()]) <= len([n for n in t.internal_nodes() if not n.is_leaf()])


def test_mutate_labeled_tree_extend_only(monkeypatch):
    t = make_tree("(((A,B),C),D);", labels=[None, "speciation", "speciation"])

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 1.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: a)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.shuffle", lambda x: None)

    out = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert out is not t
    assert len(out.internal_nodes()) >= len(t.internal_nodes())


def test_mutate_labeled_tree_elrf_allows_only_same_label_collapse(monkeypatch):
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.99)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    out = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert out is not t


def test_mutate_labeled_tree_unsupported_model_raises():
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="BAD")


def test_collapse_node_lrf_always_collapses():
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    target = [n for n in t.preorder_node_iter() if not n.is_leaf()][1]
    before = len([n for n in t.internal_nodes() if not n.is_leaf()])
    collapse_node(target, model="LRF")
    after = len([n for n in t.internal_nodes() if not n.is_leaf()])
    assert after == before - 1


def test_collapse_node_elrf_requires_matching_labels():
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    target = [n for n in t.preorder_node_iter() if not n.is_leaf()][1]
    with pytest.raises(ValueError):
        collapse_node(target, model="ELRF")


def test_flip_node_toggles_labels_and_rejects_other_values():
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    node1 = [n for n in t.preorder_node_iter() if n.label == "speciation"][0]
    node2 = [n for n in t.preorder_node_iter() if n.label == "duplication"][0]
    flip_node(node1)
    flip_node(node2)
    assert node1.label == "duplication"
    assert node2.label == "speciation"

    node3 = [n for n in t.preorder_node_iter() if not n.is_leaf()][0]
    node3.label = "other"
    with pytest.raises(ValueError):
        flip_node(node3)


def test_extend_node_lrf_raises_on_insufficient_degree():
    t = make_tree("((A,B),(C,D));", labels=[None, "speciation", "duplication"])
    target = [n for n in t.preorder_node_iter() if not n.is_leaf()][1]
    with pytest.raises(ValueError):
        extend_node(target, model="LRF")


def test_extend_node_elrf_preserves_label(monkeypatch):
    t = make_tree("(((A,B),C),D);", labels=[None, "speciation", "speciation"])
    root = t.seed_node

    monkeypatch.setattr("pylabeledrf.computeLRF.random.shuffle", lambda x: None)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: a)

    extend_node(root, model="ELRF")
    assert root.label == "speciation"


def test_compute_lrf_with_unlabeled_internal_nodes_still_works():
    t1 = make_tree("((A,B),(C,D));", labels=[None, None, None])
    t2 = make_tree("((A,B),(C,D));", labels=[None, None, None])
    assert computeLRF(t1, t2) == 0
