import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeLRF,
    extend_node,
    flip_node,
    mutateLabeledTree,
    parseEnsemblLabels,
    randomLabels,
)


def make_tree(newick, rooted=True):
    taxa = dendropy.TaxonNamespace()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxa,
        rooting="force-rooted" if rooted else "default-unrooted",
    )


def labeled_tree(newick, internal_labels, rooted=True):
    tree = make_tree(newick, rooted=rooted)
    internal_nodes = list(tree.internal_nodes())
    assert len(internal_nodes) == len(internal_labels)
    for node, label in zip(internal_nodes, internal_labels):
        node.label = label
    return tree


def internal_labels(tree):
    return [n.label for n in tree.internal_nodes()]


def test_compute_lrf_identical_trees_is_zero():
    t1 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])
    t2 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])

    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])
    t2 = labeled_tree("((A,C),(B,D));", ["speciation", "duplication", None])

    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)

    assert d12 == d21
    assert d12 >= 0


def test_compute_lrf_different_topologies_positive():
    t1 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])
    t2 = labeled_tree("((A,C),(B,D));", ["speciation", "duplication", None])

    assert computeLRF(t1, t2) > 0


def test_compute_lrf_does_not_mutate_inputs():
    t1 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])
    t2 = labeled_tree("((A,C),(B,D));", ["speciation", "duplication", None])

    before1 = internal_labels(t1)
    before2 = internal_labels(t2)
    _ = computeLRF(t1, t2)

    assert internal_labels(t1) == before1
    assert internal_labels(t2) == before2


def test_compute_lrf_raises_for_unequal_number_of_islands():
    t1 = labeled_tree("(((A,B),C),D);", ["speciation", "duplication", "speciation", None])
    t2 = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])

    with pytest.raises(ValueError, match="Unequal number of islands"):
        computeLRF(t1, t2)


def test_flip_node_swaps_speciation_and_duplication():
    t = make_tree("(A,B);")
    node = next(n for n in t.internal_nodes())
    node.label = "speciation"

    flip_node(node)
    assert node.label == "duplication"

    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_raises_on_invalid_label():
    t = make_tree("(A,B);")
    node = next(n for n in t.internal_nodes())
    node.label = None

    with pytest.raises(ValueError, match="Should not happen"):
        flip_node(node)


def test_collapse_node_lrf_collapses_any_internal_edge():
    t = labeled_tree("(((A,B),C),D);", ["speciation", "duplication", "speciation", None])
    node = next(n for n in t.internal_nodes() if n.parent_node is not None)

    pre_internal_count = len(list(t.internal_nodes()))
    collapse_node(node, model="LRF")
    post_internal_count = len(list(t.internal_nodes()))

    assert post_internal_count < pre_internal_count


def test_collapse_node_elrf_requires_same_label_as_parent():
    t = labeled_tree("(((A,B),C),D);", ["speciation", "duplication", "speciation", None])
    node = next(n for n in t.internal_nodes() if n.parent_node is not None)
    assert node.label != node.parent_node.label

    with pytest.raises(ValueError, match="cannot be collapsed without a flip"):
        collapse_node(node, model="ELRF")


def test_collapse_node_rejects_unsupported_model():
    t = labeled_tree("((A,B),C);", ["speciation", None])
    node = next(n for n in t.internal_nodes() if n.parent_node is not None)

    with pytest.raises(ValueError, match="unsupported model"):
        collapse_node(node, model="BAD")


def test_extend_node_rejects_unsupported_model():
    t = make_tree("(((A,B),C),D);")
    node = next(n for n in t.internal_nodes() if n.parent_node is not None)
    node.label = "speciation"

    with pytest.raises(ValueError, match="unsupported model"):
        extend_node(node, model="BAD")


def test_random_labels_is_deterministic_with_seed():
    random.seed(12345)
    t1 = labeled_tree("((A,B),(C,D));", [None, None, None])
    out1 = randomLabels(t1, p_speciation=0.5)

    random.seed(12345)
    t2 = labeled_tree("((A,B),(C,D));", [None, None, None])
    out2 = randomLabels(t2, p_speciation=0.5)

    assert internal_labels(out1) == internal_labels(out2)
    assert out1.seed_node.label is None
    assert out2.seed_node.label is None


def test_parse_ensembl_labels_maps_annotations_to_expected_labels():
    tree = make_tree("((A,B),(C,D));")
    internals = list(tree.internal_nodes())
    for node in internals:
        node.annotations.add_new("D", "N")
        node.annotations.add_new("DD", "N")
    internals[1].annotations["D"].value = "Y"

    parsed = parseEnsemblLabels(tree)

    assert parsed.seed_node.label is None
    labels = [n.label for n in parsed.internal_nodes() if n is not parsed.seed_node]
    assert "duplication" in labels
    assert all(label in ("speciation", "duplication", None) for label in internal_labels(parsed))


def test_mutate_labeled_tree_flip_only_is_deterministic(monkeypatch):
    random.seed(7)
    tree = labeled_tree("((A,B),(C,D));", ["speciation", "duplication", None])

    monkeypatch.setattr("pylabeledrf.computeLRF.random.random", lambda: 0.0)
    monkeypatch.setattr("pylabeledrf.computeLRF.random.randint", lambda a, b: 0)

    mutated = mutateLabeledTree(tree, n=3, p_flip=1.0, model="LRF")

    assert internal_labels(mutated).count("speciation") == 1
    assert internal_labels(mutated).count("duplication") == 1
    assert mutated.seed_node.label is None
