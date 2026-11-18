import copy
import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    computeLRF,
    collapse_node,
    extend_node,
    flip_node,
    mutateLabeledTree,
    parseEnsemblLabels,
    randomLabels,
)


def make_tree(newick, labels=None, rooted=True):
    taxon_namespace = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=taxon_namespace,
        rooting="force-rooted" if rooted else "default-unrooted",
    )
    if labels:
        assign_internal_labels(tree, labels)
    return tree


def assign_internal_labels(tree, labels):
    """
    labels: dict mapping frozenset({"A","B",...}) -> label
    Root label is preserved unless explicitly present in labels.
    """
    for node in tree.internal_nodes():
        leafset = frozenset(leaf.taxon.label for leaf in node.leaf_nodes())
        if leafset in labels:
            node.label = labels[leafset]


def internal_leafsets(tree):
    return {
        frozenset(leaf.taxon.label for leaf in node.leaf_nodes()): node
        for node in tree.internal_nodes()
    }


def simple_binary_tree():
    # Rooted binary tree with two non-root internal nodes
    return make_tree("((A,B),(C,D));")


def test_compute_lrf_identical_trees_is_zero():
    t1 = simple_binary_tree()
    t2 = simple_binary_tree()
    assign_internal_labels(
        t1,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assign_internal_labels(
        t2,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = simple_binary_tree()
    t2 = simple_binary_tree()
    assign_internal_labels(
        t1,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assign_internal_labels(
        t2,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "duplication",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_compute_lrf_label_substitution_cost_one():
    t1 = simple_binary_tree()
    t2 = simple_binary_tree()
    assign_internal_labels(
        t1,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assign_internal_labels(
        t2,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "duplication",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assert computeLRF(t1, t2) == 1


def test_compute_lrf_different_topology_positive_distance():
    t1 = make_tree("((A,B),(C,D));")
    t2 = make_tree("((A,C),(B,D));")
    assign_internal_labels(
        t1,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "speciation",
        },
    )
    assign_internal_labels(
        t2,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "C"}): "speciation",
            frozenset({"B", "D"}): "speciation",
        },
    )
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_does_not_mutate_inputs():
    t1 = simple_binary_tree()
    t2 = simple_binary_tree()
    assign_internal_labels(
        t1,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )
    assign_internal_labels(
        t2,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "duplication",
            frozenset({"C", "D"}): "duplication",
        },
    )

    before1 = t1.as_string(schema="newick")
    before2 = t2.as_string(schema="newick")
    _ = computeLRF(t1, t2)
    assert t1.as_string(schema="newick") == before1
    assert t2.as_string(schema="newick") == before2


def test_flip_node_toggles_speciation_to_duplication():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    node.label = "speciation"
    flip_node(node)
    assert node.label == "duplication"


def test_flip_node_toggles_duplication_to_speciation():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    node.label = "duplication"
    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_rejects_invalid_label():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    node.label = "x"
    with pytest.raises(ValueError):
        flip_node(node)


def test_collapse_node_lrf_reduces_internal_nodes():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    before = len(list(t.internal_nodes()))
    collapse_node(node, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_collapse_node_elrf_requires_same_label():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    parent = node.parent_node
    parent.label = "speciation"
    node.label = "speciation"
    before = len(list(t.internal_nodes()))
    collapse_node(node, model="ELRF")
    after = len(list(t.internal_nodes()))
    assert after == before - 1


def test_collapse_node_elrf_rejects_different_labels():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    node.parent_node.label = "speciation"
    node.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(node, model="ELRF")


def test_extend_node_rejects_unsupported_model():
    t = make_tree("(((A,B),C),D);")
    node = internal_leafsets(t)[frozenset({"A", "B", "C"})]
    with pytest.raises(ValueError):
        extend_node(node, model="BAD")


def test_extend_node_rejects_insufficient_degree():
    t = simple_binary_tree()
    node = internal_leafsets(t)[frozenset({"A", "B"})]
    with pytest.raises(ValueError):
        extend_node(node, model="LRF")


def test_random_labels_assigns_only_valid_internal_labels():
    t = simple_binary_tree()
    labeled = randomLabels(t, p_speciation=0.0)
    assert labeled.seed_node.label is None
    for node in labeled.internal_nodes():
        if node is labeled.seed_node:
            continue
        assert node.label in {"speciation", "duplication"}


def test_parse_ensembl_labels_uses_d_annotations():
    taxon_namespace = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data="((A,B),(C,D));",
        schema="newick",
        taxon_namespace=taxon_namespace,
        rooting="force-rooted",
    )
    # Attach annotations expected by parseEnsemblLabels
    nodes = internal_leafsets(tree)
    for clade, node in nodes.items():
        if len(clade) == 4:
            continue
        node.annotations.add_new("D", "Y" if clade == frozenset({"A", "B"}) else "N")
        node.annotations.add_new("DD", "N")
    parsed = parseEnsemblLabels(tree)
    parsed_nodes = internal_leafsets(parsed)
    assert parsed_nodes[frozenset({"A", "B"})].label == "duplication"
    assert parsed_nodes[frozenset({"C", "D"})].label == "speciation"
    assert parsed.seed_node.label is None


def test_mutate_labeled_tree_flips_a_node_when_forced(monkeypatch):
    t = simple_binary_tree()
    assign_internal_labels(
        t,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )

    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda a, b: 0)

    mutated = mutateLabeledTree(t, n=1, p_flip=1.0, model="LRF")
    labels = {frozenset(leaf.taxon.label for leaf in node.leaf_nodes()): node.label for node in mutated.internal_nodes()}
    assert labels[frozenset({"A", "B"})] == "duplication"


def test_mutate_labeled_tree_collapse_occurs_when_forced(monkeypatch):
    t = simple_binary_tree()
    assign_internal_labels(
        t,
        {
            frozenset({"A", "B", "C", "D"}): None,
            frozenset({"A", "B"}): "speciation",
            frozenset({"C", "D"}): "duplication",
        },
    )

    monkeypatch.setattr(random, "random", lambda: 1.0)  # avoid flip
    monkeypatch.setattr(random, "randint", lambda a, b: 0)
    monkeypatch.setattr(random, "shuffle", lambda x: None)

    before = len(list(t.internal_nodes()))
    mutated = mutateLabeledTree(t, n=1, p_flip=0.0, model="LRF")
    after = len(list(mutated.internal_nodes()))
    assert after < before
