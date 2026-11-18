import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeELRF,
    computeLRF,
    extend_node,
    flip_node,
    getIslands,
    identify_labelled_bipartitions,
    mutateLabeledTree,
    parseEnsemblLabels,
    randomLabels,
)


def make_tree(newick, rooted=True):
    taxa = dendropy.TaxonNamespace()
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        rooting="force-rooted" if rooted else "default-unrooted",
        taxon_namespace=taxa,
    )


def set_internal_labels(tree, labels):
    internal_nodes = list(tree.internal_nodes())
    assert len(internal_nodes) == len(labels)
    for node, lab in zip(internal_nodes, labels):
        node.label = lab
    return tree


def labeled_four_taxon_tree(topology="balanced", labels=("speciation", "duplication")):
    if topology == "balanced":
        t = make_tree("((A,B),(C,D));", rooted=True)
    elif topology == "ladder":
        t = make_tree("(((A,B),C),D);", rooted=True)
    else:
        raise ValueError("unknown topology")
    internal = [n for n in t.internal_nodes()]
    internal[-1].label = None
    internal[0].label = labels[0]
    internal[1].label = labels[1]
    return t


def test_compute_lrf_identical_trees_zero():
    t1 = labeled_four_taxon_tree()
    t2 = labeled_four_taxon_tree()
    assert computeLRF(t1, t2) == 0


def test_compute_lrf_is_symmetric():
    t1 = labeled_four_taxon_tree(labels=("speciation", "duplication"))
    t2 = labeled_four_taxon_tree(labels=("duplication", "speciation"))
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_compute_lrf_label_mismatch_positive():
    t1 = labeled_four_taxon_tree(labels=("speciation", "speciation"))
    t2 = labeled_four_taxon_tree(labels=("speciation", "duplication"))
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_different_topology_positive():
    t1 = labeled_four_taxon_tree(topology="balanced")
    t2 = labeled_four_taxon_tree(topology="ladder")
    assert computeLRF(t1, t2) > 0


def test_compute_lrf_does_not_modify_inputs():
    t1 = labeled_four_taxon_tree()
    t2 = labeled_four_taxon_tree(labels=("duplication", "speciation"))
    before1 = t1.as_string(schema="newick")
    before2 = t2.as_string(schema="newick")
    computeLRF(t1, t2)
    assert t1.as_string(schema="newick") == before1
    assert t2.as_string(schema="newick") == before2


def test_identify_labelled_bipartitions_returns_expected_count():
    t = labeled_four_taxon_tree()
    biparts, mapping = identify_labelled_bipartitions(t)
    assert len(biparts) == 2
    assert len(mapping) == 2
    assert all(bp in mapping for bp in biparts)


def test_get_islands_all_goodedges_single_island():
    t = labeled_four_taxon_tree()
    biparts, _ = identify_labelled_bipartitions(t)
    islands, node_island, sizes = getIslands(t, set(biparts))
    assert len(islands) == 3  # index 0 is dummy/root container; 2 internal islands + root/leaf handling
    assert sizes[1] == 1
    assert sizes[2] == 1
    assert len(node_island) >= 2


def test_compute_lrf_single_leaf_tree_zero():
    t1 = make_tree("(A);", rooted=True)
    t2 = make_tree("(A);", rooted=True)
    assert computeLRF(t1, t2) == 0


def test_flip_node_toggles_labels():
    t = labeled_four_taxon_tree()
    node = [n for n in t.internal_nodes() if n.label is not None][0]
    original = node.label
    flip_node(node)
    assert node.label != original
    flip_node(node)
    assert node.label == original


def test_flip_node_rejects_invalid_label():
    t = labeled_four_taxon_tree()
    node = [n for n in t.internal_nodes() if n.label is not None][0]
    node.label = "other"
    with pytest.raises(ValueError):
        flip_node(node)


def test_collapse_node_lrf_collapses_edge():
    t = labeled_four_taxon_tree()
    node = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    before = len(list(t.internal_nodes()))
    collapse_node(node, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after < before


def test_collapse_node_elrf_requires_matching_labels():
    t = labeled_four_taxon_tree(labels=("speciation", "duplication"))
    node = [n for n in t.internal_nodes() if n.parent_node is not None][0]
    if node.parent_node is not None:
        node.parent_node.label = "speciation"
        node.label = "duplication"
    with pytest.raises(ValueError):
        collapse_node(node, model="ELRF")


def test_extend_node_lrf_increases_internal_nodes(monkeypatch):
    t = make_tree("((A,B,C),D);", rooted=True)
    root = t.seed_node
    root.label = "speciation"
    child = [n for n in t.internal_nodes() if n is not root][0]
    child.label = "duplication"

    monkeypatch.setattr(random, "shuffle", lambda x: None)
    monkeypatch.setattr(random, "randint", lambda a, b: 2)

    before = len(list(t.internal_nodes()))
    extend_node(root, model="LRF")
    after = len(list(t.internal_nodes()))
    assert after > before


def test_mutate_labeled_tree_returns_clone_and_changes_tree(monkeypatch):
    t = labeled_four_taxon_tree()
    before = t.as_string(schema="newick")

    monkeypatch.setattr(random, "random", lambda: 0.0)  # always flip
    monkeypatch.setattr(random, "randint", lambda a, b: a)

    mutated = mutateLabeledTree(t, 1, p_flip=1.0, model="LRF")
    assert mutated is not t
    assert t.as_string(schema="newick") == before
    assert mutated.as_string(schema="newick") != before


def test_random_labels_assigns_expected_labels_with_seed():
    t = labeled_four_taxon_tree()
    random.seed(0)
    out = randomLabels(t, p_speciation=1.0)
    for n in out.internal_nodes():
        if n is not out.seed_node:
            assert n.label == "speciation"
    assert out.seed_node.label is None


def test_parse_ensembl_labels_sets_labels_and_root_none():
    taxa = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(
        data="((A[&&NHX:D=Y],B[&&NHX:D=N]),(C[&&NHX:DD=Y],D[&&NHX:D=N]));",
        schema="newick",
        rooting="force-rooted",
        taxon_namespace=taxa,
    )
    out = parseEnsemblLabels(t)
    assert out.seed_node.label is None
    labels = [n.label for n in out.internal_nodes() if n is not out.seed_node]
    assert all(l in ("speciation", "duplication") for l in labels)
    assert "duplication" in labels


def test_compute_elrf_identical_trees_zero_or_nonnegative():
    t1 = labeled_four_taxon_tree()
    t2 = labeled_four_taxon_tree()
    dist = computeELRF(t1, t2)
    assert dist >= 0
