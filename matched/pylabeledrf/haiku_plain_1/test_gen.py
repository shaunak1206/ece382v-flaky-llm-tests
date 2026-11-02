"""Unit tests for computeLRF, computeELRF and mutateLabeledTree (plus helpers)."""
import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeELRF,
    computeLRF,
    extend_node,
    flip_node,
    mutateLabeledTree,
)

BALANCED = "[&R] ((A,B)speciation,(C,D)speciation);"
BALANCED_FLIP_AB = "[&R] ((A,B)duplication,(C,D)speciation);"
BALANCED_FLIP_BOTH = "[&R] ((A,B)duplication,(C,D)duplication);"
BALANCED_ROOT_DUP = "[&R] ((A,B)speciation,(C,D)speciation)duplication;"
BALANCED_LABELLED_ROOT = "[&R] ((A,B)speciation,(C,D)duplication)speciation;"
REARRANGED = "[&R] ((A,C)speciation,(B,D)speciation);"
STAR4 = "[&R] (A,B,C,D);"
SIX = (
    "[&R] ((A,B)speciation,"
    "((C,D)duplication,(E,F)speciation)speciation)speciation;"
)
SIX_CD_FLIPPED = (
    "[&R] ((A,B)speciation,"
    "((C,D)speciation,(E,F)speciation)speciation)speciation;"
)


def parse(newick, ns=None):
    if ns is None:
        ns = dendropy.TaxonNamespace()
    return dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=ns)


def parse_pair(newick1, newick2):
    ns = dendropy.TaxonNamespace()
    return parse(newick1, ns), parse(newick2, ns)


def internal_labels(tree):
    return [n.label for n in tree.internal_nodes()]


def leaf_labels(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_nodes())


def non_root_internal(tree):
    return [n for n in tree.internal_nodes() if n.parent_node is not None]


# ---------------------------------------------------------------- computeLRF

def test_lrf_identical_trees_distance_zero():
    t1, t2 = parse_pair(BALANCED, BALANCED)
    assert computeLRF(t1, t2) == 0


def test_lrf_leaf_order_does_not_matter():
    t1, t2 = parse_pair(BALANCED, "[&R] ((D,C)speciation,(B,A)speciation);")
    assert computeLRF(t1, t2) == 0


def test_lrf_single_label_flip_costs_one():
    t1, t2 = parse_pair(BALANCED, BALANCED_FLIP_AB)
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_lrf_two_label_flips_cost_two():
    t1, t2 = parse_pair(BALANCED, BALANCED_FLIP_BOTH)
    assert computeLRF(t1, t2) == 2


def test_lrf_root_label_mismatch_costs_one():
    t1, t2 = parse_pair(BALANCED, BALANCED_ROOT_DUP)
    assert computeLRF(t1, t2) == 1


def test_lrf_rearranged_topology_costs_four():
    # Each rooted 4-leaf tree has two clusters the other lacks, so the
    # rooted Robinson-Foulds style count is 2 + 2.
    t1, t2 = parse_pair(BALANCED, REARRANGED)
    assert computeLRF(t1, t2) == 4


def test_lrf_star_versus_binary_costs_two():
    t_bin, t_star = parse_pair(BALANCED, STAR4)
    assert computeLRF(t_bin, t_star) == 2
    assert computeLRF(t_star, t_bin) == 2


def test_lrf_larger_identical_tree_distance_zero():
    t1, t2 = parse_pair(SIX, SIX)
    assert computeLRF(t1, t2) == 0


def test_lrf_flip_deep_in_larger_tree_costs_one():
    t1, t2 = parse_pair(SIX, SIX_CD_FLIPPED)
    assert computeLRF(t1, t2) == 1


def test_lrf_does_not_modify_inputs():
    t1, t2 = parse_pair(BALANCED, BALANCED_FLIP_AB)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# --------------------------------------------------------------- computeELRF

def test_elrf_identical_trees_distance_zero():
    t1, t2 = parse_pair(BALANCED, BALANCED)
    assert computeELRF(t1, t2) == 0


def test_elrf_single_label_flip_costs_one():
    t1, t2 = parse_pair(BALANCED, BALANCED_FLIP_AB)
    assert computeELRF(t1, t2) == 1


# ------------------------------------------------------- mutateLabeledTree

def test_mutate_zero_edits_returns_equal_copy():
    t = parse(BALANCED)
    result = mutateLabeledTree(t, 0)
    assert result is not t
    assert computeLRF(result, t) == 0


def test_mutate_returns_new_tree_and_leaves_input_unchanged():
    t = parse(BALANCED)
    before = t.as_string("newick")
    random.seed(1)
    result = mutateLabeledTree(t, 3, p_flip=1.0)
    assert result is not t
    assert t.as_string("newick") == before
    assert result.as_string("newick") != before


def test_mutate_single_flip_changes_exactly_one_label():
    t = parse(BALANCED)
    random.seed(2)
    result = mutateLabeledTree(t, 1, p_flip=1.0)
    assert computeLRF(result, t) == 1
    changed = sum(
        a != b for a, b in zip(internal_labels(result), internal_labels(t))
    )
    assert changed == 1


def test_mutate_collapse_removes_one_internal_node():
    t = parse(BALANCED)
    random.seed(3)
    result = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert len(result.internal_nodes()) == len(t.internal_nodes()) - 1
    assert leaf_labels(result) == ["A", "B", "C", "D"]


def test_mutate_preserves_leaf_set():
    t = parse(SIX)
    random.seed(7)
    result = mutateLabeledTree(t, 3)
    assert leaf_labels(result) == ["A", "B", "C", "D", "E", "F"]


def test_mutate_same_seed_gives_same_output():
    t = parse(SIX)
    random.seed(11)
    first = mutateLabeledTree(t, 3).as_string("newick")
    random.seed(11)
    second = mutateLabeledTree(t, 3).as_string("newick")
    assert first == second


def test_mutate_unknown_model_raises():
    t = parse(BALANCED_LABELLED_ROOT)
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")


# ------------------------------------------------------------------ helpers

def test_collapse_node_elrf_rejects_label_mismatch():
    t = parse(BALANCED_LABELLED_ROOT)
    cd = [n for n in non_root_internal(t) if n.label == "duplication"][0]
    with pytest.raises(ValueError):
        collapse_node(cd, model="ELRF")
    assert len(t.internal_nodes()) == 3


def test_collapse_node_elrf_allows_matching_label():
    t = parse(BALANCED_LABELLED_ROOT)
    ab = [n for n in non_root_internal(t) if n.label == "speciation"][0]
    collapse_node(ab, model="ELRF")
    assert len(t.internal_nodes()) == 2


def test_flip_node_swaps_labels_and_rejects_unlabelled():
    t = parse(BALANCED_LABELLED_ROOT)
    s_node = [n for n in non_root_internal(t) if n.label == "speciation"][0]
    d_node = [n for n in non_root_internal(t) if n.label == "duplication"][0]
    flip_node(s_node)
    assert s_node.label == "duplication"
    flip_node(d_node)
    assert d_node.label == "speciation"

    unlabelled = parse("[&R] ((A,B),(C,D));")
    ab = non_root_internal(unlabelled)[0]
    assert ab.label is None
    with pytest.raises(ValueError):
        flip_node(ab)


def test_extend_node_rejects_low_degree_node():
    t = parse(BALANCED)
    ab = non_root_internal(t)[0]
    with pytest.raises(ValueError):
        extend_node(ab)


def test_extend_node_splits_star_keeping_leaves():
    random.seed(5)
    t = parse("[&R] (A,B,C,D,E);")
    extend_node(t.seed_node)
    assert len(t.internal_nodes()) == 2
    assert leaf_labels(t) == ["A", "B", "C", "D", "E"]
    assert 2 <= len(t.seed_node.child_nodes()) <= 4
