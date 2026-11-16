"""Unit tests for computeLRF and mutateLabeledTree in pylabeledrf.computeLRF."""
import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tree(newick, taxa):
    return dendropy.Tree.get(data=newick, schema="newick", rooted=True,
                             taxon_namespace=taxa)


def _pair(newick1, newick2):
    # both trees must share one taxon namespace so bipartitions are comparable
    taxa = dendropy.TaxonNamespace()
    return _tree(newick1, taxa), _tree(newick2, taxa)


def _leaves(tree):
    return frozenset(leaf.taxon.label for leaf in tree.leaf_nodes())


def _clade_labels(tree):
    """Map each internal node's leaf set (frozenset) to its label."""
    out = {}
    for node in tree.internal_nodes():
        key = frozenset(leaf.taxon.label for leaf in node.leaf_nodes())
        out[key] = node.label
    return out


SPEC = "speciation"
DUP = "duplication"


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

def test_identical_trees_have_zero_distance():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)duplication);",
        "((A,B)speciation,(C,D)duplication);",
    )
    assert computeLRF(t1, t2) == 0


def test_single_label_change_costs_one():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)speciation);",
        "((A,B)duplication,(C,D)speciation);",
    )
    assert computeLRF(t1, t2) == 1


def test_two_label_changes_cost_two():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)speciation);",
        "((A,B)duplication,(C,D)duplication);",
    )
    assert computeLRF(t1, t2) == 2


def test_label_change_deep_in_tree_costs_one():
    t1, t2 = _pair(
        "(((A,B)speciation,C)speciation,((D,E)speciation,F)speciation);",
        "(((A,B)speciation,C)speciation,((D,E)duplication,F)speciation);",
    )
    assert computeLRF(t1, t2) == 1


def test_different_topologies_all_speciation_costs_four():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)speciation);",
        "((A,C)speciation,(B,D)speciation);",
    )
    assert computeLRF(t1, t2) == 4


def test_different_topologies_mixed_labels_costs_four():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)duplication);",
        "((A,C)speciation,(B,D)duplication);",
    )
    assert computeLRF(t1, t2) == 4


def test_polytomy_requires_one_collapse():
    # t2 lacks the CD clade, so one collapse is needed on the t1 side
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)speciation);",
        "((A,B)speciation,C,D);",
    )
    assert computeLRF(t1, t2) == 1


def test_distance_is_symmetric():
    pairs = [
        ("((A,B)speciation,(C,D)duplication);",
         "((A,C)speciation,(B,D)duplication);"),
        ("((A,B)speciation,(C,D)speciation);",
         "((A,B)duplication,C,D);"),
    ]
    for nw1, nw2 in pairs:
        t1, t2 = _pair(nw1, nw2)
        assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_returns_non_negative_integer():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)duplication);",
        "((A,C)duplication,(B,D)speciation);",
    )
    d = computeLRF(t1, t2)
    assert isinstance(d, int)
    assert d >= 0


def test_inputs_are_not_modified():
    t1, t2 = _pair(
        "((A,B)speciation,(C,D)speciation);",
        "((A,C)duplication,(B,D)speciation);",
    )
    before1 = t1.as_string(schema="newick")
    before2 = t2.as_string(schema="newick")
    computeLRF(t1, t2)
    assert t1.as_string(schema="newick") == before1
    assert t2.as_string(schema="newick") == before2


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_zero_edits_returns_equivalent_tree():
    tree = _tree("((A,B)speciation,(C,D)duplication);", dendropy.TaxonNamespace())
    out = mutateLabeledTree(tree, 0)
    assert out is not tree
    assert _leaves(out) == _leaves(tree)
    assert _clade_labels(out) == _clade_labels(tree)


def test_mutation_returns_new_tree_and_leaves_input_unchanged():
    tree = _tree("((A,B)speciation,(C,D)duplication);", dendropy.TaxonNamespace())
    before = tree.as_string(schema="newick")
    random.seed(0)
    out = mutateLabeledTree(tree, 1, p_flip=0.3)
    assert out is not tree
    assert tree.as_string(schema="newick") == before


def test_flip_changes_exactly_one_label_and_keeps_topology():
    tree = _tree("((A,B)speciation,(C,D)duplication);", dendropy.TaxonNamespace())
    original = _clade_labels(tree)
    random.seed(1)
    out = mutateLabeledTree(tree, 1, p_flip=1.0)
    mutated = _clade_labels(out)
    assert set(mutated.keys()) == set(original.keys())
    changed = [k for k in original if original[k] != mutated[k]]
    assert len(changed) == 1
    k = changed[0]
    assert {original[k], mutated[k]} == {SPEC, DUP}


def test_collapse_removes_exactly_one_internal_node():
    tree = _tree("((A,B)speciation,(C,D)duplication);", dendropy.TaxonNamespace())
    assert len(tree.internal_nodes()) == 3
    random.seed(2)
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model='LRF')
    assert len(out.internal_nodes()) == 2
    assert _leaves(out) == _leaves(tree)


def test_elrf_only_collapses_edges_matching_parent_label():
    # root and AB share a label, CD does not: only AB may be collapsed
    tree = _tree(
        "((A,B)speciation,(C,D)duplication)speciation;",
        dendropy.TaxonNamespace(),
    )
    random.seed(3)
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model='ELRF')
    keys = set(_clade_labels(out).keys())
    assert frozenset({"A", "B"}) not in keys
    assert frozenset({"C", "D"}) in keys


def test_expansion_or_collapse_of_polytomy_keeps_leaves():
    # node with 4 children can be either collapsed (1 internal node left)
    # or expanded (3 internal nodes)
    tree = _tree("((A,B,C,D)speciation,E);", dendropy.TaxonNamespace())
    random.seed(4)
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model='LRF')
    assert _leaves(out) == _leaves(tree)
    assert len(out.internal_nodes()) in (1, 3)


def test_same_seed_gives_same_mutation():
    tree = _tree(
        "(((A,B)speciation,C)speciation,((D,E)duplication,F)speciation);",
        dendropy.TaxonNamespace(),
    )
    random.seed(123)
    out1 = mutateLabeledTree(tree, 2, p_flip=0.3)
    random.seed(123)
    out2 = mutateLabeledTree(tree, 2, p_flip=0.3)
    assert out1.as_string(schema="newick") == out2.as_string(schema="newick")


def test_leaf_set_preserved_over_several_seeds():
    tree = _tree("((A,B,C,D)speciation,E);", dendropy.TaxonNamespace())
    leaves = _leaves(tree)
    for seed in range(10):
        random.seed(seed)
        out = mutateLabeledTree(tree, 2, p_flip=0.0, model='LRF')
        assert _leaves(out) == leaves
        assert out is not tree
