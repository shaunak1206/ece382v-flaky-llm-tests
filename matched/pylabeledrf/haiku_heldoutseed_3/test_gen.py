import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def make_tree(newick, labels, taxa):
    """Parse a rooted newick tree and assign labels to internal nodes.

    ``labels`` maps frozensets of leaf names to 'speciation'/'duplication'.
    Internal nodes not present in the mapping (including the root) get None.
    """
    tree = dendropy.Tree.get(
        data=newick, schema="newick", rooted=True, taxon_namespace=taxa
    )
    for node in tree.internal_nodes():
        leafset = frozenset(leaf.taxon.label for leaf in node.leaf_iter())
        node.label = labels.get(leafset)
    return tree


def label_map(tree):
    """Map each internal node's leaf set to its label."""
    return {
        frozenset(leaf.taxon.label for leaf in node.leaf_iter()): node.label
        for node in tree.internal_nodes()
    }


def leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_nodes())


S = "speciation"
D = "duplication"


def test_identical_trees_have_zero_distance():
    taxa = dendropy.TaxonNamespace()
    labels = {frozenset("AB"): S, frozenset("CD"): D}
    t1 = make_tree("((A,B),(C,D));", labels, taxa)
    t2 = make_tree("((A,B),(C,D));", labels, taxa)
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one_substitution():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    t2 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): S}, taxa)
    assert computeLRF(t1, t2) == 1


def test_two_label_differences_cost_two():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D),(E,F));",
                   {frozenset("AB"): S, frozenset("CD"): D, frozenset("EF"): S},
                   taxa)
    t2 = make_tree("((A,B),(C,D),(E,F));",
                   {frozenset("AB"): D, frozenset("CD"): S, frozenset("EF"): S},
                   taxa)
    assert computeLRF(t1, t2) == 2


def test_completely_different_topologies_four_taxa():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    t2 = make_tree("((A,C),(B,D));",
                   {frozenset("AC"): S, frozenset("BD"): D}, taxa)
    # no shared good edges: two stars of size 3 each, 3 + 3 - 2 = 4 collapses
    assert computeLRF(t1, t2) == 4


def test_one_collapse_costs_one():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    t2 = make_tree("(A,B,(C,D));",
                   {frozenset("CD"): D}, taxa)
    assert computeLRF(t1, t2) == 1


def test_distance_is_symmetric():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    t2 = make_tree("(A,B,(C,D));",
                   {frozenset("CD"): D}, taxa)
    assert computeLRF(t1, t2) == computeLRF(t2, t1) == 1


def test_computeLRF_does_not_modify_inputs():
    taxa = dendropy.TaxonNamespace()
    t1 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    t2 = make_tree("((A,B),(C,D));",
                   {frozenset("AB"): S, frozenset("CD"): S}, taxa)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    labels_before = label_map(t1), label_map(t2)
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2
    assert (label_map(t1), label_map(t2)) == labels_before


def test_single_flip_mutation_gives_distance_one():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D));",
                     {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    random.seed(0)
    mutated = mutateLabeledTree(orig, 1, p_flip=1.0)
    assert computeLRF(orig, mutated) == 1


def test_mutate_returns_new_tree_and_leaves_input_unchanged():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D));",
                     {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    before = orig.as_string("newick")
    random.seed(1)
    out = mutateLabeledTree(orig, 2, p_flip=0.5)
    assert out is not orig
    assert orig.as_string("newick") == before


def test_mutate_zero_edits_preserves_tree():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D));",
                     {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    out = mutateLabeledTree(orig, 0)
    assert label_map(out) == label_map(orig)
    assert leaf_names(out) == leaf_names(orig)
    assert computeLRF(orig, out) == 0


def test_mutate_flip_only_changes_exactly_one_label():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D));",
                     {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    random.seed(2)
    out = mutateLabeledTree(orig, 1, p_flip=1.0)
    before = label_map(orig)
    after = label_map(out)
    assert set(before) == set(after)  # topology unchanged
    differing = [k for k in before if before[k] != after[k]]
    assert len(differing) == 1
    assert before[differing[0]] in (S, D)
    assert after[differing[0]] in (S, D)
    assert before[differing[0]] != after[differing[0]]


def test_mutate_collapse_only_removes_one_internal_node():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D));",
                     {frozenset("AB"): S, frozenset("CD"): D}, taxa)
    random.seed(3)
    out = mutateLabeledTree(orig, 1, p_flip=0.0)
    assert len(list(out.internal_nodes())) == len(list(orig.internal_nodes())) - 1
    assert leaf_names(out) == leaf_names(orig)
    assert computeLRF(orig, out) == 1


def test_mutate_expansion_only_adds_one_internal_node():
    taxa = dendropy.TaxonNamespace()
    # only the root (degree 3) can be expanded; there are no labels to flip
    orig = make_tree("(A,B,C);", {}, taxa)
    random.seed(4)
    out = mutateLabeledTree(orig, 1, p_flip=0.0)
    assert len(list(out.internal_nodes())) == len(list(orig.internal_nodes())) + 1
    assert leaf_names(out) == ["A", "B", "C"]
    assert len(out.seed_node.child_nodes()) == 2


def test_mutate_is_reproducible_with_same_seed():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D),(E,F));",
                     {frozenset("AB"): S, frozenset("CD"): D, frozenset("EF"): S},
                     taxa)
    random.seed(1234)
    first = mutateLabeledTree(orig, 2).as_string("newick")
    random.seed(1234)
    second = mutateLabeledTree(orig, 2).as_string("newick")
    assert first == second


def test_mutate_preserves_leaf_set_across_seeds():
    taxa = dendropy.TaxonNamespace()
    orig = make_tree("((A,B),(C,D),(E,F));",
                     {frozenset("AB"): S, frozenset("CD"): D, frozenset("EF"): S},
                     taxa)
    for seed in range(10):
        random.seed(seed)
        out = mutateLabeledTree(orig, 2, p_flip=0.3)
        assert leaf_names(out) == ["A", "B", "C", "D", "E", "F"]
