import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def _parse(newick, taxa):
    return dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)


def _pair(n1, n2):
    taxa = dendropy.TaxonNamespace()
    return _parse(n1, taxa), _parse(n2, taxa)


def _labels(tree):
    return [n.label for n in tree.preorder_node_iter()]


def _leaf_names(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


def _num_internal(tree):
    return len(list(tree.internal_nodes()))


SIX = "[&U] ((A,B)speciation,(C,D)speciation,(E,F)speciation);"
SIX_MIXED = "[&U] ((A,B)speciation,(C,D)duplication,(E,F)speciation);"


def test_identical_trees_distance_zero():
    t1, t2 = _pair(SIX, SIX)
    assert computeLRF(t1, t2) == 0


def test_identical_star_trees_distance_zero():
    t1, t2 = _pair("[&U] (A,B,C,D,E);", "[&U] (A,B,C,D,E);")
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    t1, t2 = _pair(
        "[&U] ((A,B)speciation,(C,D)speciation,(E,F)speciation);",
        "[&U] ((A,B)duplication,(C,D)speciation,(E,F)speciation);",
    )
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_single_collapsed_edge_costs_one():
    t1, t2 = _pair(
        "[&U] ((A,B)speciation,(C,D)speciation,(E,F)speciation);",
        "[&U] ((A,B)speciation,C,D,(E,F)speciation);",
    )
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_different_topology_is_classic_rf():
    t1, t2 = _pair(
        "[&U] ((A,B)speciation,(C,D)speciation,(E,F)speciation);",
        "[&U] ((A,C)speciation,(B,D)speciation,(E,F)speciation);",
    )
    assert computeLRF(t1, t2) == 4
    assert computeLRF(t2, t1) == 4


def test_no_shared_edges_four_taxa():
    t1, t2 = _pair(
        "[&U] (A,B,(C,D)speciation);",
        "[&U] (A,C,(B,D)speciation);",
    )
    assert computeLRF(t1, t2) == 2


def test_label_difference_on_only_edge():
    t1, t2 = _pair(
        "[&U] (A,B,(C,D)speciation);",
        "[&U] (A,B,(C,D)duplication);",
    )
    assert computeLRF(t1, t2) == 1


def test_distance_is_symmetric_and_integer():
    t1, t2 = _pair(
        "[&U] ((A,B)speciation,(C,D)duplication,(E,F)speciation);",
        "[&U] ((A,C)speciation,(B,D)speciation,(E,F)duplication);",
    )
    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)
    assert isinstance(d12, int)
    assert d12 >= 0
    assert d12 == d21


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = _pair(
        "[&U] ((A,B)speciation,(C,D)speciation,(E,F)speciation);",
        "[&U] ((A,B)duplication,C,D,(E,F)speciation);",
    )
    s1 = t1.as_string("newick")
    s2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == s1
    assert t2.as_string("newick") == s2


def test_mutate_zero_edits_returns_equal_copy():
    random.seed(1)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(SIX_MIXED, taxa)
    out = mutateLabeledTree(tree, 0)
    assert out is not tree
    assert _labels(out) == _labels(tree)
    assert _leaf_names(out) == _leaf_names(tree)
    assert computeLRF(tree, out) == 0


def test_mutate_leaves_input_unchanged():
    random.seed(2)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(SIX_MIXED, taxa)
    before = tree.as_string("newick")
    out = mutateLabeledTree(tree, 3)
    assert out is not tree
    assert tree.as_string("newick") == before


def test_mutate_only_flips_changes_exactly_one_label():
    random.seed(3)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(SIX_MIXED, taxa)
    out = mutateLabeledTree(tree, 1, p_flip=1.0)
    diffs = [(a, b) for a, b in zip(_labels(tree), _labels(out)) if a != b]
    assert len(_labels(out)) == len(_labels(tree))
    assert len(diffs) == 1
    assert set(diffs[0]) == {"speciation", "duplication"}
    assert computeLRF(tree, out) == 1


def test_mutate_two_flips_change_even_number_of_labels():
    random.seed(4)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(SIX_MIXED, taxa)
    out = mutateLabeledTree(tree, 2, p_flip=1.0)
    ndiff = sum(1 for a, b in zip(_labels(tree), _labels(out)) if a != b)
    assert ndiff in (0, 2)


def test_mutate_lrf_collapse_only_on_cherries():
    random.seed(5)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(SIX_MIXED, taxa)
    # root is skipped and cherries cannot be expanded, so a collapse is forced
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model="LRF")
    assert _num_internal(tree) == 4
    assert _num_internal(out) == 3
    assert _leaf_names(out) == _leaf_names(tree)
    assert computeLRF(tree, out) == 1


def test_mutate_elrf_collapses_edge_with_same_label_as_parent():
    random.seed(6)
    taxa = dendropy.TaxonNamespace()
    tree = _parse("[&U] ((A,B)speciation,C,D)speciation;", taxa)
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
    assert _num_internal(out) == 1
    assert _leaf_names(out) == ["A", "B", "C", "D"]
    assert out.seed_node.label == "speciation"


def test_mutate_unsupported_model_raises():
    random.seed(7)
    taxa = dendropy.TaxonNamespace()
    tree = _parse("[&U] ((A,B)speciation,C,D)speciation;", taxa)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="XYZ")


def test_mutate_flip_without_labels_raises():
    random.seed(8)
    taxa = dendropy.TaxonNamespace()
    tree = _parse("[&U] ((A,B),(C,D),(E,F));", taxa)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=1.0)


def test_mutate_same_seed_gives_same_result():
    random.seed(123)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(
        "[&U] ((A,B)speciation,(C,D)duplication,(E,F)speciation,(G,H)duplication)speciation;",
        taxa,
    )
    random.seed(123)
    out1 = mutateLabeledTree(tree, 5)
    random.seed(123)
    out2 = mutateLabeledTree(tree, 5)
    assert out1.as_string("newick") == out2.as_string("newick")


def test_mutate_preserves_leaves_for_many_seeds():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    tree = _parse(
        "[&U] ((A,B)speciation,(C,D)duplication,(E,F)speciation,(G,H)duplication)speciation;",
        taxa,
    )
    before = tree.as_string("newick")
    expected = _leaf_names(tree)
    for seed in range(15):
        random.seed(seed)
        out = mutateLabeledTree(tree, 3)
        assert _leaf_names(out) == expected
    assert tree.as_string("newick") == before
