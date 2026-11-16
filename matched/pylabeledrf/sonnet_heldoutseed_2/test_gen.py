import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def parse(newick, tns):
    return dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns)


def parse_pair(nw1, nw2):
    tns = dendropy.TaxonNamespace()
    return parse(nw1, tns), parse(nw2, tns)


def label_map(tree):
    return {
        frozenset(l.taxon.label for l in n.leaf_iter()): n.label
        for n in tree.internal_nodes()
    }


def leaf_names(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


BASE5 = "[&U] ((A,B)speciation,(C,D)speciation,E);"


def test_identical_unrooted_trees_distance_zero():
    t1, t2 = parse_pair(BASE5, BASE5)
    assert computeLRF(t1, t2) == 0


def test_identical_rooted_trees_distance_zero():
    nw = "[&R] ((A,B)speciation,(C,D)duplication);"
    t1, t2 = parse_pair(nw, nw)
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    t1, t2 = parse_pair(
        BASE5, "[&U] ((A,B)duplication,(C,D)speciation,E);"
    )
    assert computeLRF(t1, t2) == 1


def test_two_label_differences_cost_two():
    t1, t2 = parse_pair(
        BASE5, "[&U] ((A,B)duplication,(C,D)duplication,E);"
    )
    assert computeLRF(t1, t2) == 2


def test_rooted_label_difference_costs_one():
    t1, t2 = parse_pair(
        "[&R] ((A,B)speciation,(C,D)speciation);",
        "[&R] ((A,B)duplication,(C,D)speciation);",
    )
    assert computeLRF(t1, t2) == 1


def test_child_rotation_does_not_change_distance():
    t1, t2 = parse_pair(
        BASE5, "[&U] ((C,D)speciation,(A,B)speciation,E);"
    )
    assert computeLRF(t1, t2) == 0


def test_single_collapse_costs_one_and_is_symmetric():
    t1, t2 = parse_pair(
        BASE5, "[&U] (A,B,(C,D)speciation,E);"
    )
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_star_trees_distance_zero():
    t1, t2 = parse_pair("[&U] (A,B,C,D);", "[&U] (A,B,C,D);")
    assert computeLRF(t1, t2) == 0


def test_star_versus_tree_with_one_internal_node():
    t1, t2 = parse_pair("[&U] (A,B,C,D);", "[&U] ((A,B)speciation,C,D);")
    assert computeLRF(t1, t2) == 1


def test_nested_tree_single_substitution():
    nw1 = "[&U] (((A,B)speciation,C)duplication,(D,E)speciation,F);"
    nw2 = "[&U] (((A,B)speciation,C)speciation,(D,E)speciation,F);"
    t1, t2 = parse_pair(nw1, nw1)
    assert computeLRF(t1, t2) == 0
    t3, t4 = parse_pair(nw1, nw2)
    assert computeLRF(t3, t4) == 1


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = parse_pair(
        BASE5, "[&U] (A,B,(C,D)duplication,E);"
    )
    s1 = t1.as_string("newick")
    s2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == s1
    assert t2.as_string("newick") == s2


def test_mutate_zero_edits_returns_equal_copy():
    tree = parse(BASE5, dendropy.TaxonNamespace())
    before = tree.as_string("newick")
    out = mutateLabeledTree(tree, 0)
    assert out is not tree
    assert out.as_string("newick") == before
    assert tree.as_string("newick") == before


def test_mutate_does_not_modify_input_tree():
    tree = parse("[&U] ((A,B)speciation,(C,D)duplication,E,F);", dendropy.TaxonNamespace())
    before = tree.as_string("newick")
    random.seed(1)
    mutateLabeledTree(tree, 3, p_flip=0.0)
    assert tree.as_string("newick") == before


def test_mutate_single_flip_changes_exactly_one_label():
    tree = parse("[&U] ((A,B)speciation,(C,D)duplication,E);", dendropy.TaxonNamespace())
    orig = label_map(tree)
    random.seed(42)
    out = mutateLabeledTree(tree, 1, p_flip=1.0)
    new = label_map(out)
    assert set(new) == set(orig)
    diffs = [k for k in orig if orig[k] != new[k]]
    assert len(diffs) == 1
    assert new[frozenset("ABCDE")] is None
    assert computeLRF(tree, out) == 1


def test_mutate_two_flips_on_single_labeled_node_restore_label():
    tree = parse("[&U] ((A,B)speciation,C,D,E);", dendropy.TaxonNamespace())
    random.seed(3)
    once = mutateLabeledTree(tree, 1, p_flip=1.0)
    assert label_map(once)[frozenset("AB")] == "duplication"
    random.seed(3)
    twice = mutateLabeledTree(tree, 2, p_flip=1.0)
    assert label_map(twice)[frozenset("AB")] == "speciation"


def test_mutate_flip_without_labels_raises_value_error():
    tree = parse("[&U] ((A,B),(C,D),E);", dendropy.TaxonNamespace())
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=1.0)


def test_mutate_topology_edits_preserve_leaf_set():
    nw = "[&U] ((A,B)speciation,(C,D)duplication,(E,F)speciation,G);"
    for seed in range(8):
        tree = parse(nw, dendropy.TaxonNamespace())
        random.seed(seed)
        out = mutateLabeledTree(tree, 4, p_flip=0.0)
        assert leaf_names(out) == leaf_names(tree)


def test_mutate_lrf_single_edit_changes_internal_node_count_by_one():
    nw = "[&U] ((A,B)speciation,(C,D)speciation,E,F);"
    for seed in range(6):
        tree = parse(nw, dendropy.TaxonNamespace())
        random.seed(seed)
        out = mutateLabeledTree(tree, 1, p_flip=0.0, model='LRF')
        diff = len(out.internal_nodes()) - len(tree.internal_nodes())
        assert abs(diff) == 1
        assert leaf_names(out) == leaf_names(tree)


def test_mutate_elrf_expands_root_when_no_valid_collapse():
    tree = parse("[&U] ((A,B)speciation,(C,D)speciation,E,F);", dendropy.TaxonNamespace())
    random.seed(5)
    out = mutateLabeledTree(tree, 1, p_flip=0.0, model='ELRF')
    assert len(out.internal_nodes()) == len(tree.internal_nodes()) + 1
    assert leaf_names(out) == leaf_names(tree)


def test_mutate_unsupported_model_raises_value_error():
    tree = parse("[&U] ((A,B)speciation,(C,D)speciation,E,F);", dendropy.TaxonNamespace())
    random.seed(0)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model='BOGUS')
