import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


S = "speciation"
D = "duplication"


def make_trees(*newicks):
    """Parse several newick strings sharing one taxon namespace."""
    tns = dendropy.TaxonNamespace()
    return [
        dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=tns)
        for nwk in newicks
    ]


def clade_labels(tree):
    """Map sorted leaf-label tuple of each internal node -> node label."""
    out = {}
    for nd in tree.internal_nodes():
        key = tuple(sorted(lf.taxon.label for lf in nd.leaf_iter()))
        out[key] = nd.label
    return out


def leaf_names(tree):
    return sorted(nd.taxon.label for nd in tree.leaf_node_iter())


T4 = "[&R] ((A,B)speciation,(C,D)speciation)duplication;"
T5 = "[&R] (((A,B)speciation,C)duplication,(D,E)speciation)duplication;"
T5_DE_FLIPPED = "[&R] (((A,B)speciation,C)duplication,(D,E)duplication)duplication;"
T6_ALL_S = (
    "[&R] (((A,B)speciation,(C,D)speciation)speciation,(E,F)speciation)speciation;"
)


# ---------------------------------------------------------------- computeLRF

def test_identical_trees_distance_zero():
    random.seed(0)
    t1, t2 = make_trees(T4, T4)
    assert computeLRF(t1, t2) == 0


def test_identical_larger_trees_distance_zero():
    random.seed(0)
    t1, t2 = make_trees(T5, T5)
    assert computeLRF(t1, t2) == 0


def test_rotated_children_distance_zero():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] ((D,C)speciation,(B,A)speciation)duplication;",
    )
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] ((A,B)duplication,(C,D)speciation)duplication;",
    )
    assert computeLRF(t1, t2) == 1


def test_single_label_difference_in_larger_tree():
    random.seed(0)
    t1, t2 = make_trees(T5, T5_DE_FLIPPED)
    assert computeLRF(t1, t2) == 1


def test_two_label_differences_cost_two():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] ((A,B)duplication,(C,D)duplication)duplication;",
    )
    assert computeLRF(t1, t2) == 2


def test_root_label_difference_costs_one():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] ((A,B)speciation,(C,D)speciation)speciation;",
    )
    assert computeLRF(t1, t2) == 1


def test_one_collapsed_edge_costs_one_and_is_symmetric():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] (A,B,(C,D)speciation)duplication;",
    )
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_star_versus_fully_resolved_costs_two():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] (A,B,C,D)duplication;",
        "[&R] ((A,B)duplication,(C,D)duplication)duplication;",
    )
    assert computeLRF(t1, t2) == 2
    assert computeLRF(t2, t1) == 2


def test_different_topologies_no_shared_edges():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] ((A,C)speciation,(B,D)speciation)duplication;",
    )
    assert computeLRF(t1, t2) == 4
    assert computeLRF(t2, t1) == 4


def test_star_trees_same_and_different_labels():
    random.seed(0)
    a, b, c = make_trees(
        "[&R] (A,B,C,D)duplication;",
        "[&R] (A,B,C,D)duplication;",
        "[&R] (A,B,C,D)speciation;",
    )
    assert computeLRF(a, b) == 0
    assert computeLRF(a, c) == 1


def test_computeLRF_does_not_modify_inputs():
    random.seed(0)
    t1, t2 = make_trees(
        "[&R] ((A,B)speciation,(C,D)speciation)duplication;",
        "[&R] (A,B,(C,D)duplication)duplication;",
    )
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# ---------------------------------------------------------- mutateLabeledTree

def test_mutate_zero_edits_returns_equal_copy():
    random.seed(1)
    (t,) = make_trees(T5)
    m = mutateLabeledTree(t, 0)
    assert m is not t
    assert m.as_string("newick") == t.as_string("newick")


def test_mutate_leaves_input_unchanged():
    random.seed(2)
    (t,) = make_trees(T5)
    before = t.as_string("newick")
    for p in (0.0, 0.5, 1.0):
        mutateLabeledTree(t, 4, p_flip=p)
        assert t.as_string("newick") == before


def test_mutate_flip_only_changes_labels_not_topology():
    random.seed(3)
    (t,) = make_trees(T5)
    m = mutateLabeledTree(t, 1, p_flip=1.0)
    orig = clade_labels(t)
    new = clade_labels(m)
    assert set(orig) == set(new)
    diff = [k for k in orig if orig[k] != new[k]]
    assert len(diff) == 1
    assert {orig[diff[0]], new[diff[0]]} == {S, D}


def test_mutate_flip_only_distance_matches_number_of_changed_labels():
    random.seed(4)
    (t,) = make_trees(T5)
    n = 3
    m = mutateLabeledTree(t, n, p_flip=1.0)
    orig = clade_labels(t)
    new = clade_labels(m)
    assert set(orig) == set(new)
    d = sum(1 for k in orig if orig[k] != new[k])
    # each flip toggles exactly one node: d <= n and d has the parity of n
    assert d <= n
    assert d % 2 == n % 2
    # same topology, so LRF is just the number of differing labels
    assert computeLRF(t, m) == d


def test_mutate_flip_without_labels_raises_value_error():
    random.seed(5)
    (t,) = make_trees("[&R] ((A,B),(C,D));")
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=1.0)


def test_mutate_lrf_preserves_leaves_and_changes_size_consistently():
    random.seed(6)
    (t,) = make_trees(T5)
    n_internal = len(t.internal_nodes())
    n = 5
    for seed in range(6):
        random.seed(seed)
        m = mutateLabeledTree(t, n, p_flip=0.0, model="LRF")
        assert leaf_names(m) == ["A", "B", "C", "D", "E"]
        delta = len(m.internal_nodes()) - n_internal
        # each edit is a single collapse (-1) or expansion (+1)
        assert abs(delta) <= n
        assert delta % 2 == n % 2
        assert len(m.internal_nodes()) >= 1


def test_mutate_elrf_without_flips_keeps_consistent_labels():
    random.seed(7)
    (t,) = make_trees(T6_ALL_S)
    for seed in range(6):
        random.seed(seed)
        m = mutateLabeledTree(t, 5, p_flip=0.0, model="ELRF")
        assert leaf_names(m) == ["A", "B", "C", "D", "E", "F"]
        assert all(nd.label == S for nd in m.internal_nodes())


def test_mutate_same_seed_gives_same_result():
    (t,) = make_trees(T5)
    random.seed(1234)
    m1 = mutateLabeledTree(t, 6, p_flip=0.3)
    random.seed(1234)
    m2 = mutateLabeledTree(t, 6, p_flip=0.3)
    assert m1.as_string("newick") == m2.as_string("newick")


def test_mutate_unsupported_model_raises_value_error():
    random.seed(8)
    (t,) = make_trees(T6_ALL_S)
    with pytest.raises(ValueError, match="unsupported model"):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")
