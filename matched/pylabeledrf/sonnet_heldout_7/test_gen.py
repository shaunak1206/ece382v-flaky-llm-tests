import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def _parse(*newicks):
    """Parse several newick strings into trees sharing one taxon namespace."""
    ns = dendropy.TaxonNamespace()
    return [
        dendropy.Tree.get(data=s, schema="newick", taxon_namespace=ns)
        for s in newicks
    ]


def _internal_labels(t):
    return [n.label for n in t.preorder_node_iter() if not n.is_leaf()]


def _leaf_names(t):
    return sorted(n.taxon.label for n in t.leaf_node_iter())


BASE = "[&U] ((a,b)speciation,c,(d,e)duplication);"
BASE_SS = "[&U] ((a,b)speciation,c,(d,e)speciation);"


# ---------------------------------------------------------------- computeLRF

def test_identical_trees_have_zero_distance():
    t1, t2 = _parse(BASE, BASE)
    assert computeLRF(t1, t2) == 0


def test_identical_larger_nested_trees_zero_distance():
    s = "[&U] (((a,b)speciation,c)duplication,(d,e)speciation,(f,g)speciation);"
    t1, t2 = _parse(s, s)
    assert computeLRF(t1, t2) == 0


def test_same_tree_different_child_order_is_zero():
    t1, t2 = _parse(
        BASE,
        "[&U] ((e,d)duplication,c,(b,a)speciation);",
    )
    assert computeLRF(t1, t2) == 0


def test_single_label_flip_costs_one():
    t1, t2 = _parse(
        BASE,
        "[&U] ((a,b)duplication,c,(d,e)duplication);",
    )
    assert computeLRF(t1, t2) == 1


def test_two_label_flips_cost_two():
    t1, t2 = _parse(
        BASE,
        "[&U] ((a,b)duplication,c,(d,e)speciation);",
    )
    assert computeLRF(t1, t2) == 2


def test_label_flip_in_nested_tree_costs_one():
    t1, t2 = _parse(
        "[&U] (((a,b)speciation,c)duplication,(d,e)speciation,(f,g)speciation);",
        "[&U] (((a,b)speciation,c)speciation,(d,e)speciation,(f,g)speciation);",
    )
    assert computeLRF(t1, t2) == 1


def test_collapse_against_star_tree():
    # two internal edges must be collapsed to reach the star tree
    t1, t2 = _parse(BASE_SS, "[&U] (a,b,c,d,e);")
    assert computeLRF(t1, t2) == 2


def test_collapse_distance_is_symmetric():
    t1, t2 = _parse(BASE_SS, "[&U] (a,b,c,d,e);")
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_single_collapse_costs_one():
    t1, t2 = _parse(BASE_SS, "[&U] (a,b,c,(d,e)speciation);")
    assert computeLRF(t1, t2) == 1


def test_different_topology_same_labels():
    t1, t2 = _parse(
        BASE_SS,
        "[&U] ((a,c)speciation,b,(d,e)speciation);",
    )
    assert computeLRF(t1, t2) == 2
    assert computeLRF(t2, t1) == 2


def test_symmetry_on_label_difference():
    t1, t2 = _parse(
        BASE,
        "[&U] ((a,b)duplication,c,(d,e)duplication);",
    )
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = _parse(BASE_SS, "[&U] ((a,c)duplication,b,(d,e)speciation);")
    before1 = (_internal_labels(t1), len(t1.internal_nodes()), t1.as_string("newick"))
    before2 = (_internal_labels(t2), len(t2.internal_nodes()), t2.as_string("newick"))
    computeLRF(t1, t2)
    assert (_internal_labels(t1), len(t1.internal_nodes()), t1.as_string("newick")) == before1
    assert (_internal_labels(t2), len(t2.internal_nodes()), t2.as_string("newick")) == before2


# ---------------------------------------------------------- mutateLabeledTree

def test_mutate_zero_edits_returns_equal_copy():
    (t,) = _parse(BASE)
    m = mutateLabeledTree(t, 0)
    assert m is not t
    assert _internal_labels(m) == _internal_labels(t)
    assert len(m.internal_nodes()) == len(t.internal_nodes())
    assert computeLRF(t, m) == 0


def test_mutate_does_not_modify_input():
    (t,) = _parse("[&U] ((a,b)speciation,c,d,e,f);")
    before = (_internal_labels(t), len(t.internal_nodes()), len(t.leaf_nodes()))
    random.seed(1)
    mutateLabeledTree(t, 3, p_flip=0.5)
    after = (_internal_labels(t), len(t.internal_nodes()), len(t.leaf_nodes()))
    assert before == after


def test_mutate_pure_flip_changes_exactly_one_label():
    (t,) = _parse(BASE)
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=1.0)
        assert len(m.internal_nodes()) == len(t.internal_nodes())
        diffs = [
            1 for x, y in zip(_internal_labels(t), _internal_labels(m)) if x != y
        ]
        assert len(diffs) == 1
        # root label (None) must never be flipped
        assert _internal_labels(m)[0] is None
        # a single flip is exactly one LRF edit
        assert computeLRF(t, m) == 1


def test_mutate_odd_number_of_flips_leaves_odd_number_of_differences():
    # only two labelled nodes exist, so 3 flips leave exactly 1 node differing
    (t,) = _parse(BASE)
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 3, p_flip=1.0)
        diffs = [
            1 for x, y in zip(_internal_labels(t), _internal_labels(m)) if x != y
        ]
        assert len(diffs) == 1


def test_mutate_flip_only_swaps_between_valid_labels():
    (t,) = _parse(BASE)
    random.seed(7)
    m = mutateLabeledTree(t, 5, p_flip=1.0)
    for lab in _internal_labels(m)[1:]:
        assert lab in ("speciation", "duplication")


def test_mutate_flip_without_labels_raises():
    (t,) = _parse("[&U] ((a,b),c,(d,e));")
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=1.0)


def test_mutate_collapse_reduces_internal_nodes():
    (t,) = _parse(BASE_SS)
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
        assert len(m.internal_nodes()) == len(t.internal_nodes()) - 1
        assert _leaf_names(m) == _leaf_names(t)
        assert computeLRF(t, m) == 1


def test_mutate_expand_star_tree_adds_internal_node():
    (t,) = _parse("[&U] (a,b,c,d);")
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0)
        assert len(m.internal_nodes()) == 2
        assert _leaf_names(m) == ["a", "b", "c", "d"]
    assert len(t.internal_nodes()) == 1


def test_mutate_collapse_or_expand_preserves_leaves():
    (t,) = _parse("[&U] ((a,b)speciation,c,d,e,f);")
    for seed in range(20):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0)
        assert _leaf_names(m) == ["a", "b", "c", "d", "e", "f"]
        # either the (a,b) clade was collapsed or a new node was added
        assert len(m.internal_nodes()) in (1, 3)


def test_mutate_is_reproducible_with_seed():
    (t,) = _parse("[&U] (((a,b)speciation,c)duplication,(d,e)speciation,f,g);")
    random.seed(123)
    m1 = mutateLabeledTree(t, 4, p_flip=0.3)
    random.seed(123)
    m2 = mutateLabeledTree(t, 4, p_flip=0.3)
    assert m1.as_string("newick") == m2.as_string("newick")


def test_mutate_elrf_collapses_only_same_label_edges():
    # Only Y=(a,b) has the same label as its parent X, so it must be collapsed
    (t,) = _parse("[&U] (((a,b)speciation,c)speciation,d,e);")
    random.seed(0)
    m = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert len(m.internal_nodes()) == len(t.internal_nodes()) - 1
    assert _leaf_names(m) == ["a", "b", "c", "d", "e"]


def test_mutate_unsupported_model_raises():
    (t,) = _parse("[&U] (a,b,c,d);")
    with pytest.raises(ValueError, match="unsupported model"):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")
