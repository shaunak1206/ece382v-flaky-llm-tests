import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def _parse(*newicks):
    """Parse newick strings into trees sharing one taxon namespace."""
    tns = dendropy.TaxonNamespace()
    return [
        dendropy.Tree.get(data=s, schema="newick", taxon_namespace=tns)
        for s in newicks
    ]


def _labels(tree):
    return [n.label for n in tree.preorder_node_iter()]


def _leaf_names(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


def _clades(tree):
    return set(
        frozenset(l.taxon.label for l in n.leaf_iter())
        for n in tree.internal_nodes()
    )


# ---------------------------------------------------------------- computeLRF

def test_identical_unrooted_trees_have_zero_distance():
    s = "[&U] ((A,B)speciation,(C,D)speciation,E)duplication;"
    t1, t2 = _parse(s, s)
    assert computeLRF(t1, t2) == 0


def test_child_order_does_not_matter():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,(C,D)speciation,E)duplication;",
        "[&U] ((C,D)speciation,(A,B)speciation,E)duplication;",
    )
    assert computeLRF(t1, t2) == 0


def test_identical_rooted_trees_have_zero_distance():
    s = "[&R] (((A,B)speciation,C)speciation,(D,E)speciation)speciation;"
    t1, t2 = _parse(s, s)
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,(C,D)speciation,E)duplication;",
        "[&U] ((A,B)duplication,(C,D)speciation,E)duplication;",
    )
    assert computeLRF(t1, t2) == 1


def test_star_versus_one_resolved_edge_same_labels_costs_one():
    t1, t2 = _parse(
        "[&U] (A,B,C,D)speciation;",
        "[&U] ((A,B)duplication,C,D)speciation;",
    )
    assert computeLRF(t1, t2) == 1


def test_star_versus_one_resolved_edge_different_labels_costs_two():
    t1, t2 = _parse(
        "[&U] (A,B,C,D)speciation;",
        "[&U] ((A,B)duplication,C,D)duplication;",
    )
    assert computeLRF(t1, t2) == 2


def test_conflicting_four_taxon_topologies():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,C,D)speciation;",
        "[&U] ((A,C)speciation,B,D)speciation;",
    )
    assert computeLRF(t1, t2) == 2


def test_fully_conflicting_five_taxon_topologies():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,(C,D)speciation,E)speciation;",
        "[&U] ((A,C)speciation,(B,D)speciation,E)speciation;",
    )
    assert computeLRF(t1, t2) == 4


def test_collapse_plus_flip_is_symmetric():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,(C,D)speciation,E)duplication;",
        "[&U] ((A,B)duplication,C,D,E)duplication;",
    )
    assert computeLRF(t1, t2) == 2
    assert computeLRF(t2, t1) == 2


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = _parse(
        "[&U] ((A,B)speciation,(C,D)speciation,E)duplication;",
        "[&U] ((A,B)duplication,C,D,E)duplication;",
    )
    before1, before2 = _labels(t1), _labels(t2)
    clades1, clades2 = _clades(t1), _clades(t2)
    computeLRF(t1, t2)
    assert _labels(t1) == before1
    assert _labels(t2) == before2
    assert _clades(t1) == clades1
    assert _clades(t2) == clades2


# ---------------------------------------------------------- mutateLabeledTree

def test_zero_edits_returns_equal_copy():
    (t,) = _parse("[&U] ((A,B)speciation,(C,D)duplication,E)speciation;")
    m = mutateLabeledTree(t, 0)
    assert m is not t
    assert _labels(m) == _labels(t)
    assert _clades(m) == _clades(t)


def test_input_tree_is_left_unchanged():
    (t,) = _parse(
        "[&U] (((A,B)speciation,C)duplication,(D,E)speciation,F)speciation;"
    )
    labels, clades = _labels(t), _clades(t)
    random.seed(3)
    mutateLabeledTree(t, 6, p_flip=0.5)
    assert _labels(t) == labels
    assert _clades(t) == clades


def test_forced_flip_changes_only_the_labeled_node():
    (t,) = _parse("[&U] ((A,B)speciation,C,D);")
    m = mutateLabeledTree(t, 1, p_flip=1.0)
    non_root = [n for n in m.internal_nodes() if n.parent_node is not None]
    assert [n.label for n in non_root] == ["duplication"]
    assert m.seed_node.label is None
    assert _clades(m) == _clades(t)


def test_flip_only_edits_keep_topology_and_valid_labels():
    (t,) = _parse("[&U] ((A,B)speciation,(C,D)duplication,E)speciation;")
    random.seed(11)
    m = mutateLabeledTree(t, 5, p_flip=1.0)
    assert _clades(m) == _clades(t)
    assert all(
        n.label in ("speciation", "duplication") for n in m.internal_nodes()
    )


def test_lrf_collapse_removes_one_internal_node():
    (t,) = _parse("[&U] ((A,B)speciation,(C,D)speciation,E)duplication;")
    random.seed(0)
    m = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert len(m.internal_nodes()) == len(t.internal_nodes()) - 1
    assert _leaf_names(m) == _leaf_names(t)


def test_expansion_of_star_adds_an_internal_node():
    (t,) = _parse("[&U] (A,B,C,D,E)speciation;")
    random.seed(5)
    m = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert len(m.internal_nodes()) == 2
    assert _leaf_names(m) == ["A", "B", "C", "D", "E"]
    assert all(n.label == "speciation" for n in m.internal_nodes())


def test_elrf_only_collapses_edges_between_same_labels():
    (t,) = _parse("[&U] ((A,B)speciation,(C,D)duplication,E)speciation;")
    for seed in range(5):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
        non_root = [n for n in m.internal_nodes() if n.parent_node is not None]
        assert [n.label for n in non_root] == ["duplication"]
        assert frozenset("CD") in _clades(m)
        assert frozenset("AB") not in _clades(m)


def test_unsupported_model_raises_value_error():
    (t,) = _parse("[&U] ((A,B)speciation,(C,D)speciation,E)speciation;")
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")


@pytest.mark.parametrize("seed", range(10))
def test_random_edits_preserve_leaves_and_label_validity(seed):
    (t,) = _parse(
        "[&U] (((A,B)speciation,C)duplication,(D,E)speciation,F)speciation;"
    )
    random.seed(seed)
    m = mutateLabeledTree(t, 8, p_flip=0.3)
    assert _leaf_names(m) == _leaf_names(t)
    assert all(
        n.label in ("speciation", "duplication") for n in m.internal_nodes()
    )
