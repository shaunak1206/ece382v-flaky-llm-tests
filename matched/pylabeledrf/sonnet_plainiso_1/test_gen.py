import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree

T4 = "((A,B),(C,D));"
T6 = "(((A,B),C),((D,E),F));"


def _parse(nwk, tns, labels=None, default="speciation"):
    """Parse a rooted newick string and label the internal nodes.

    The seed node gets no label; every other internal node gets the label
    found in ``labels`` (keyed by the sorted concatenation of its leaves),
    or ``default``.
    """
    tree = dendropy.Tree.get(
        data=nwk, schema="newick", taxon_namespace=tns, rooting="force-rooted"
    )
    labels = labels or {}
    for node in tree.internal_nodes():
        if node is tree.seed_node:
            node.label = None
            continue
        key = "".join(sorted(leaf.taxon.label for leaf in node.leaf_iter()))
        node.label = labels.get(key, default)
    return tree


def _pair(nwk1, nwk2, labels1=None, labels2=None, default1="speciation",
          default2="speciation"):
    tns = dendropy.TaxonNamespace()
    t1 = _parse(nwk1, tns, labels1, default1)
    t2 = _parse(nwk2, tns, labels2, default2)
    return t1, t2


def _signature(tree):
    sig = []
    for node in tree.internal_nodes():
        key = "".join(sorted(leaf.taxon.label for leaf in node.leaf_iter()))
        sig.append((key, node.label))
    return sorted(sig, key=lambda x: (x[0], str(x[1])))


def _labels(tree):
    return [node.label for node in tree.internal_nodes()]


def _leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_node_iter())


def _single(nwk, labels=None, default="speciation"):
    return _parse(nwk, dendropy.TaxonNamespace(), labels, default)


# ---------------------------------------------------------------- computeLRF

def test_lrf_identical_balanced_trees_is_zero():
    t1, t2 = _pair(T4, T4)
    assert computeLRF(t1, t2) == 0


def test_lrf_identical_mixed_labels_is_zero():
    labels = {"ABC": "duplication", "DE": "duplication"}
    t1, t2 = _pair(T6, T6, labels, labels)
    assert computeLRF(t1, t2) == 0


def test_lrf_single_label_difference_is_one():
    t1, t2 = _pair(T4, T4, labels2={"AB": "duplication"})
    assert computeLRF(t1, t2) == 1


def test_lrf_all_internal_labels_flipped_counts_each_node():
    t1, t2 = _pair(T6, T6, default1="speciation", default2="duplication")
    # four non-root internal nodes: ABC, AB, DEF, DE
    assert computeLRF(t1, t2) == 4


def test_lrf_child_order_does_not_matter():
    labels = {"ABC": "duplication", "DE": "duplication"}
    t1, t2 = _pair(T6, "((F,(E,D)),(C,(B,A)));", labels, labels)
    assert computeLRF(t1, t2) == 0


def test_lrf_no_common_edges_uses_single_island():
    t1, t2 = _pair(T4, "((A,C),(B,D));")
    assert computeLRF(t1, t2) == 4


def test_lrf_star_vs_resolved_tree_counts_collapsed_edges():
    t1, t2 = _pair(T4, "(A,B,C,D);")
    assert computeLRF(t1, t2) == 2


def test_lrf_one_collapsed_edge_costs_one():
    t1, t2 = _pair("(((A,B),C),(D,E,F));", "((A,B,C),(D,E,F));")
    assert computeLRF(t1, t2) == 1


def test_lrf_collapse_plus_substitution():
    t1, t2 = _pair(
        "(((A,B),C),(D,E,F));",
        "((A,B,C),(D,E,F));",
        labels1={"ABC": "duplication", "AB": "duplication"},
    )
    assert computeLRF(t1, t2) == 2


def test_lrf_is_symmetric():
    cases = [
        (T4, "((A,C),(B,D));", None, None),
        (T6, T6, {"ABC": "duplication"}, {"DE": "duplication", "AB": "duplication"}),
        ("(((A,B),C),(D,E,F));", "((A,B,C),(D,E,F));", None, None),
        (T4, "(A,B,C,D);", None, None),
    ]
    for n1, n2, l1, l2 in cases:
        a, b = _pair(n1, n2, l1, l2)
        assert computeLRF(a, b) == computeLRF(b, a)


def test_lrf_does_not_modify_inputs():
    t1, t2 = _pair(
        "(((A,B),C),(D,E,F));",
        "((A,B,C),(D,E,F));",
        labels1={"ABC": "duplication"},
    )
    before1, before2 = _signature(t1), _signature(t2)
    computeLRF(t1, t2)
    assert _signature(t1) == before1
    assert _signature(t2) == before2


def test_lrf_returns_nonnegative_int():
    t1, t2 = _pair(T6, "(((A,D),C),((B,E),F));")
    d = computeLRF(t1, t2)
    assert isinstance(d, int)
    assert d >= 0


# ------------------------------------------------------------ mutateLabeledTree

def test_mutate_zero_edits_gives_equivalent_tree():
    tree = _single(T6, {"ABC": "duplication"})
    result = mutateLabeledTree(tree, 0)
    assert result is not tree
    assert _signature(result) == _signature(tree)
    assert computeLRF(tree, result) == 0


def test_mutate_does_not_modify_input_tree():
    tree = _single(T6, {"ABC": "duplication", "DE": "duplication"})
    before = _signature(tree)
    random.seed(1)
    mutateLabeledTree(tree, 5, p_flip=0.0)
    random.seed(2)
    mutateLabeledTree(tree, 5, p_flip=1.0)
    assert _signature(tree) == before


def test_mutate_single_flip_changes_exactly_one_label():
    tree = _single(T6)
    random.seed(7)
    result = mutateLabeledTree(tree, 1, p_flip=1.0)
    diffs = sum(1 for a, b in zip(_labels(tree), _labels(result)) if a != b)
    assert diffs == 1
    assert len(result.internal_nodes()) == len(tree.internal_nodes())
    assert computeLRF(tree, result) == 1


def test_mutate_flips_keep_topology_and_valid_labels():
    tree = _single(T6)
    random.seed(3)
    result = mutateLabeledTree(tree, 3, p_flip=1.0)
    assert result.seed_node.label is None
    for node in result.internal_nodes():
        if node is not result.seed_node:
            assert node.label in ("speciation", "duplication")
    diffs = sum(1 for a, b in zip(_labels(tree), _labels(result)) if a != b)
    assert 0 <= diffs <= 3
    assert computeLRF(tree, result) == diffs


def test_mutate_collapse_removes_one_internal_node_per_edit():
    tree = _single(T4)
    random.seed(5)
    result = mutateLabeledTree(tree, 1, p_flip=0.0)
    assert len(tree.internal_nodes()) == 3
    assert len(result.internal_nodes()) == 2
    assert _leaf_names(result) == ["A", "B", "C", "D"]


def test_mutate_two_collapses_on_larger_tree():
    tree = _single(T6)
    random.seed(11)
    result = mutateLabeledTree(tree, 2, p_flip=0.0)
    assert len(tree.internal_nodes()) == 5
    assert len(result.internal_nodes()) == 3
    assert _leaf_names(result) == ["A", "B", "C", "D", "E", "F"]


def test_mutate_collapse_and_expand_preserve_leaves():
    tree = _single(T6, {"ABC": "duplication"})
    for seed in range(5):
        random.seed(seed)
        result = mutateLabeledTree(tree, 6, p_flip=0.0)
        assert _leaf_names(result) == ["A", "B", "C", "D", "E", "F"]
        for node in result.internal_nodes():
            assert len(node.child_nodes()) >= 2


def test_mutate_elrf_only_collapses_edges_between_same_labels():
    tree = _single(T6)
    random.seed(4)
    result = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
    assert len(result.internal_nodes()) == len(tree.internal_nodes()) - 1
    for node in result.internal_nodes():
        if node is not result.seed_node:
            assert node.label == "speciation"


def test_mutate_elrf_without_candidates_raises_value_error():
    tree = _single(T4)
    random.seed(0)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")


def test_mutate_unsupported_model_raises_value_error():
    tree = _single(T6)
    random.seed(0)
    with pytest.raises(ValueError, match="unsupported model"):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="XYZ")
