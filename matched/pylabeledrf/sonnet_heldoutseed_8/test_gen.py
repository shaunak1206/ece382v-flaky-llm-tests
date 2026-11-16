import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def make_tree(newick, labels, tns):
    """Build an unrooted tree; `labels` maps frozenset(leaf names) -> label."""
    tree = dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=tns,
        rooting="force-unrooted",
    )
    for node in tree.internal_nodes():
        key = frozenset(leaf.taxon.label for leaf in node.leaf_iter())
        node.label = labels.get(key)
    return tree


def fs(s):
    return frozenset(s)


def signature(tree):
    return sorted(
        (tuple(sorted(l.taxon.label for l in n.leaf_iter())), str(n.label))
        for n in tree.internal_nodes()
    )


def leaf_names(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


def five_taxa_tree(tns=None, ab="speciation", de="duplication"):
    tns = tns or dendropy.TaxonNamespace()
    return make_tree(
        "((A,B),C,(D,E));",
        {fs("AB"): ab, fs("DE"): de},
        tns,
    )


# ---------------------------------------------------------------- computeLRF


def test_identical_trees_have_zero_distance():
    tns = dendropy.TaxonNamespace()
    t1 = five_taxa_tree(tns)
    t2 = five_taxa_tree(tns)
    assert computeLRF(t1, t2) == 0


def test_identical_larger_tree_zero_distance():
    tns = dendropy.TaxonNamespace()
    labels = {fs("AB"): "speciation", fs("ABC"): "duplication", fs("DE"): "speciation"}
    t1 = make_tree("(((A,B),C),(D,E),F);", labels, tns)
    t2 = make_tree("(((A,B),C),(D,E),F);", labels, tns)
    assert computeLRF(t1, t2) == 0


def test_child_order_does_not_matter():
    tns = dendropy.TaxonNamespace()
    labels = {fs("AB"): "speciation", fs("DE"): "duplication"}
    t1 = make_tree("((A,B),C,(D,E));", labels, tns)
    t2 = make_tree("((E,D),(B,A),C);", labels, tns)
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    tns = dendropy.TaxonNamespace()
    t1 = five_taxa_tree(tns, ab="speciation")
    t2 = five_taxa_tree(tns, ab="duplication")
    assert computeLRF(t1, t2) == 1


def test_single_collapsed_edge_costs_one():
    tns = dendropy.TaxonNamespace()
    t1 = make_tree(
        "((A,B),C,(D,E));", {fs("AB"): "speciation", fs("DE"): "duplication"}, tns
    )
    t2 = make_tree("(A,B,C,(D,E));", {fs("DE"): "duplication"}, tns)
    assert computeLRF(t1, t2) == 1


def test_no_shared_bipartitions():
    tns = dendropy.TaxonNamespace()
    t1 = make_tree(
        "((A,B),(C,D),E);", {fs("AB"): "speciation", fs("CD"): "speciation"}, tns
    )
    t2 = make_tree(
        "((A,C),(B,D),E);", {fs("AC"): "speciation", fs("BD"): "speciation"}, tns
    )
    assert computeLRF(t1, t2) == 4


def test_collapse_plus_label_difference():
    tns = dendropy.TaxonNamespace()
    t1 = make_tree(
        "((A,B),(C,D),E);", {fs("AB"): "speciation", fs("CD"): "duplication"}, tns
    )
    t2 = make_tree("((A,B),C,D,E);", {fs("AB"): "duplication"}, tns)
    assert computeLRF(t1, t2) == 2


def test_symmetric():
    tns = dendropy.TaxonNamespace()
    t1 = make_tree(
        "((A,B),(C,D),E);", {fs("AB"): "speciation", fs("CD"): "duplication"}, tns
    )
    t2 = make_tree("((A,B),C,D,E);", {fs("AB"): "duplication"}, tns)
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_star_tree_versus_resolved_tree():
    tns = dendropy.TaxonNamespace()
    star = make_tree("(A,B,C,D,E);", {}, tns)
    resolved = five_taxa_tree(tns)
    assert computeLRF(star, resolved) == 2
    assert computeLRF(resolved, star) == 2


def test_inputs_are_not_modified():
    tns = dendropy.TaxonNamespace()
    t1 = make_tree(
        "((A,B),(C,D),E);", {fs("AB"): "speciation", fs("CD"): "duplication"}, tns
    )
    t2 = make_tree("((A,B),C,D,E);", {fs("AB"): "duplication"}, tns)
    sig1, sig2 = signature(t1), signature(t2)
    n1 = len(t1.internal_nodes())
    n2 = len(t2.internal_nodes())
    computeLRF(t1, t2)
    assert signature(t1) == sig1
    assert signature(t2) == sig2
    assert len(t1.internal_nodes()) == n1
    assert len(t2.internal_nodes()) == n2


# ---------------------------------------------------------- mutateLabeledTree


def test_mutate_zero_edits_returns_equal_copy():
    t = five_taxa_tree()
    out = mutateLabeledTree(t, 0)
    assert out is not t
    assert signature(out) == signature(t)


def test_mutate_does_not_change_input():
    t = make_tree(
        "(((A,B),C),(D,E),(F,G,H));",
        {fs("AB"): "speciation", fs("ABC"): "duplication",
         fs("DE"): "speciation", fs("FGH"): "duplication"},
        dendropy.TaxonNamespace(),
    )
    before = signature(t)
    random.seed(1)
    mutateLabeledTree(t, 5, p_flip=0.0)
    assert signature(t) == before


def test_mutate_only_flips_changes_exactly_one_label():
    t = five_taxa_tree()
    random.seed(3)
    out = mutateLabeledTree(t, 1, p_flip=1.0)
    old = dict(signature(t))
    new = dict(signature(out))
    assert set(old) == set(new)  # topology unchanged
    diffs = [k for k in old if old[k] != new[k]]
    assert len(diffs) == 1
    assert {old[diffs[0]], new[diffs[0]]} == {"speciation", "duplication"}
    assert computeLRF(t, out) == 1


def test_mutate_flips_keep_topology_and_valid_labels():
    t = five_taxa_tree()
    random.seed(11)
    out = mutateLabeledTree(t, 4, p_flip=1.0)
    assert [k for k, _ in signature(out)] == [k for k, _ in signature(t)]
    for n in out.internal_nodes():
        if n.parent_node is not None:
            assert n.label in ("speciation", "duplication")


def test_mutate_lrf_single_collapse():
    t = five_taxa_tree()
    random.seed(5)
    out = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert len(out.internal_nodes()) == len(t.internal_nodes()) - 1
    assert set(k for k, _ in signature(out)).issubset(set(k for k, _ in signature(t)))
    assert leaf_names(out) == leaf_names(t)
    assert computeLRF(t, out) == 1


def test_mutate_star_tree_is_expanded():
    t = make_tree("(A,B,C,D,E);", {}, dendropy.TaxonNamespace())
    random.seed(7)
    out = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert len(t.internal_nodes()) == 1
    assert len(out.internal_nodes()) == 2
    assert leaf_names(out) == ["A", "B", "C", "D", "E"]


def test_mutate_elrf_collapses_only_same_label_edge():
    t = make_tree(
        "(((A,B),C),D,E);",
        {fs("AB"): "speciation", fs("ABC"): "speciation"},
        dendropy.TaxonNamespace(),
    )
    random.seed(2)
    out = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    sig = signature(out)
    assert len(out.internal_nodes()) == 2
    assert (("A", "B"), "speciation") not in sig
    assert (("A", "B", "C"), "speciation") in sig
    assert leaf_names(out) == ["A", "B", "C", "D", "E"]


def test_mutate_elrf_without_candidates_raises():
    # labels differ from the root's, so nothing can be collapsed or expanded
    t = five_taxa_tree()
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")


def test_mutate_unsupported_model_raises():
    t = make_tree(
        "(((A,B),C),D,E);",
        {fs("AB"): "speciation", fs("ABC"): "speciation"},
        dendropy.TaxonNamespace(),
    )
    with pytest.raises(ValueError, match="unsupported model"):
        mutateLabeledTree(t, 1, p_flip=0.0, model="foo")


def test_mutate_is_reproducible_with_seed():
    t = make_tree(
        "(((A,B),C),(D,E),(F,G,H));",
        {fs("AB"): "speciation", fs("ABC"): "duplication",
         fs("DE"): "speciation", fs("FGH"): "duplication"},
        dendropy.TaxonNamespace(),
    )
    random.seed(123)
    a = mutateLabeledTree(t, 4, p_flip=0.0)
    random.seed(123)
    b = mutateLabeledTree(t, 4, p_flip=0.0)
    assert signature(a) == signature(b)


def test_mutate_many_edits_preserves_leaves():
    t = make_tree(
        "(((A,B),C),(D,E),(F,G,H));",
        {fs("AB"): "speciation", fs("ABC"): "duplication",
         fs("DE"): "speciation", fs("FGH"): "duplication"},
        dendropy.TaxonNamespace(),
    )
    expected = leaf_names(t)
    for seed in range(5):
        random.seed(seed)
        out = mutateLabeledTree(t, 4, p_flip=0.0, model="LRF")
        assert out is not t
        assert leaf_names(out) == expected
