import random

import pytest
import dendropy

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree

S = "speciation"
D = "duplication"
ALL = frozenset("ABCDE")


def fs(s):
    return frozenset(s)


def new_ns():
    return dendropy.TaxonNamespace(["A", "B", "C", "D", "E"])


def build(newick, labels, ns):
    """Parse a newick string and assign labels to internal nodes based on the
    set of leaves below them (nodes not in ``labels`` get None)."""
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=ns)
    for node in tree.internal_nodes():
        key = frozenset(l.taxon.label for l in node.leaf_iter())
        node.label = labels.get(key)
    return tree


def signature(tree):
    sig = {}
    for node in tree.internal_nodes():
        key = frozenset(l.taxon.label for l in node.leaf_iter())
        sig[key] = node.label
    return sig


BASE = "[&U] ((A,B),C,(D,E));"


def base_tree(ns, ab=S, de=D, root=S):
    return build(BASE, {fs("AB"): ab, fs("DE"): de, ALL: root}, ns)


def test_identical_trees_distance_zero():
    ns = new_ns()
    t1 = base_tree(ns)
    t2 = base_tree(ns)
    assert computeLRF(t1, t2) == 0


def test_identical_topology_different_child_order_distance_zero():
    ns = new_ns()
    t1 = base_tree(ns)
    t2 = build("[&U] ((E,D),C,(B,A));",
               {fs("AB"): S, fs("DE"): D, ALL: S}, ns)
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    ns = new_ns()
    t1 = base_tree(ns, ab=S)
    t2 = base_tree(ns, ab=D)
    assert computeLRF(t1, t2) == 1


def test_two_label_differences_cost_two():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=D)
    t2 = base_tree(ns, ab=D, de=S)
    assert computeLRF(t1, t2) == 2


def test_one_collapsed_edge_with_compatible_labels_costs_one():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=D, root=S)
    t2 = build("[&U] ((A,B),C,D,E);", {fs("AB"): S, ALL: S}, ns)
    assert computeLRF(t1, t2) == 1


def test_collapsed_edge_plus_label_mismatch_costs_two():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=S, root=S)
    t2 = build("[&U] ((A,B),C,D,E);", {fs("AB"): S, ALL: D}, ns)
    assert computeLRF(t1, t2) == 2


def test_resolved_tree_versus_star_without_substitution():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=D, root=S)
    star = build("[&U] (A,B,C,D,E);", {ALL: S}, ns)
    # two edges need to be collapsed, labels overlap
    assert computeLRF(t1, star) == 2


def test_resolved_tree_versus_star_with_substitution():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=S, root=S)
    star = build("[&U] (A,B,C,D,E);", {ALL: D}, ns)
    # two collapses and one substitution of the single remaining island
    assert computeLRF(t1, star) == 3


def test_quartet_with_different_resolutions():
    ns = new_ns()
    t1 = build("[&U] ((A,B),C,D);", {fs("AB"): S, fs("ABCD"): S}, ns)
    t2 = build("[&U] ((A,C),B,D);", {fs("AC"): S, fs("ABCD"): S}, ns)
    assert computeLRF(t1, t2) == 2


def test_computeLRF_is_symmetric():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=S, root=S)
    t2 = build("[&U] ((A,B),C,D,E);", {fs("AB"): S, ALL: D}, ns)
    star = build("[&U] (A,B,C,D,E);", {ALL: D}, ns)
    assert computeLRF(t1, t2) == computeLRF(t2, t1)
    assert computeLRF(t1, star) == computeLRF(star, t1)


def test_computeLRF_does_not_modify_inputs():
    ns = new_ns()
    t1 = base_tree(ns, ab=S, de=D)
    t2 = build("[&U] ((A,B),C,D,E);", {fs("AB"): D, ALL: S}, ns)
    sig1, sig2 = signature(t1), signature(t2)
    computeLRF(t1, t2)
    assert signature(t1) == sig1
    assert signature(t2) == sig2


def test_mutate_zero_edits_returns_equal_copy():
    ns = new_ns()
    tree = base_tree(ns)
    result = mutateLabeledTree(tree, 0)
    assert result is not tree
    assert signature(result) == signature(tree)
    assert computeLRF(tree, result) == 0


def test_mutate_does_not_modify_input_tree():
    ns = new_ns()
    tree = base_tree(ns)
    before = signature(tree)
    random.seed(5)
    mutateLabeledTree(tree, 3, p_flip=0.5)
    assert signature(tree) == before


def test_mutate_flip_only_changes_exactly_one_label():
    ns = new_ns()
    tree = base_tree(ns, ab=S, de=D, root=S)
    random.seed(3)
    mutated = mutateLabeledTree(tree, 1, p_flip=1.0)
    before, after = signature(tree), signature(mutated)
    assert set(before) == set(after)
    diffs = [k for k in before if before[k] != after[k]]
    assert len(diffs) == 1
    k = diffs[0]
    assert {before[k], after[k]} == {S, D}


def test_mutate_flip_without_labels_raises_value_error():
    ns = new_ns()
    tree = build(BASE, {}, ns)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=1.0)


def test_mutate_lrf_collapse_removes_one_internal_node():
    for seed in range(5):
        ns = new_ns()
        tree = base_tree(ns)
        random.seed(seed)
        mutated = mutateLabeledTree(tree, 1, p_flip=0.0, model="LRF")
        assert len(mutated.internal_nodes()) == len(tree.internal_nodes()) - 1
        assert len(mutated.leaf_nodes()) == 5


def test_mutate_expansion_of_star_tree_adds_internal_node():
    for seed in range(5):
        ns = new_ns()
        star = build("[&U] (A,B,C,D,E);", {ALL: S}, ns)
        random.seed(seed)
        mutated = mutateLabeledTree(star, 1, p_flip=0.0, model="LRF")
        assert len(mutated.internal_nodes()) == 2
        assert sorted(l.taxon.label for l in mutated.leaf_nodes()) == list("ABCDE")
        # the original star is untouched
        assert len(star.internal_nodes()) == 1
        for node in mutated.internal_nodes():
            assert len(node.child_nodes()) >= 2


def test_mutate_elrf_expansion_keeps_label():
    ns = new_ns()
    star = build("[&U] (A,B,C,D,E);", {ALL: S}, ns)
    random.seed(11)
    mutated = mutateLabeledTree(star, 1, p_flip=0.0, model="ELRF")
    assert len(mutated.internal_nodes()) == 2
    assert all(n.label == S for n in mutated.internal_nodes())


def test_mutate_elrf_collapses_only_edge_with_same_label_as_parent():
    ns = new_ns()
    tree = base_tree(ns, ab=S, de=D, root=S)
    for seed in range(3):
        random.seed(seed)
        mutated = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
        sig = signature(mutated)
        # only the AB edge (speciation under speciation) can be collapsed
        assert fs("AB") not in sig
        assert sig[fs("DE")] == D
        assert sig[ALL] == S


def test_mutate_unsupported_model_raises_value_error():
    ns = new_ns()
    star = build("[&U] (A,B,C,D,E);", {ALL: S}, ns)
    with pytest.raises(ValueError):
        mutateLabeledTree(star, 1, p_flip=0.0, model="foo")


def test_mutate_is_deterministic_given_seed():
    ns = new_ns()
    tree = base_tree(ns)
    random.seed(42)
    m1 = mutateLabeledTree(tree, 2, p_flip=0.4)
    random.seed(42)
    m2 = mutateLabeledTree(tree, 2, p_flip=0.4)
    assert signature(m1) == signature(m2)


def test_distance_after_single_flip_is_one():
    for seed in range(5):
        ns = new_ns()
        tree = base_tree(ns)
        random.seed(seed)
        mutated = mutateLabeledTree(tree, 1, p_flip=1.0)
        assert computeLRF(tree, mutated) == 1


def test_distance_after_single_lrf_collapse_is_one():
    for seed in range(5):
        ns = new_ns()
        tree = base_tree(ns)
        random.seed(seed)
        mutated = mutateLabeledTree(tree, 1, p_flip=0.0, model="LRF")
        assert computeLRF(tree, mutated) == 1
