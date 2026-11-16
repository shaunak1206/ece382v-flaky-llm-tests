import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


BINARY_NWK = (
    "(((A,B)speciation,C)duplication,"
    "((D,E)speciation,F)speciation)speciation;"
)


def parse(newick, tns, rooted=True):
    return dendropy.Tree.get(
        data=newick,
        schema="newick",
        taxon_namespace=tns,
        rooting="force-rooted" if rooted else "force-unrooted",
    )


def signature(tree):
    return {
        tuple(sorted(l.taxon.label for l in n.leaf_iter())): n.label
        for n in tree.internal_nodes()
    }


def leaf_labels(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


def test_identical_trees_have_zero_distance():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    t2 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    assert computeLRF(t1, t2) == 0


def test_single_label_flip_costs_one():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    t2 = parse("((A,B)duplication,(C,D)speciation)speciation;", tns)
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_two_label_flips_cost_two():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    t2 = parse("((A,B)duplication,(C,D)duplication)speciation;", tns)
    assert computeLRF(t1, t2) == 2


def test_child_rotation_does_not_change_distance():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    t2 = parse("((C,D)speciation,(A,B)speciation)speciation;", tns)
    assert computeLRF(t1, t2) == 0
    t3 = parse("((C,D)duplication,(A,B)speciation)speciation;", tns)
    assert computeLRF(t1, t3) == 1


def test_resolved_tree_versus_star_tree():
    tns = dendropy.TaxonNamespace()
    resolved = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    star = parse("(A,B,C,D)speciation;", tns)
    assert computeLRF(resolved, star) == 2
    assert computeLRF(star, resolved) == 2


def test_different_topologies_same_labels():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    t2 = parse("((A,C)speciation,(B,D)speciation)speciation;", tns)
    assert computeLRF(t1, t2) == 4


def test_partial_polytomy_costs_one_and_is_symmetric():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,C,D)speciation;", tns)
    t2 = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_collapse_plus_substitution():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,C,D)speciation;", tns)
    t2 = parse("((A,B)speciation,(C,D)duplication)duplication;", tns)
    # one collapse plus one substitution of the root island label
    assert computeLRF(t1, t2) == 2
    assert computeLRF(t2, t1) == 2


def test_unrooted_identical_trees_zero_and_flip_one():
    tns = dendropy.TaxonNamespace()
    nwk = "((A,B)speciation,(C,D)speciation,E)speciation;"
    t1 = parse(nwk, tns, rooted=False)
    t2 = parse(nwk, tns, rooted=False)
    assert computeLRF(t1, t2) == 0
    t3 = parse("((A,B)duplication,(C,D)speciation,E)speciation;", tns, rooted=False)
    assert computeLRF(t1, t3) == 1


def test_computeLRF_does_not_modify_inputs():
    tns = dendropy.TaxonNamespace()
    t1 = parse("((A,B)speciation,C,D)speciation;", tns)
    t2 = parse("((A,B)duplication,(C,D)speciation)speciation;", tns)
    s1, s2 = signature(t1), signature(t2)
    computeLRF(t1, t2)
    assert signature(t1) == s1
    assert signature(t2) == s2
    assert leaf_labels(t1) == ["A", "B", "C", "D"]


def test_mutate_zero_edits_returns_equal_copy():
    tns = dendropy.TaxonNamespace()
    t = parse(BINARY_NWK, tns)
    m = mutateLabeledTree(t, 0)
    assert m is not t
    assert signature(m) == signature(t)
    assert computeLRF(t, m) == 0


def test_mutate_leaves_input_unchanged():
    tns = dendropy.TaxonNamespace()
    t = parse(BINARY_NWK, tns)
    before = signature(t)
    n_internal = len(t.internal_nodes())
    random.seed(7)
    mutateLabeledTree(t, 5)
    assert signature(t) == before
    assert len(t.internal_nodes()) == n_internal


def test_mutate_flip_only_changes_exactly_one_label():
    tns = dendropy.TaxonNamespace()
    t = parse(BINARY_NWK, tns)
    before = signature(t)
    for seed in range(5):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=1.0)
        after = signature(m)
        assert set(after) == set(before)
        changed = [k for k in before if before[k] != after[k]]
        assert len(changed) == 1
        k = changed[0]
        assert {before[k], after[k]} == {"speciation", "duplication"}
        assert computeLRF(t, m) == 1


def test_mutate_collapse_only_on_binary_tree():
    tns = dendropy.TaxonNamespace()
    t = parse(BINARY_NWK, tns)
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0)
        assert len(m.internal_nodes()) == len(t.internal_nodes()) - 1
        assert leaf_labels(m) == leaf_labels(t)
        assert computeLRF(t, m) == 1


def test_mutate_unrooted_root_is_never_expanded():
    tns = dendropy.TaxonNamespace()
    t = parse("((A,B)speciation,(C,D)speciation,E)speciation;", tns, rooted=False)
    for seed in range(20):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0)
        # only collapses are possible: 3 internal nodes -> 2
        assert len(m.internal_nodes()) == 2
        assert leaf_labels(m) == ["A", "B", "C", "D", "E"]


@pytest.mark.parametrize("model", ["LRF", "ELRF"])
def test_mutate_expands_polytomy(model):
    tns = dendropy.TaxonNamespace()
    t = parse("(A,B,C,D)speciation;", tns)
    for seed in range(10):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0, model=model)
        assert len(m.internal_nodes()) == 2
        assert leaf_labels(m) == ["A", "B", "C", "D"]
        assert len(t.internal_nodes()) == 1
    # ELRF expansion keeps the label of the expanded node
    random.seed(0)
    m = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert all(n.label == "speciation" for n in m.internal_nodes())


def test_mutate_elrf_collapses_only_same_label_edges():
    tns = dendropy.TaxonNamespace()
    t = parse("((A,B)speciation,(C,D)duplication)speciation;", tns)
    for seed in range(5):
        random.seed(seed)
        m = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
        sig = signature(m)
        assert len(sig) == 2
        assert sig[("C", "D")] == "duplication"
        assert ("A", "B") not in sig


def test_mutate_elrf_without_possible_edit_raises():
    tns = dendropy.TaxonNamespace()
    t = parse("((A,B)duplication,(C,D)duplication)speciation;", tns)
    random.seed(1)
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")


def test_mutate_unsupported_model_raises():
    tns = dendropy.TaxonNamespace()
    t = parse("((A,B)speciation,(C,D)speciation)speciation;", tns)
    random.seed(1)
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")


def test_mutate_flip_without_labels_raises():
    tns = dendropy.TaxonNamespace()
    t = parse("((A,B),(C,D));", tns)
    random.seed(1)
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=1.0)


def test_mutate_is_reproducible_and_preserves_leaves():
    tns = dendropy.TaxonNamespace()
    t = parse(BINARY_NWK, tns)
    random.seed(123)
    m1 = mutateLabeledTree(t, 8)
    random.seed(123)
    m2 = mutateLabeledTree(t, 8)
    assert leaf_labels(m1) == leaf_labels(t)
    assert leaf_labels(m2) == leaf_labels(t)
    assert len(m1.internal_nodes()) == len(m2.internal_nodes())
    assert computeLRF(m1, m2) == 0
