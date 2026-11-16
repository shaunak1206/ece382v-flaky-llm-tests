import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree

SPE = "speciation"
DUP = "duplication"


def build(newick, labels, tns):
    """Parse a newick string and assign labels to internal nodes by leaf set."""
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns)
    tree.is_rooted = False
    for nd in tree.internal_nodes():
        key = frozenset(leaf.taxon.label for leaf in nd.leaf_iter())
        nd.label = labels.get(key)
    return tree


def label_map(tree):
    return {
        frozenset(leaf.taxon.label for leaf in nd.leaf_iter()): nd.label
        for nd in tree.internal_nodes()
    }


def leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_node_iter())


def fs(s):
    return frozenset(s)


SIX = "[&U]((A,B),(C,D),(E,F));"
SIX_LABELS = {fs("AB"): SPE, fs("CD"): SPE, fs("EF"): SPE}

EIGHT = "[&U]((A,B),(C,D),(E,F,G,H));"
EIGHT_LABELS = {fs("AB"): SPE, fs("CD"): DUP, fs("EFGH"): SPE}


@pytest.fixture
def tns():
    return dendropy.TaxonNamespace()


# ---------------------------------------------------------------- computeLRF

def test_identical_trees_distance_zero(tns):
    t1 = build(SIX, SIX_LABELS, tns)
    t2 = build(SIX, SIX_LABELS, tns)
    assert computeLRF(t1, t2) == 0


def test_tree_against_itself_with_mixed_labels(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    assert computeLRF(t, t) == 0


def test_child_order_does_not_matter(tns):
    t1 = build(SIX, SIX_LABELS, tns)
    t2 = build("[&U]((E,F),(A,B),(C,D));", SIX_LABELS, tns)
    assert computeLRF(t1, t2) == 0


def test_single_label_flip_costs_one(tns):
    t1 = build(SIX, SIX_LABELS, tns)
    labels2 = dict(SIX_LABELS)
    labels2[fs("AB")] = DUP
    t2 = build(SIX, labels2, tns)
    assert computeLRF(t1, t2) == 1


def test_two_label_flips_cost_two(tns):
    t1 = build(SIX, SIX_LABELS, tns)
    labels2 = dict(SIX_LABELS)
    labels2[fs("AB")] = DUP
    labels2[fs("EF")] = DUP
    t2 = build(SIX, labels2, tns)
    assert computeLRF(t1, t2) == 2


def test_single_collapse_same_labels_costs_one(tns):
    t1 = build("[&U](A,B,((C,D),E,F));",
               {fs("CDEF"): SPE, fs("CD"): SPE}, tns)
    t2 = build("[&U](A,B,(C,D,E,F));", {fs("CDEF"): SPE}, tns)
    assert computeLRF(t1, t2) == 1


def test_collapse_of_other_type_node_costs_one(tns):
    # the collapsed duplication node is absorbed in an island that
    # still shares a label with the other side, so no substitution
    t1 = build("[&U](A,B,((C,D),E,F));",
               {fs("CDEF"): SPE, fs("CD"): DUP}, tns)
    t2 = build("[&U](A,B,(C,D,E,F));", {fs("CDEF"): SPE}, tns)
    assert computeLRF(t1, t2) == 1


def test_collapse_plus_substitution_costs_two(tns):
    t1 = build("[&U](A,B,((C,D),E,F));",
               {fs("CDEF"): SPE, fs("CD"): SPE}, tns)
    t2 = build("[&U](A,B,(C,D,E,F));", {fs("CDEF"): DUP}, tns)
    assert computeLRF(t1, t2) == 2


def test_distance_is_symmetric(tns):
    t1 = build("[&U](A,B,((C,D),E,F));",
               {fs("CDEF"): SPE, fs("CD"): SPE}, tns)
    t2 = build("[&U](A,B,(C,D,E,F));", {fs("CDEF"): DUP}, tns)
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_two_star_trees_distance_zero(tns):
    t1 = build("[&U](A,B,C,D);", {}, tns)
    t2 = build("[&U](A,B,C,D);", {}, tns)
    assert computeLRF(t1, t2) == 0


def test_star_versus_one_resolved_edge(tns):
    t1 = build("[&U](A,B,C,D,E);", {}, tns)
    t2 = build("[&U]((A,B),C,D,E);", {fs("AB"): SPE}, tns)
    assert computeLRF(t1, t2) == 1


def test_disjoint_topologies_equal_robinson_foulds(tns):
    labels = {fs("AB"): SPE, fs("CD"): SPE, fs("AC"): SPE, fs("BD"): SPE}
    t1 = build("[&U]((A,B),(C,D),E,F);", labels, tns)
    t2 = build("[&U]((A,C),(B,D),E,F);", labels, tns)
    assert computeLRF(t1, t2) == 4


def test_computeLRF_does_not_modify_inputs(tns):
    t1 = build("[&U](A,B,((C,D),E,F));",
               {fs("CDEF"): SPE, fs("CD"): SPE}, tns)
    t2 = build("[&U](A,B,(C,D,E,F));", {fs("CDEF"): DUP}, tns)
    before1, before2 = label_map(t1), label_map(t2)
    s1 = t1.as_string(schema="newick")
    s2 = t2.as_string(schema="newick")
    computeLRF(t1, t2)
    assert label_map(t1) == before1
    assert label_map(t2) == before2
    assert t1.as_string(schema="newick") == s1
    assert t2.as_string(schema="newick") == s2


# ----------------------------------------------------------- mutateLabeledTree

def test_mutate_returns_new_tree_and_leaves_input_unchanged(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    before = label_map(t)
    newick = t.as_string(schema="newick")
    random.seed(1)
    out = mutateLabeledTree(t, 2)
    assert out is not t
    assert label_map(t) == before
    assert t.as_string(schema="newick") == newick


def test_mutate_zero_edits_gives_identical_tree(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    out = mutateLabeledTree(t, 0)
    assert label_map(out) == label_map(t)
    assert computeLRF(t, out) == 0


def test_mutate_flip_only_changes_exactly_one_label(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    for seed in range(10):
        random.seed(seed)
        out = mutateLabeledTree(t, 1, p_flip=1.0)
        before, after = label_map(t), label_map(out)
        assert set(before) == set(after)  # topology unchanged
        diff = [k for k in before if before[k] != after[k]]
        assert len(diff) == 1
        assert after[diff[0]] in (SPE, DUP)
        assert computeLRF(t, out) == 1


def test_mutate_many_flips_bounded_distance(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    random.seed(7)
    out = mutateLabeledTree(t, 3, p_flip=1.0)
    assert set(label_map(out)) == set(label_map(t))
    assert computeLRF(t, out) <= 3


def test_mutate_topology_edit_changes_node_count_by_one(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    orig = len(t.internal_nodes())
    for seed in range(15):
        random.seed(seed)
        out = mutateLabeledTree(t, 1, p_flip=0.0)
        assert abs(len(out.internal_nodes()) - orig) == 1
        assert leaf_names(out) == leaf_names(t)
        assert computeLRF(t, out) == 1


def test_mutate_preserves_leaves_and_valid_labels(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    for seed in range(20):
        random.seed(seed)
        out = mutateLabeledTree(t, 2, p_flip=0.3)
        assert leaf_names(out) == leaf_names(t)
        for nd in out.internal_nodes():
            assert nd.label in (SPE, DUP, None)


def test_mutate_elrf_only_expands_when_no_collapse_is_consistent(tns):
    # all internal nodes hang off the root, whose label is None, so no
    # edge may be collapsed under ELRF; the only edit is an expansion
    t = build(EIGHT, EIGHT_LABELS, tns)
    orig = len(t.internal_nodes())
    for seed in range(5):
        random.seed(seed)
        out = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
        assert len(out.internal_nodes()) == orig + 1
        assert leaf_names(out) == leaf_names(t)
        assert computeLRF(t, out) == 1


def test_mutate_unsupported_model_raises(tns):
    t = build(EIGHT, EIGHT_LABELS, tns)
    random.seed(0)
    with pytest.raises(ValueError):
        mutateLabeledTree(t, 1, p_flip=0.0, model="XYZ")
