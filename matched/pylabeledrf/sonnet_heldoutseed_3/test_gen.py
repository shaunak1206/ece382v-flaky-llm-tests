import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def make_trees(*newicks):
    """Parse several newick strings sharing one taxon namespace."""
    tns = dendropy.TaxonNamespace()
    return [
        dendropy.Tree.get(data=nw, schema="newick", taxon_namespace=tns)
        for nw in newicks
    ]


def internal_labels(tree):
    return sorted(str(n.label) for n in tree.internal_nodes())


def leaf_labels(tree):
    return sorted(l.taxon.label for l in tree.leaf_nodes())


BALANCED_R = "[&R] ((A,B)speciation,(C,D)speciation)speciation;"
BALANCED_R_SWAP = "[&R] ((A,C)speciation,(B,D)speciation)speciation;"
STAR_R = "[&R] (A,B,C,D)speciation;"


# ---------------------------------------------------------------- computeLRF

def test_identical_rooted_trees_have_zero_distance():
    t1, t2 = make_trees(BALANCED_R, BALANCED_R)
    assert computeLRF(t1, t2) == 0


def test_identical_unrooted_trees_have_zero_distance():
    nw = "[&U] ((A,B)speciation,(C,D)speciation,E)speciation;"
    t1, t2 = make_trees(nw, nw)
    assert computeLRF(t1, t2) == 0


def test_same_unrooted_tree_with_different_rooting_has_zero_distance():
    t1, t2 = make_trees(
        "[&U] ((A,B)speciation,(C,D)speciation,E)speciation;",
        "[&U] (A,B,((C,D)speciation,E)speciation)speciation;",
    )
    assert computeLRF(t1, t2) == 0


def test_single_label_difference_costs_one():
    t1, t2 = make_trees(
        BALANCED_R,
        "[&R] ((A,B)duplication,(C,D)speciation)speciation;",
    )
    assert computeLRF(t1, t2) == 1


def test_all_labels_different_costs_one_per_node():
    t1, t2 = make_trees(
        BALANCED_R,
        "[&R] ((A,B)duplication,(C,D)duplication)duplication;",
    )
    assert computeLRF(t1, t2) == 3


def test_different_topologies_same_labels_is_plain_rf():
    t1, t2 = make_trees(BALANCED_R, BALANCED_R_SWAP)
    assert computeLRF(t1, t2) == 4


def test_resolved_tree_vs_star_costs_number_of_collapses():
    t1, t2 = make_trees(BALANCED_R, STAR_R)
    assert computeLRF(t1, t2) == 2


def test_one_collapse_with_good_edges_present():
    t1, t2 = make_trees(
        "[&R] (((A,B)speciation,C)speciation,(D,E)speciation)speciation;",
        "[&R] ((A,B)speciation,C,(D,E)speciation)speciation;",
    )
    assert computeLRF(t1, t2) == 1


def test_collapsed_node_with_other_label_still_costs_one():
    t1, t2 = make_trees(
        "[&R] (((A,B)speciation,C)duplication,(D,E)speciation)speciation;",
        "[&R] ((A,B)speciation,C,(D,E)speciation)speciation;",
    )
    assert computeLRF(t1, t2) == 1


@pytest.mark.parametrize(
    "nw1,nw2",
    [
        (BALANCED_R, STAR_R),
        (BALANCED_R, BALANCED_R_SWAP),
        (BALANCED_R, "[&R] ((A,B)duplication,(C,D)speciation)speciation;"),
    ],
)
def test_distance_is_symmetric(nw1, nw2):
    a1, a2 = make_trees(nw1, nw2)
    b1, b2 = make_trees(nw1, nw2)
    assert computeLRF(a1, a2) == computeLRF(b2, b1)


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = make_trees(BALANCED_R, STAR_R)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# ---------------------------------------------------------- mutateLabeledTree

def test_mutate_returns_new_tree_and_leaves_input_unchanged():
    (tree,) = make_trees(BALANCED_R)
    before = tree.as_string("newick")
    random.seed(1)
    mutated = mutateLabeledTree(tree, 5)
    assert mutated is not tree
    assert tree.as_string("newick") == before


def test_mutate_zero_edits_gives_identical_tree():
    (tree,) = make_trees(BALANCED_R)
    mutated = mutateLabeledTree(tree, 0)
    assert leaf_labels(mutated) == ["A", "B", "C", "D"]
    assert internal_labels(mutated) == internal_labels(tree)
    assert computeLRF(tree, mutated) == 0


def test_mutate_single_flip_changes_exactly_one_label():
    (tree,) = make_trees(BALANCED_R)
    random.seed(3)
    mutated = mutateLabeledTree(tree, 1, p_flip=1.0)
    # topology is unchanged
    assert len(mutated.internal_nodes()) == 3
    assert leaf_labels(mutated) == ["A", "B", "C", "D"]
    assert computeLRF(tree, mutated) == 1
    assert internal_labels(mutated) != internal_labels(tree)


def test_mutate_only_flips_keeps_distance_bounded_by_edits():
    (tree,) = make_trees(BALANCED_R)
    random.seed(11)
    mutated = mutateLabeledTree(tree, 3, p_flip=1.0)
    assert len(mutated.internal_nodes()) == 3
    # a node flipped twice returns to its original state, so 1 or 3
    assert computeLRF(tree, mutated) in (1, 3)


def test_mutate_collapse_reduces_number_of_internal_nodes():
    (tree,) = make_trees(BALANCED_R)
    random.seed(7)
    mutated = mutateLabeledTree(tree, 1, p_flip=0.0)
    assert len(mutated.internal_nodes()) == 2
    assert leaf_labels(mutated) == ["A", "B", "C", "D"]


def test_mutate_unrooted_never_collapses_or_expands_the_root():
    (tree,) = make_trees("[&U] ((A,B)speciation,(C,D)speciation,E)speciation;")
    random.seed(2)
    mutated = mutateLabeledTree(tree, 1, p_flip=0.0)
    # one of the two inner nodes got collapsed into the root
    assert len(mutated.internal_nodes()) == 2
    assert len(mutated.seed_node.child_nodes()) == 4
    assert leaf_labels(mutated) == ["A", "B", "C", "D", "E"]


def test_mutate_elrf_collapses_only_edges_between_same_labels():
    (tree,) = make_trees("[&R] ((A,B)speciation,(C,D)duplication)speciation;")
    random.seed(4)
    mutated = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
    # the only collapsible edge is the (A,B) one
    assert internal_labels(mutated) == ["duplication", "speciation"]
    assert len(mutated.internal_nodes()) == 2


def test_mutate_elrf_without_possible_edit_raises():
    (tree,) = make_trees("[&R] ((A,B)duplication,(C,D)duplication)speciation;")
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")


def test_mutate_unsupported_model_raises():
    (tree,) = make_trees(BALANCED_R)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="XYZ")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mutate_many_edits_preserves_leaves_and_valid_labels(seed):
    (tree,) = make_trees(
        "[&R] (((A,B)speciation,C)duplication,((D,E)speciation,F)speciation)speciation;"
    )
    random.seed(seed)
    mutated = mutateLabeledTree(tree, 10)
    assert leaf_labels(mutated) == ["A", "B", "C", "D", "E", "F"]
    for n in mutated.internal_nodes():
        assert n.label in ("speciation", "duplication")


def test_mutate_is_reproducible_with_same_seed():
    (tree,) = make_trees(
        "[&R] (((A,B)speciation,C)duplication,((D,E)speciation,F)speciation)speciation;"
    )
    random.seed(42)
    m1 = mutateLabeledTree(tree, 6)
    random.seed(42)
    m2 = mutateLabeledTree(tree, 6)
    assert m1.as_string("newick") == m2.as_string("newick")
