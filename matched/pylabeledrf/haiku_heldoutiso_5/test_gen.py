import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeLRF,
    extend_node,
    flip_node,
    mutateLabeledTree,
)


def parse(*newicks):
    """Parse rooted labeled trees that all share one taxon namespace.

    A shared namespace is required so that bipartition bitmasks are comparable.
    """
    tns = dendropy.TaxonNamespace()
    return [
        dendropy.Tree.get_from_string(s, "newick", taxon_namespace=tns, rooted=True)
        for s in newicks
    ]


def signature(tree):
    """Set of (leaf-label set, node label) for every internal node."""
    return {
        (frozenset(leaf.taxon.label for leaf in node.leaf_iter()), node.label)
        for node in tree.internal_nodes()
    }


def leaf_labels(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_node_iter())


def internal_with_leaves(tree, labels):
    """Return the non-root internal node whose leaf set equals `labels`."""
    target = frozenset(labels)
    for node in tree.internal_nodes():
        if node.parent_node is None:
            continue
        if frozenset(leaf.taxon.label for leaf in node.leaf_iter()) == target:
            return node
    raise AssertionError("no internal node with leaves %s" % sorted(labels))


SPEC = "speciation"
DUP = "duplication"
BALANCED = "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC)
SIX_LEAVES = "(((A,B)%s,C)%s,(E,F)%s)%s;" % (SPEC, DUP, SPEC, SPEC)


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

def test_identical_trees_have_zero_distance():
    t1, t2 = parse(BALANCED, BALANCED)
    assert computeLRF(t1, t2) == 0


def test_single_label_mismatch_costs_one():
    t1, t2 = parse(
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC),
        "((A,B)%s,(C,D)%s)%s;" % (DUP, SPEC, SPEC),
    )
    assert computeLRF(t1, t2) == 1


def test_two_label_mismatches_cost_two():
    t1, t2 = parse(
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC),
        "((A,B)%s,(C,D)%s)%s;" % (DUP, DUP, SPEC),
    )
    assert computeLRF(t1, t2) == 2


def test_root_label_mismatch_costs_one():
    t1, t2 = parse(
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC),
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, DUP),
    )
    assert computeLRF(t1, t2) == 1


def test_distance_is_symmetric():
    t1, t2 = parse(
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC),
        "((A,B)%s,(C,D)%s)%s;" % (DUP, SPEC, SPEC),
    )
    assert computeLRF(t1, t2) == computeLRF(t2, t1) == 1


def test_different_topologies_same_labels_gives_four():
    # Rooted RF between ((A,B),(C,D)) and ((A,C),(B,D)) is 4 when all
    # internal nodes share the same label; no good edges are shared.
    t1, t2 = parse(BALANCED, "((A,C)%s,(B,D)%s)%s;" % (SPEC, SPEC, SPEC))
    assert computeLRF(t1, t2) == 4


def test_resolving_polytomy_costs_one():
    t1, t2 = parse("(A,B,C)%s;" % SPEC, "((A,B)%s,C)%s;" % (SPEC, SPEC))
    assert computeLRF(t1, t2) == 1


def test_computeLRF_does_not_modify_inputs():
    t1, t2 = parse(
        "((A,B)%s,(C,D)%s)%s;" % (SPEC, SPEC, SPEC),
        "((A,B)%s,(C,D)%s)%s;" % (DUP, SPEC, SPEC),
    )
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_mutate_zero_edits_returns_equal_tree():
    (tree,) = parse(SIX_LEAVES)
    result = mutateLabeledTree(tree, 0)
    assert signature(result) == signature(tree)
    assert computeLRF(tree, result) == 0


def test_mutate_returns_new_object_and_leaves_input_unchanged():
    (tree,) = parse(SIX_LEAVES)
    before = signature(tree)
    result = mutateLabeledTree(tree, 3, p_flip=0.0)
    assert result is not tree
    assert signature(tree) == before


def test_mutate_single_flip_changes_exactly_one_label():
    (tree,) = parse(SIX_LEAVES)
    random.seed(1)
    result = mutateLabeledTree(tree, 1, p_flip=1.0)
    # Topology is unchanged; exactly one (leafset, label) entry differs
    # on each side of the symmetric difference.
    assert len(signature(tree) ^ signature(result)) == 2


def test_mutate_is_deterministic_with_seed():
    (tree,) = parse(SIX_LEAVES)
    random.seed(123)
    a = mutateLabeledTree(tree, 4)
    random.seed(123)
    b = mutateLabeledTree(tree, 4)
    assert signature(a) == signature(b)


def test_mutate_preserves_leaf_set():
    (tree,) = parse(SIX_LEAVES)
    random.seed(7)
    result = mutateLabeledTree(tree, 5, p_flip=0.0)
    assert leaf_labels(result) == leaf_labels(tree)


def test_mutate_unsupported_model_raises():
    (tree,) = parse(BALANCED)
    # With p_flip=0 and an unknown model, the only candidates are collapses,
    # which must raise for an unsupported model.
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="XYZ")


# ---------------------------------------------------------------------------
# Helper functions used by mutateLabeledTree
# ---------------------------------------------------------------------------

def test_flip_node_swaps_speciation_and_duplication():
    (tree,) = parse(BALANCED)
    root = tree.seed_node
    assert root.label == SPEC
    flip_node(root)
    assert root.label == DUP
    flip_node(root)
    assert root.label == SPEC


def test_flip_node_unlabeled_raises():
    (tree,) = parse("((A,B),(C,D));")
    assert tree.seed_node.label is None
    with pytest.raises(ValueError):
        flip_node(tree.seed_node)


def test_collapse_node_elrf_matching_labels_collapses():
    (tree,) = parse(BALANCED)
    node = internal_with_leaves(tree, {"A", "B"})
    before = len(list(tree.internal_nodes()))
    collapse_node(node, model="ELRF")
    assert len(list(tree.internal_nodes())) == before - 1


def test_collapse_node_elrf_mismatched_labels_raises():
    (tree,) = parse("((A,B)%s,(C,D)%s)%s;" % (DUP, SPEC, SPEC))
    node = internal_with_leaves(tree, {"A", "B"})
    with pytest.raises(ValueError):
        collapse_node(node, model="ELRF")


def test_collapse_node_lrf_collapses_regardless_of_labels():
    (tree,) = parse("((A,B)%s,(C,D)%s)%s;" % (DUP, SPEC, SPEC))
    node = internal_with_leaves(tree, {"A", "B"})
    before = len(list(tree.internal_nodes()))
    collapse_node(node, model="LRF")
    assert len(list(tree.internal_nodes())) == before - 1


def test_extend_node_adds_internal_node_and_preserves_leaves():
    (tree,) = parse("(A,B,C,D)%s;" % SPEC)
    random.seed(0)
    before = len(list(tree.internal_nodes()))
    extend_node(tree.seed_node, model="LRF")
    assert len(list(tree.internal_nodes())) == before + 1
    assert leaf_labels(tree) == ["A", "B", "C", "D"]


def test_extend_node_unsupported_model_raises():
    (tree,) = parse("(A,B,C,D)%s;" % SPEC)
    with pytest.raises(ValueError):
        extend_node(tree.seed_node, model="XYZ")
