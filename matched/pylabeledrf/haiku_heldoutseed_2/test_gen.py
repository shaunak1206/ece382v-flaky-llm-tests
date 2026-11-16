import random

import dendropy
import pytest

from pylabeledrf.computeLRF import (
    collapse_node,
    computeLRF,
    flip_node,
    mutateLabeledTree,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse(newick, taxa):
    """Parse a rooted newick string into the given shared TaxonNamespace."""
    return dendropy.Tree.get_from_string(newick, "newick", taxon_namespace=taxa)


def parse_pair(newick1, newick2):
    """Parse two trees sharing one taxon namespace (required for comparisons)."""
    taxa = dendropy.TaxonNamespace()
    return parse(newick1, taxa), parse(newick2, taxa)


def clade_labels(tree):
    """Map each internal node's leaf set (frozenset of names) to its label."""
    return {
        frozenset(leaf.taxon.label for leaf in node.leaf_iter()): node.label
        for node in tree.internal_nodes()
    }


def leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_node_iter())


def find_node(tree, names):
    """Return the internal node whose leaf set equals ``names``."""
    target = set(names)
    for node in tree.internal_nodes():
        if {leaf.taxon.label for leaf in node.leaf_iter()} == target:
            return node
    raise AssertionError("no internal node with clade %r" % sorted(target))


QUARTET_LABELED = "[&R] ((A,B)speciation,(C,D)duplication);"
QUARTET_ALT_TOPOLOGY = "[&R] ((A,C)speciation,(B,D)duplication);"
QUARTET_ONE_FLIPPED = "[&R] ((A,B)duplication,(C,D)duplication);"
QUARTET_BOTH_FLIPPED = "[&R] ((A,B)duplication,(C,D)speciation);"
STAR4 = "[&R] (A,B,C,D);"
STAR6 = "[&R] (A,B,C,D,E,F);"


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

def test_identical_trees_have_zero_distance():
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_LABELED)
    assert computeLRF(t1, t2) == 0


def test_single_label_mismatch_costs_one():
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_ONE_FLIPPED)
    assert computeLRF(t1, t2) == 1


def test_two_label_mismatches_cost_two():
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_BOTH_FLIPPED)
    assert computeLRF(t1, t2) == 2


def test_label_mismatch_distance_is_symmetric():
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_ONE_FLIPPED)
    assert computeLRF(t1, t2) == computeLRF(t2, t1) == 1


def test_star_vs_star_has_zero_distance():
    t1, t2 = parse_pair(STAR4, STAR4)
    assert computeLRF(t1, t2) == 0


def test_resolved_quartet_vs_star_costs_two():
    star, resolved = parse_pair(STAR4, QUARTET_LABELED)
    assert computeLRF(star, resolved) == 2
    assert computeLRF(resolved, star) == 2


def test_single_resolved_node_vs_star_costs_one():
    resolved, star = parse_pair("[&R] ((A,B)speciation,C,D);", STAR4)
    assert computeLRF(resolved, star) == 1
    assert computeLRF(star, resolved) == 1


def test_quartet_topologies_without_shared_splits():
    # no bipartition is shared, so the whole tree is one island on each side
    # (3 internal nodes each): 3 + 3 - 2 = 4, and the root labels intersect
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_ALT_TOPOLOGY)
    assert computeLRF(t1, t2) == 4


def test_compute_returns_int_and_leaves_inputs_unchanged():
    t1, t2 = parse_pair(QUARTET_LABELED, QUARTET_BOTH_FLIPPED)
    before1 = clade_labels(t1)
    before2 = clade_labels(t2)
    result = computeLRF(t1, t2)
    assert isinstance(result, int)
    assert result == 2
    assert clade_labels(t1) == before1
    assert clade_labels(t2) == before2
    assert leaf_names(t1) == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_zero_edits_returns_equivalent_copy():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    random.seed(0)
    mutated = mutateLabeledTree(t, 0)
    assert mutated is not t
    assert clade_labels(mutated) == clade_labels(t)


def test_mutation_returns_new_tree_and_leaves_input_unchanged():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    snapshot = clade_labels(t)
    random.seed(1)
    mutated = mutateLabeledTree(t, 3, p_flip=1.0)
    assert mutated is not t
    assert clade_labels(t) == snapshot


def test_single_flip_changes_exactly_one_label():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    before = clade_labels(t)
    random.seed(2)
    mutated = mutateLabeledTree(t, 1, p_flip=1.0)
    after = clade_labels(mutated)

    assert set(after.keys()) == set(before.keys())
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1
    old, new = before[changed[0]], after[changed[0]]
    assert {old, new} == {"speciation", "duplication"}


def test_single_flip_mutation_is_distance_one_under_computeLRF():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    random.seed(3)
    mutated = mutateLabeledTree(t, 1, p_flip=1.0)
    assert computeLRF(t, mutated) == 1


def test_collapse_only_removes_one_internal_node():
    # 4-taxon rooted tree: only collapses are possible with p_flip=0
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    assert len(list(t.internal_nodes())) == 3
    random.seed(4)
    mutated = mutateLabeledTree(t, 1, p_flip=0.0)
    assert len(list(mutated.internal_nodes())) == 2
    assert leaf_names(mutated) == ["A", "B", "C", "D"]


def test_expansion_on_star_adds_one_resolved_clade():
    star = parse(STAR4, dendropy.TaxonNamespace())
    random.seed(5)
    mutated = mutateLabeledTree(star, 1, p_flip=0.0)

    assert leaf_names(mutated) == ["A", "B", "C", "D"]
    inner = [n for n in mutated.internal_nodes() if n.parent_node is not None]
    assert len(inner) == 1
    assert len(list(inner[0].leaf_iter())) in (2, 3)


def test_many_edits_preserve_leaves_and_clade_sizes():
    star = parse(STAR6, dendropy.TaxonNamespace())
    random.seed(6)
    mutated = mutateLabeledTree(star, 5, p_flip=0.0)

    assert leaf_names(mutated) == ["A", "B", "C", "D", "E", "F"]
    for node in mutated.internal_nodes():
        assert len(list(node.leaf_iter())) >= 2


def test_same_seed_gives_same_mutation():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    for seed in range(5):
        random.seed(seed)
        first = mutateLabeledTree(t, 1, p_flip=0.5)
        random.seed(seed)
        second = mutateLabeledTree(t, 1, p_flip=0.5)
        assert clade_labels(first) == clade_labels(second)


def test_elrf_collapses_only_edge_with_matching_parent_label():
    # X=(A,B) and its parent Z share the label, so X is the only
    # collapsible edge under the ELRF model
    t = parse(
        "[&R] (((A,B)speciation,C)speciation,D);",
        dendropy.TaxonNamespace(),
    )
    random.seed(7)
    mutated = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    keys = set(clade_labels(mutated).keys())
    assert frozenset({"A", "B", "C"}) in keys
    assert frozenset({"A", "B"}) not in keys
    assert len(list(mutated.internal_nodes())) == 2


# ---------------------------------------------------------------------------
# collapse_node / flip_node (helpers used by mutateLabeledTree)
# ---------------------------------------------------------------------------

def test_elrf_collapse_rejects_edge_with_different_parent_label():
    t = parse(
        "[&R] (((A,B)speciation,C)speciation,D);",
        dendropy.TaxonNamespace(),
    )
    z = find_node(t, ["A", "B", "C"])  # label 'speciation', parent root has None
    with pytest.raises(ValueError):
        collapse_node(z, model="ELRF")


def test_collapse_node_rejects_unknown_model():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    x = find_node(t, ["A", "B"])
    with pytest.raises(ValueError):
        collapse_node(x, model="unknown")


def test_flip_node_toggles_speciation_and_duplication():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    x = find_node(t, ["A", "B"])
    y = find_node(t, ["C", "D"])
    assert x.label == "speciation"
    assert y.label == "duplication"
    flip_node(x)
    flip_node(y)
    assert x.label == "duplication"
    assert y.label == "speciation"


def test_flip_node_rejects_unlabeled_node():
    t = parse(QUARTET_LABELED, dendropy.TaxonNamespace())
    with pytest.raises(ValueError):
        flip_node(t.seed_node)  # root has no label
