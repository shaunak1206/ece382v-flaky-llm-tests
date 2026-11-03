"""Unit tests for computeLRF and mutateLabeledTree in pylabeledrf.computeLRF."""

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

# All trees are rooted and use a trifurcating root, so that no two root
# children are complementary (their splits would otherwise collide).
BASE_NEWICK = (
    "[&R] ((A,B)speciation,(C,D)duplication,"
    "(E,(F,G)speciation)duplication);"
)
REORDERED_NEWICK = (
    "[&R] ((C,D)duplication,((F,G)speciation,E)duplication,"
    "(B,A)speciation);"
)
AB_FLIPPED_NEWICK = (
    "[&R] ((A,B)duplication,(C,D)duplication,"
    "(E,(F,G)speciation)duplication);"
)
TWO_FLIPS_NEWICK = (
    "[&R] ((A,B)duplication,(C,D)speciation,"
    "(E,(F,G)speciation)duplication);"
)
PAIR_AB_CD_NEWICK = "[&R] ((A,B)speciation,(C,D)speciation,(E,F)speciation);"
PAIR_AC_BD_NEWICK = "[&R] ((A,C)speciation,(B,D)speciation,(E,F)speciation);"
STAR_NEWICK = "[&R] (A,B,C,D,E,F);"
SMALL_NEWICK = "[&R] ((A,B)speciation,(C,(D,E)speciation)speciation,F);"


def parse_tree(newick, namespace):
    return dendropy.Tree.get(
        data=newick, schema="newick", taxon_namespace=namespace
    )


def parse_pair(newick1, newick2):
    namespace = dendropy.TaxonNamespace()
    return parse_tree(newick1, namespace), parse_tree(newick2, namespace)


def base_tree():
    return parse_tree(BASE_NEWICK, dendropy.TaxonNamespace())


def newick_of(tree):
    return tree.as_string(schema="newick")


def leaf_labels_under(node):
    if node.is_leaf():
        return frozenset([node.taxon.label])
    labels = set()
    for child in node.child_node_iter():
        labels |= leaf_labels_under(child)
    return frozenset(labels)


def labeled_clades(tree):
    """Map each internal node's clade (frozenset of leaf names) to its label."""
    return {leaf_labels_under(n): n.label for n in tree.internal_nodes()}


def leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_nodes())


def find_clade(tree, names):
    target = frozenset(names)
    for node in tree.internal_nodes():
        if leaf_labels_under(node) == target:
            return node
    raise AssertionError("no clade %s in tree" % sorted(target))


# ---------------------------------------------------------------- computeLRF


def test_lrf_identical_trees_is_zero():
    t1, t2 = parse_pair(BASE_NEWICK, BASE_NEWICK)
    assert computeLRF(t1, t2) == 0


def test_lrf_reordered_children_is_zero():
    t1, t2 = parse_pair(BASE_NEWICK, REORDERED_NEWICK)
    assert computeLRF(t1, t2) == 0


def test_lrf_single_label_flip_is_one():
    t1, t2 = parse_pair(BASE_NEWICK, AB_FLIPPED_NEWICK)
    assert computeLRF(t1, t2) == 1


def test_lrf_two_label_flips_is_two():
    t1, t2 = parse_pair(BASE_NEWICK, TWO_FLIPS_NEWICK)
    assert computeLRF(t1, t2) == 2


def test_lrf_topology_swap_is_four_both_ways():
    # Same labels everywhere; the two clade sets {AB,CD,EF} and {AC,BD,EF}
    # differ in four clades, so the rooted RF distance is 4.
    t1, t2 = parse_pair(PAIR_AB_CD_NEWICK, PAIR_AC_BD_NEWICK)
    assert computeLRF(t1, t2) == 4
    assert computeLRF(t2, t1) == 4


def test_lrf_star_vs_resolved_is_three_both_ways():
    # Resolving a 6-leaf star with 3 internal clades costs 3 collapses.
    star, resolved = parse_pair(STAR_NEWICK, PAIR_AB_CD_NEWICK)
    assert computeLRF(star, resolved) == 3
    assert computeLRF(resolved, star) == 3


def test_lrf_does_not_modify_inputs():
    t1, t2 = parse_pair(BASE_NEWICK, TWO_FLIPS_NEWICK)
    before1 = newick_of(t1)
    before2 = newick_of(t2)
    assert computeLRF(t1, t2) == 2
    assert newick_of(t1) == before1
    assert newick_of(t2) == before2


def test_lrf_equals_label_mismatches_after_mutation():
    # With p_flip=1 the topology is untouched, so LRF should count exactly
    # the clades whose label changed.
    for seed in range(5):
        random.seed(seed)
        tree = base_tree()
        mutated = mutateLabeledTree(tree, 3, p_flip=1.0)
        before = labeled_clades(tree)
        after = labeled_clades(mutated)
        assert set(before) == set(after)
        mismatches = sum(1 for k in before if before[k] != after[k])
        assert computeLRF(tree, mutated) == mismatches


# ------------------------------------------------------- mutateLabeledTree


def test_mutate_zero_edits_keeps_clades_and_labels():
    tree = base_tree()
    out = mutateLabeledTree(tree, 0)
    assert out is not tree
    assert isinstance(out, dendropy.Tree)
    assert labeled_clades(out) == labeled_clades(tree)


def test_mutate_does_not_modify_input():
    random.seed(1)
    tree = base_tree()
    before = newick_of(tree)
    mutateLabeledTree(tree, 6, p_flip=0.5)
    assert newick_of(tree) == before


def test_mutate_flip_only_changes_exactly_one_label():
    random.seed(0)
    tree = base_tree()
    before = labeled_clades(tree)
    out = mutateLabeledTree(tree, 1, p_flip=1.0)
    after = labeled_clades(out)
    assert set(after) == set(before)
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1
    assert {before[changed[0]], after[changed[0]]} == {
        "speciation",
        "duplication",
    }


def test_mutate_flip_only_keeps_topology():
    random.seed(7)
    tree = base_tree()
    before = labeled_clades(tree)
    out = mutateLabeledTree(tree, 25, p_flip=1.0)
    after = labeled_clades(out)
    assert set(after) == set(before)
    assert leaf_names(out) == leaf_names(tree)


def test_mutate_structural_edit_changes_size_by_one():
    # p_flip=0 means each edit is a collapse (-1 internal node) or an
    # extension (+1 internal node).
    for seed in range(20):
        random.seed(seed)
        tree = base_tree()
        before_count = len(tree.internal_nodes())
        out = mutateLabeledTree(tree, 1, p_flip=0.0)
        assert abs(len(out.internal_nodes()) - before_count) == 1
        assert leaf_names(out) == leaf_names(tree)


def test_mutate_elrf_extends_when_no_matching_labels():
    # In BASE no non-root node has the same label as its parent, so the
    # ELRF model has nothing to collapse and must extend the root.
    for seed in range(10):
        random.seed(seed)
        tree = base_tree()
        before_count = len(tree.internal_nodes())
        out = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
        assert len(out.internal_nodes()) == before_count + 1
        assert leaf_names(out) == leaf_names(tree)


def test_mutate_preserves_leaves_and_min_degree():
    for seed in range(5):
        random.seed(seed)
        tree = base_tree()
        out = mutateLabeledTree(tree, 8, p_flip=0.3)
        assert leaf_names(out) == leaf_names(tree)
        for node in out.internal_nodes():
            assert len(node.child_nodes()) >= 2


def test_mutate_same_seed_is_reproducible():
    def run(seed):
        random.seed(seed)
        return newick_of(mutateLabeledTree(base_tree(), 8, p_flip=0.4))

    assert run(123) == run(123)


def test_mutate_unknown_model_raises():
    random.seed(0)
    tree = base_tree()
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="bogus")


# ------------------------------------------------- single-edit helper tests


def test_collapse_node_lrf_removes_clade():
    tree = base_tree()
    node = find_clade(tree, {"F", "G"})
    before_count = len(tree.internal_nodes())
    collapse_node(node, model="LRF")
    assert len(tree.internal_nodes()) == before_count - 1
    clades = labeled_clades(tree)
    assert frozenset({"F", "G"}) not in clades
    assert frozenset({"E", "F", "G"}) in clades


def test_collapse_node_elrf_checks_parent_label():
    tree = base_tree()
    # (F,G) is "speciation" under a "duplication" parent: not collapsible.
    fg = find_clade(tree, {"F", "G"})
    before_count = len(tree.internal_nodes())
    with pytest.raises(ValueError):
        collapse_node(fg, model="ELRF")
    assert len(tree.internal_nodes()) == before_count

    # (D,E) is "speciation" under a "speciation" parent: collapsible.
    small = parse_tree(SMALL_NEWICK, dendropy.TaxonNamespace())
    de = find_clade(small, {"D", "E"})
    before_small = len(small.internal_nodes())
    collapse_node(de, model="ELRF")
    assert len(small.internal_nodes()) == before_small - 1
    assert frozenset({"D", "E"}) not in labeled_clades(small)


def test_flip_node_swaps_speciation_and_duplication():
    node = dendropy.Node(label="speciation")
    flip_node(node)
    assert node.label == "duplication"
    flip_node(node)
    assert node.label == "speciation"


def test_flip_node_rejects_unlabeled_node():
    with pytest.raises(ValueError):
        flip_node(dendropy.Node())


def test_extend_node_rejects_bad_model_and_low_degree():
    with pytest.raises(ValueError):
        extend_node(dendropy.Node(), model="bogus")

    # (A,B) has a parent plus two children: degree 3, too small to split.
    tree = base_tree()
    ab = find_clade(tree, {"A", "B"})
    with pytest.raises(ValueError):
        extend_node(ab)


def test_extend_root_adds_one_node_and_keeps_leaves():
    # The root has three children, so k is forced to 2: the root keeps
    # one old child plus the new node, and the new node takes two.
    random.seed(3)
    tree = base_tree()
    before_leaves = leaf_names(tree)
    before_count = len(tree.internal_nodes())
    root = tree.seed_node
    extend_node(root, model="LRF")
    assert leaf_names(tree) == before_leaves
    assert len(tree.internal_nodes()) == before_count + 1
    assert len(root.child_nodes()) == 2
