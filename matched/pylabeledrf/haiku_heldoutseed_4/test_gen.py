import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

AB = frozenset(["A", "B"])
CD = frozenset(["C", "D"])
DE = frozenset(["D", "E"])
AC = frozenset(["A", "C"])


@pytest.fixture
def taxa():
    # both trees in a comparison must share one taxon namespace
    return dendropy.TaxonNamespace()


def make_tree(newick, taxa, root_label=None, clade_labels=None, default="speciation"):
    """Parse a rooted newick tree and label its internal nodes.

    clade_labels maps frozenset(leaf names) -> label. Unlisted non-root
    internal nodes get `default`. The root gets `root_label`.
    """
    tree = dendropy.Tree.get_from_string(
        newick, "newick", rooted=True, taxon_namespace=taxa
    )
    clade_labels = clade_labels or {}
    for node in tree.internal_nodes():
        if node is tree.seed_node:
            node.label = root_label
        else:
            key = frozenset(leaf.taxon.label for leaf in node.leaf_nodes())
            node.label = clade_labels.get(key, default)
    return tree


def clade_labels(tree):
    """Map each non-root internal clade (as frozenset of leaves) to its label."""
    return {
        frozenset(leaf.taxon.label for leaf in n.leaf_nodes()): n.label
        for n in tree.internal_nodes()
        if n.parent_node is not None
    }


def leafset(tree):
    return frozenset(leaf.taxon.label for leaf in tree.leaf_nodes())


BASE5 = "((A,B),C,(D,E));"


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

def test_identical_trees_have_zero_distance(taxa):
    t = make_tree(BASE5, taxa, clade_labels={AB: "speciation", DE: "duplication"})
    assert computeLRF(t, t) == 0


def test_identical_stars_have_zero_distance(taxa):
    t = make_tree("(A,B,C,D,E);", taxa)
    assert computeLRF(t, t) == 0


def test_star_versus_one_resolution_costs_one(taxa):
    star = make_tree("(A,B,C,D,E);", taxa)
    resolved = make_tree("((A,B),C,D,E);", taxa)
    assert computeLRF(star, resolved) == 1


def test_single_label_flip_costs_one(taxa):
    t1 = make_tree(BASE5, taxa)
    t2 = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    assert computeLRF(t1, t2) == 1


def test_two_label_flips_cost_two(taxa):
    t1 = make_tree(BASE5, taxa)
    t2 = make_tree(BASE5, taxa, clade_labels={AB: "duplication", DE: "duplication"})
    assert computeLRF(t1, t2) == 2


def test_collapsing_one_node_costs_one(taxa):
    t1 = make_tree("(((A,B),C),D,E);", taxa)
    t2 = make_tree("((A,B),C,D,E);", taxa)
    assert computeLRF(t1, t2) == 1


def test_topology_change_costs_two(taxa):
    # ((A,B),C,(D,E)) vs ((A,C),B,(D,E)): one nontrivial split replaced
    t1 = make_tree(BASE5, taxa)
    t2 = make_tree("((A,C),B,(D,E));", taxa)
    assert computeLRF(t1, t2) == 2


def test_distance_is_symmetric_for_topology_change(taxa):
    t1 = make_tree(BASE5, taxa)
    t2 = make_tree("((A,C),B,(D,E));", taxa)
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_distance_is_symmetric_for_mixed_changes(taxa):
    t1 = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    t2 = make_tree("((A,C),B,(D,E));", taxa, clade_labels={DE: "duplication"})
    d12 = computeLRF(t1, t2)
    d21 = computeLRF(t2, t1)
    assert d12 == d21
    assert isinstance(d12, int) and d12 >= 0


def test_inputs_are_not_modified(taxa):
    t1 = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    t2 = make_tree("((A,C),B,(D,E));", taxa)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


def test_returns_integer(taxa):
    t1 = make_tree(BASE5, taxa)
    t2 = make_tree(BASE5, taxa, clade_labels={DE: "duplication"})
    result = computeLRF(t1, t2)
    assert isinstance(result, int)
    assert result == 1


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_mutate_returns_new_tree_and_leaves_input_unchanged(taxa):
    tree = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    before = tree.as_string("newick")
    random.seed(0)
    result = mutateLabeledTree(tree, 3, p_flip=0.5)
    assert result is not tree
    assert tree.as_string("newick") == before


def test_mutate_zero_edits_returns_equivalent_tree(taxa):
    tree = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    result = mutateLabeledTree(tree, 0)
    assert clade_labels(result) == clade_labels(tree)
    assert leafset(result) == leafset(tree)


def test_single_flip_changes_exactly_one_label(taxa):
    tree = make_tree(BASE5, taxa, clade_labels={AB: "speciation", DE: "duplication"})
    before = clade_labels(tree)
    random.seed(1)
    result = mutateLabeledTree(tree, 1, p_flip=1.0)
    after = clade_labels(result)
    assert set(after) == set(before)
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1
    k = changed[0]
    assert {before[k], after[k]} == {"speciation", "duplication"}


def test_flips_preserve_topology(taxa):
    tree = make_tree(BASE5, taxa)
    random.seed(2)
    result = mutateLabeledTree(tree, 6, p_flip=1.0)
    assert set(clade_labels(result)) == set(clade_labels(tree))
    assert leafset(result) == leafset(tree)


def test_mutate_is_reproducible_with_seed(taxa):
    tree = make_tree(BASE5, taxa, clade_labels={AB: "duplication"})
    random.seed(42)
    r1 = mutateLabeledTree(tree, 4, p_flip=0.0)
    random.seed(42)
    r2 = mutateLabeledTree(tree, 4, p_flip=0.0)
    assert r1.as_string("newick") == r2.as_string("newick")


def test_collapse_only_removes_one_internal_node(taxa):
    tree = make_tree("((A,B),(C,D));", taxa)
    assert len(tree.internal_nodes()) == 3
    random.seed(3)
    result = mutateLabeledTree(tree, 1, p_flip=0.0)
    # one of the two non-root internal nodes is collapsed
    assert len(result.internal_nodes()) == 2
    assert leafset(result) == leafset(tree)


def test_expansion_only_on_wide_root(taxa):
    tree = make_tree("(A,B,C,D,E);", taxa)
    random.seed(4)
    result = mutateLabeledTree(tree, 1, p_flip=0.0)
    assert len(result.internal_nodes()) == 2
    root_children = len(result.seed_node.child_nodes())
    assert 2 <= root_children <= 4
    assert leafset(result) == leafset(tree)


def test_elrf_only_collapses_edges_matching_parent_label(taxa):
    # AB has the same label as the root and may be collapsed;
    # CD differs from the root and must not be collapsed under ELRF.
    tree = make_tree(
        "((A,B),(C,D));",
        taxa,
        root_label="speciation",
        clade_labels={AB: "speciation", CD: "duplication"},
    )
    random.seed(5)
    result = mutateLabeledTree(tree, 1, p_flip=0.0, model="ELRF")
    assert clade_labels(result) == {CD: "duplication"}


def test_unsupported_model_raises_when_collapse_is_chosen(taxa):
    tree = make_tree(
        "((A,B),(C,D));",
        taxa,
        root_label="speciation",
        clade_labels={AB: "speciation", CD: "duplication"},
    )
    random.seed(6)
    with pytest.raises(ValueError):
        mutateLabeledTree(tree, 1, p_flip=0.0, model="foo")
