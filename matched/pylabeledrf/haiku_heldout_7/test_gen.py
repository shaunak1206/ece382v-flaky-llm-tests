```python
import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def new_taxa():
    return dendropy.TaxonNamespace()


def make_tree(newick, taxa, labels=None, default="speciation"):
    """
    Build a rooted labelled tree. Internal nodes get `default` unless their
    leaf set is a key in `labels`; the root (seed node) gets label None.
    All trees that are compared must share the same TaxonNamespace `taxa`.
    """
    labels = labels or {}
    tree = dendropy.Tree.get_from_string(
        "[&R] " + newick + ";", "newick", taxon_namespace=taxa)
    for node in tree.internal_nodes():
        key = frozenset(leaf.taxon.label for leaf in node.leaf_iter())
        node.label = labels.get(key, default)
    tree.seed_node.label = None
    return tree


def label_map(tree):
    """Map the leaf set of each internal node to its label."""
    return {
        frozenset(leaf.taxon.label for leaf in node.leaf_iter()): node.label
        for node in tree.internal_nodes()
    }


def leaf_names(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_nodes())


def internal_count(tree):
    return len(list(tree.internal_nodes()))


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("newick", [
    "((A,B),(C,D))",
    "((A,B),C,(D,E))",
    "(((A,B),C),(D,E))",
    "((A,B),(C,D),E,F)",
])
def test_identical_trees_have_zero_distance(newick):
    taxa = new_taxa()
    t1 = make_tree(newick, taxa)
    t2 = make_tree(newick, taxa)
    assert computeLRF(t1, t2) == 0


def test_identical_trees_with_mixed_labels_have_zero_distance():
    taxa = new_taxa()
    labels = {frozenset({"A", "B"}): "duplication",
              frozenset({"D", "E"}): "speciation"}
    t1 = make_tree("((A,B),C,(D,E))", taxa, labels)
    t2 = make_tree("((A,B),C,(D,E))", taxa, labels)
    assert computeLRF(t1, t2) == 0


def test_tree_compared_with_itself_has_zero_distance():
    taxa = new_taxa()
    labels = {frozenset({"A", "B"}): "duplication"}
    t = make_tree("((A,B),C,(D,E))", taxa, labels)
    assert computeLRF(t, t) == 0


def test_single_label_mismatch_costs_one_in_both_directions():
    taxa = new_taxa()
    t1 = make_tree("((A,B),(C,D))", taxa)
    t2 = make_tree("((A,B),(C,D))", taxa,
                   {frozenset({"A", "B"}): "duplication"})
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_two_label_mismatches_cost_two():
    taxa = new_taxa()
    t1 = make_tree("((A,B),(C,D))", taxa)
    t2 = make_tree("((A,B),(C,D))", taxa, {
        frozenset({"A", "B"}): "duplication",
        frozenset({"C", "D"}): "duplication",
    })
    assert computeLRF(t1, t2) == 2
    assert computeLRF(t2, t1) == 2


def test_single_extra_internal_node_costs_one_in_both_directions():
    taxa = new_taxa()
    t1 = make_tree("((A,B),C,(D,E))", taxa)
    t2 = make_tree("(((A,B),C),(D,E))", taxa)
    assert computeLRF(t1, t2) == 1
    assert computeLRF(t2, t1) == 1


def test_incompatible_topologies_with_labels_cost_four():
    taxa = new_taxa()
    t1 = make_tree("((A,B),(C,D))", taxa,
                   {frozenset({"A", "B"}): "duplication"})
    t2 = make_tree("((A,C),(B,D))", taxa)
    # Two rooted clades differ on each side: 2 collapses per tree, labels
    # of the remaining star still share a speciation/root label.
    assert computeLRF(t1, t2) == 4
    assert computeLRF(t2, t1) == 4


def test_star_versus_resolved_tree_costs_two():
    taxa = new_taxa()
    star = make_tree("(A,B,C,D)", taxa)
    resolved = make_tree("((A,B),(C,D))", taxa)
    assert computeLRF(star, resolved) == 2
    assert computeLRF(resolved, star) == 2


def test_computeLRF_returns_integer():
    taxa = new_taxa()
    t1 = make_tree("((A,B),(C,D))", taxa)
    t2 = make_tree("((A,C),(B,D))", taxa)
    result = computeLRF(t1, t2)
    assert isinstance(result, int)


def test_computeLRF_does_not_modify_inputs():
    taxa = new_taxa()
    labels = {frozenset({"A", "B"}): "duplication"}
    t1 = make_tree("((A,B),C,(D,E))", taxa, labels)
    t2 = make_tree("(((A,B),C),(D,E))", taxa)
    before1, before2 = t1.as_string("newick"), t2.as_string("newick")
    labels1, labels2 = label_map(t1), label_map(t2)

    computeLRF(t1, t2)

    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2
    assert label_map(t1) == labels1
    assert label_map(t2) == labels2


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_mutate_returns_new_tree_and_leaves_input_unchanged():
    taxa = new_taxa()
    src = make_tree("((A,B),C,(D,E))", taxa)
    before_str = src.as_string("newick")
    before_labels = label_map(src)

    out = mutateLabeledTree(src, 3, p_flip=0.0)

    assert out is not src
    assert src.as_string("newick") == before_str
    assert label_map(src) == before_labels


def test_zero_edits_preser