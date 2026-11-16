```python
import random

import dendropy
import pytest

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree

SPEC = "speciation"
DUP = "duplication"


@pytest.fixture
def taxa():
    # All trees in a test must share one namespace so bipartition bitmasks are comparable.
    return dendropy.TaxonNamespace()


def make_tree(newick, labels, taxa):
    """Parse a rooted Newick string and label internal nodes by their leaf sets."""
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=taxa)
    for node in tree.internal_nodes():
        clade = frozenset(leaf.taxon.label for leaf in node.leaf_iter())
        node.label = labels.get(clade)
    return tree


def clade_labels(tree):
    """Map each internal node's leaf set to its label."""
    return {
        frozenset(leaf.taxon.label for leaf in node.leaf_iter()): node.label
        for node in tree.internal_nodes()
    }


def leaf_set(tree):
    return frozenset(leaf.taxon.label for leaf in tree.leaf_nodes())


def four_leaf(taxa, ab=SPEC, cd=DUP):
    return make_tree(
        "[&R] ((A,B),(C,D));",
        {frozenset(["A", "B"]): ab, frozenset(["C", "D"]): cd},
        taxa,
    )


def five_leaf_nested(taxa, abc=DUP):
    # (((A,B),C),(D,E)) with labels ABC=abc, AB=speciation, DE=speciation
    return make_tree(
        "[&R] (((A,B),C),(D,E));",
        {
            frozenset(["A", "B", "C"]): abc,
            frozenset(["A", "B"]): SPEC,
            frozenset(["D", "E"]): SPEC,
        },
        taxa,
    )


def five_leaf_ternary(taxa):
    # ((A,B,C),(D,E)) with ABC=speciation, DE=duplication
    return make_tree(
        "[&R] ((A,B,C),(D,E));",
        {frozenset(["A", "B", "C"]): SPEC, frozenset(["D", "E"]): DUP},
        taxa,
    )


# ---------------------------------------------------------------- computeLRF

def test_identical_trees_have_zero_distance(taxa):
    t1 = four_leaf(taxa)
    t2 = four_leaf(taxa)
    assert computeLRF(t1, t2) == 0


def test_identical_trees_with_trichotomous_root_have_zero_distance(taxa):
    t1 = make_tree("[&R] (A,B,(C,D));", {frozenset(["C", "D"]): SPEC}, taxa)
    t2 = make_tree("[&R] (A,B,(C,D));", {frozenset(["C", "D"]): SPEC}, taxa)
    assert computeLRF(t1, t2) == 0


def test_single_label_flip_costs_one(taxa):
    t1 = four_leaf(taxa, ab=SPEC, cd=DUP)
    t2 = four_leaf(taxa, ab=DUP, cd=DUP)
    assert computeLRF(t1, t2) == 1


def test_flipping_both_labels_costs_two(taxa):
    t1 = four_leaf(taxa, ab=SPEC, cd=DUP)
    t2 = four_leaf(taxa, ab=DUP, cd=SPEC)
    assert computeLRF(t1, t2) == 2


def test_flip_of_nested_node_in_five_leaf_tree_costs_one(taxa):
    t1 = five_leaf_nested(taxa, abc=DUP)
    t2 = five_leaf_nested(taxa, abc=SPEC)
    assert computeLRF(t1, t2) == 1


def test_different_topologies_give_standard_rf_of_four(taxa):
    # Two rooted 4-leaf trees sharing no nontrivial clade: standard RF = 4
    t1 = four_leaf(taxa, ab=SPEC, cd=SPEC)
    t2 = make_tree(
        "[&R] ((A,C),(B,D));",
        {frozenset(["A", "C"]): SPEC, frozenset(["B", "D"]): SPEC},
        taxa,
    )
    assert computeLRF(t1, t2) == 4


def test_extra_internal_node_costs_one(taxa):
    t1 = make_tree("[&R] ((A,B),C,D);", {frozenset(["A", "B"]): SPEC}, taxa)
    t2 = make_tree(
        "[&R] ((A,B),(C