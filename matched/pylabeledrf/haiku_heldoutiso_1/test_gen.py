import os
import sys
import random

import dendropy
import pytest

# Make the project importable whether pytest is launched from the project root
# or from a subdirectory.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


def _tree(newick, namespace=None):
    ns = namespace if namespace is not None else dendropy.TaxonNamespace()
    return dendropy.Tree.get(data=newick, schema="newick",
                             taxon_namespace=ns, rooted=True)


def _pair(newick1, newick2):
    # both trees must share one taxon namespace for bipartition comparison
    ns = dendropy.TaxonNamespace()
    return _tree(newick1, ns), _tree(newick2, ns)


def _internal_labels(tree):
    return [n.label for n in tree.internal_nodes()]


def _leaf_labels(tree):
    return sorted(l.taxon.label for l in tree.leaf_node_iter())


BASE = "((A,B)speciation,(C,D)duplication)"
POLY = "((A,B)speciation,C,D)"
BIG = "(((A,B)speciation,C)duplication,((D,E)speciation,F)duplication,G)"


# ---------------------------------------------------------------- computeLRF

def test_identical_trees_have_zero_distance():
    t1, t2 = _pair(BASE, BASE)
    assert computeLRF(t1, t2) == 0


def test_leaf_order_does_not_matter():
    t1, t2 = _pair("((A,B)speciation,(C,D)duplication)",
                   "((D,C)duplication,(B,A)speciation)")
    assert computeLRF(t1, t2) == 0


def test_single_label_change_costs_one():
    t1, t2 = _pair(BASE, "((A,B)duplication,(C,D)duplication)")
    assert computeLRF(t1, t2) == 1


def test_two_label_changes_cost_two():
    t1, t2 = _pair(BASE, "((A,B)duplication,(C,D)speciation)")
    assert computeLRF(t1, t2) == 2


def test_root_label_mismatch_costs_one():
    t1, t2 = _pair(BASE, "((A,B)speciation,(C,D)duplication)speciation")
    assert computeLRF(t1, t2) == 1


def test_deep_internal_relabel_costs_one():
    t1, t2 = _pair("(((A,B)speciation,C)duplication,(D,E)speciation)",
                   "(((A,B)speciation,C)speciation,(D,E)speciation)")
    assert computeLRF(t1, t2) == 1


def test_one_collapse_costs_one():
    t1, t2 = _pair(POLY, "((A,B)speciation,(C,D)duplication)")
    assert computeLRF(t1, t2) == 1


def test_collapse_distance_is_symmetric():
    t1, t2 = _pair(POLY, "((A,B)speciation,(C,D)duplication)")
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_star_versus_resolved_costs_two():
    t1, t2 = _pair(BASE, "(A,B,C,D)")
    assert computeLRF(t1, t2) == 2


def test_same_labels_different_topology_costs_four():
    t1, t2 = _pair(BASE, "((A,C)speciation,(B,D)duplication)")
    assert computeLRF(t1, t2) == 4


def test_symmetry_on_label_change():
    t1, t2 = _pair(BASE, "((A,B)duplication,(C,D)duplication)")
    assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_computelrf_returns_integer():
    t1, t2 = _pair(BASE, "((A,B)duplication,(C,D)duplication)")
    result = computeLRF(t1, t2)
    assert isinstance(result, int)


def test_computelrf_does_not_modify_inputs():
    t1, t2 = _pair(POLY, "((A,C)speciation,(B,D)duplication)")
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# ------------------------------------------------------- mutateLabeledTree

def test_zero_edits_returns_equal_copy():
    t = _tree(BIG)
    out = mutateLabeledTree(t, 0)
    assert out is not t
    assert _internal_labels(out) == _internal_labels(t)
    assert _leaf_labels(out) == _leaf_labels(t)
    assert computeLRF(t, out) == 0


def test_input_tree_is_left_unchanged():
    t = _tree(BIG)
    before = t.as_string("newick")
    random.seed(3)
    mutateLabeledTree(t, 2, p_flip=0.5)
    assert t.as_string("newick") == before


def test_single_flip_changes_exactly_one_label():
    t = _tree(BASE)
    random.seed(1)
    out = mutateLabeledTree(t, 1, p_flip=1.0)
    orig = _internal_labels(t)
    new = _internal_labels(out)
    assert len(orig) == len(new)
    diffs = [(a, b) for a, b in zip(orig, new) if a != b]
    assert len(diffs) == 1
    a, b = diffs[0]
    assert {a, b} == {"speciation", "duplication"}


def test_single_flip_has_lrf_distance_one():
    t = _tree(BASE)
    random.seed(1)
    out = mutateLabeledTree(t, 1, p_flip=1.0)
    assert computeLRF(t, out) == 1


@pytest.mark.parametrize("seed", list(range(10)))
def test_single_contraction_or_expansion_costs_one(seed):
    # On this tree, the only possible edits are collapsing the AB node
    # or expanding the root; both should cost exactly one in LRF.
    t = _tree(POLY)
    random.seed(seed)
    out = mutateLabeledTree(t, 1, p_flip=0.0)
    assert _leaf_labels(out) == _leaf_labels(t)
    assert computeLRF(t, out) == 1


def test_multiple_edits_preserve_taxa():
    t = _tree(BIG)
    random.seed(11)
    out = mutateLabeledTree(t, 3, p_flip=0.3)
    assert isinstance(out, dendropy.Tree)
    assert _leaf_labels(out) == _leaf_labels(t)


def test_same_seed_gives_same_output():
    t = _tree(BIG)
    random.seed(7)
    out1 = mutateLabeledTree(t, 3, p_flip=0.3)
    random.seed(7)
    out2 = mutateLabeledTree(t, 3, p_flip=0.3)
    assert out1.as_string("newick") == out2.as_string("newick")
