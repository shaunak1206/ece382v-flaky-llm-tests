"""Unit tests for computeLRF and mutateLabeledTree in pylabeledrf/computeLRF.py.

Trees are built from Newick strings with rooted=True and a fresh TaxonNamespace
per test, so bipartition bitmasks are deterministic. Every test seeds the
stdlib ``random`` module (the only RNG imported by the module under test) at the
start of its body, and any replacement of module attributes goes through
pytest's monkeypatch fixture.
"""
import importlib
import random

import dendropy

# Import the module by its full name: the package may re-export a function
# called computeLRF, which would shadow the module attribute.
mod = importlib.import_module("pylabeledrf.computeLRF")
computeLRF = mod.computeLRF
mutateLabeledTree = mod.mutateLabeledTree

SPEC = "speciation"
DUP = "duplication"


def _parse(newick, taxa):
    return dendropy.Tree.get(
        data=newick, schema="newick", rooted=True, taxon_namespace=taxa
    )


def _leaves(tree):
    return sorted(leaf.taxon.label for leaf in tree.leaf_node_iter())


def _clade_labels(tree):
    """Map each internal clade (sorted leaf-label tuple) to its node label."""
    result = {}
    for node in tree.internal_nodes():
        key = tuple(sorted(leaf.taxon.label for leaf in node.leaf_iter()))
        result[key] = node.label
    return result


def _n_internal(tree):
    return len(list(tree.internal_nodes()))


# ---------------------------------------------------------------- computeLRF

def test_identical_resolved_trees_have_zero_distance():
    random.seed(1)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    assert computeLRF(t, t) == 0


def test_identical_trees_with_duplication_labels_have_zero_distance():
    random.seed(2)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)duplication;", taxa)
    assert computeLRF(t, t) == 0


def test_identical_star_trees_have_zero_distance():
    random.seed(3)
    taxa = dendropy.TaxonNamespace()
    t = _parse("(A,B,C,D)speciation;", taxa)
    assert computeLRF(t, t) == 0


def test_root_label_difference_costs_one():
    random.seed(4)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _parse("((A,B)speciation,(C,D)speciation)duplication;", taxa)
    assert computeLRF(t1, t2) == 1


def test_child_label_difference_costs_one():
    random.seed(5)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    t2 = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    assert computeLRF(t1, t2) == 1


def test_star_versus_one_collapse_costs_one():
    random.seed(6)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,C,D)speciation;", taxa)
    t2 = _parse("(A,B,C,D)speciation;", taxa)
    assert computeLRF(t1, t2) == 1


def test_star_versus_fully_resolved_costs_two():
    random.seed(7)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _parse("(A,B,C,D)speciation;", taxa)
    assert computeLRF(t1, t2) == 2


def test_crossing_topologies_all_speciation_costs_four():
    # With only speciation nodes, LRF reduces to the rooted RF distance:
    # the clades {A,B},{C,D} and {A,C},{B,D} are all unshared (4 total).
    random.seed(8)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _parse("((A,C)speciation,(B,D)speciation)speciation;", taxa)
    assert computeLRF(t1, t2) == 4


def test_distance_is_symmetric_on_known_pairs():
    random.seed(9)
    taxa = dendropy.TaxonNamespace()
    cases = [
        ("((A,B)speciation,(C,D)speciation)speciation;",
         "((A,C)speciation,(B,D)speciation)speciation;", 4),
        ("((A,B)speciation,(C,D)speciation)speciation;",
         "(A,B,C,D)speciation;", 2),
        ("((A,B)speciation,C,D)speciation;",
         "(A,B,C,D)speciation;", 1),
        ("((A,B)speciation,(C,D)speciation)speciation;",
         "((A,B)speciation,(C,D)speciation)duplication;", 1),
    ]
    for nw1, nw2, expected in cases:
        t1 = _parse(nw1, taxa)
        t2 = _parse(nw2, taxa)
        assert computeLRF(t1, t2) == expected
        assert computeLRF(t2, t1) == expected


def test_input_trees_are_not_modified():
    random.seed(10)
    taxa = dendropy.TaxonNamespace()
    t1 = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    t2 = _parse("((A,C)speciation,(B,D)speciation)duplication;", taxa)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


# ------------------------------------------------------- mutateLabeledTree

def test_zero_edits_returns_equal_independent_copy():
    random.seed(11)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    out = mutateLabeledTree(t, 0)
    assert out is not t
    assert _clade_labels(out) == _clade_labels(t)
    assert _leaves(out) == _leaves(t)


def test_input_tree_is_unchanged_after_many_edits():
    random.seed(12)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    before = t.as_string("newick")
    mutateLabeledTree(t, 10, p_flip=0.3)
    assert t.as_string("newick") == before


def test_flip_probability_one_changes_exactly_one_label():
    random.seed(13)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    before = _clade_labels(t)
    out = mutateLabeledTree(t, 1, p_flip=1.0)
    after = _clade_labels(out)
    assert set(after) == set(before)  # topology unchanged
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1
    assert after[changed[0]] in (SPEC, DUP)


def test_collapse_only_removes_one_internal_node():
    random.seed(14)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    out = mutateLabeledTree(t, 1, p_flip=0.0)
    assert _n_internal(t) == 3
    assert _n_internal(out) == 2
    assert _leaves(out) == ["A", "B", "C", "D"]


def test_extend_only_adds_one_internal_node():
    random.seed(15)
    taxa = dendropy.TaxonNamespace()
    t = _parse("(A,B,C,D)speciation;", taxa)
    out = mutateLabeledTree(t, 1, p_flip=0.0)
    assert _n_internal(t) == 1
    assert _n_internal(out) == 2
    assert _leaves(out) == ["A", "B", "C", "D"]


def test_elrf_collapses_only_edges_matching_parent_label():
    random.seed(16)
    taxa = dendropy.TaxonNamespace()
    # Only the (C,D) node has the same label as its parent, so it is the
    # only edge ELRF allows to be collapsed without a flip.
    t = _parse("((A,B)duplication,(C,D)speciation)speciation;", taxa)
    out = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert _clade_labels(out) == {
        ("A", "B", "C", "D"): SPEC,
        ("A", "B"): DUP,
    }


def test_leaves_are_preserved_after_many_mixed_edits():
    random.seed(17)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    out = mutateLabeledTree(t, 15, p_flip=0.3)
    assert _leaves(out) == ["A", "B", "C", "D"]
    assert len(list(out.leaf_node_iter())) == 4


def test_same_seed_gives_same_output():
    random.seed(18)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    random.seed(18)
    out1 = mutateLabeledTree(t, 8, p_flip=0.3)
    random.seed(18)
    out2 = mutateLabeledTree(t, 8, p_flip=0.3)
    assert _clade_labels(out1) == _clade_labels(out2)


def test_monkeypatched_random_forces_a_flip(monkeypatch):
    random.seed(19)
    # random.random() returning 0.0 is always < p_flip, so the flip branch runs.
    monkeypatch.setattr(mod.random, "random", lambda: 0.0)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    before = _clade_labels(t)
    out = mutateLabeledTree(t, 1, p_flip=0.3)
    after = _clade_labels(out)
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1


# ------------------------------------------------------------ helper funcs

def test_flip_node_swaps_speciation_and_duplication():
    random.seed(20)
    node = dendropy.Node(label=SPEC)
    mod.flip_node(node)
    assert node.label == DUP
    mod.flip_node(node)
    assert node.label == SPEC


def test_flip_node_rejects_unlabelled_node():
    random.seed(21)
    node = dendropy.Node()
    try:
        mod.flip_node(node)
    except ValueError:
        pass
    else:
        raise AssertionError("flip_node should raise ValueError for label None")


def test_collapse_node_elrf_rejects_label_mismatch():
    random.seed(22)
    taxa = dendropy.TaxonNamespace()
    t = _parse("((A,B)duplication,(C,D)speciation)speciation;", taxa)
    dup_node = [n for n in t.internal_nodes() if n.label == DUP][0]
    try:
        mod.collapse_node(dup_node, model="ELRF")
    except ValueError:
        pass
    else:
        raise AssertionError("collapse_node should reject a mismatched label in ELRF")
    assert _n_internal(t) == 3


def test_extend_node_rejects_unknown_model():
    random.seed(23)
    taxa = dendropy.TaxonNamespace()
    t = _parse("(A,B,C,D)speciation;", taxa)
    try:
        mod.extend_node(t.seed_node, model="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("extend_node should reject an unknown model")
