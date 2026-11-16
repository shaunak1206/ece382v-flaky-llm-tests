import random

import dendropy

from pylabeledrf.computeLRF import computeLRF, mutateLabeledTree


# ---------------------------------------------------------------------------
# helpers (pure functions, no module or global state is changed)
# ---------------------------------------------------------------------------

def _tree(newick, taxa, rooted=True):
    return dendropy.Tree.get(
        data=newick, schema="newick", taxon_namespace=taxa, rooted=rooted
    )


def _subtree_leaves(node):
    if not node.child_nodes():
        return [node.taxon.label]
    out = []
    for c in node.child_nodes():
        out.extend(_subtree_leaves(c))
    return out


def _internal_labels(t):
    return [n.label for n in t.internal_nodes()]


def _non_root_internal_nodes(t):
    return [n for n in t.internal_nodes() if n.parent_node is not None]


# ---------------------------------------------------------------------------
# computeLRF
# ---------------------------------------------------------------------------

def test_identical_trees_distance_zero():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    nwk = "((A,B)speciation,(C,D)speciation)speciation;"
    t1 = _tree(nwk, taxa)
    t2 = _tree(nwk, taxa)
    assert computeLRF(t1, t2) == 0


def test_different_topology_all_speciation_distance_four():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t1 = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _tree("((A,C)speciation,(B,D)speciation)speciation;", taxa)
    # two internal edges in each tree must be replaced: 2 collapses + 2 expansions
    assert computeLRF(t1, t2) == 4


def test_single_label_mismatch_distance_one():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t1 = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _tree("((A,B)duplication,(C,D)speciation)speciation;", taxa)
    assert computeLRF(t1, t2) == 1


def test_collapsed_edge_distance_one():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t1 = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _tree("(A,B,(C,D)speciation)speciation;", taxa)
    assert computeLRF(t1, t2) == 1


def test_distance_is_symmetric():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    pairs = [
        ("((A,B)speciation,(C,D)speciation)speciation;",
         "((A,B)duplication,(C,D)speciation)speciation;"),
        ("((A,B)speciation,(C,D)speciation)speciation;",
         "((A,C)speciation,(B,D)speciation)speciation;"),
    ]
    for s1, s2 in pairs:
        t1 = _tree(s1, taxa)
        t2 = _tree(s2, taxa)
        assert computeLRF(t1, t2) == computeLRF(t2, t1)


def test_computeLRF_does_not_modify_inputs():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t1 = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    t2 = _tree("((A,C)duplication,(B,D)speciation)speciation;", taxa)
    before1 = t1.as_string("newick")
    before2 = t2.as_string("newick")
    computeLRF(t1, t2)
    assert t1.as_string("newick") == before1
    assert t2.as_string("newick") == before2


def test_computeLRF_returns_nonnegative_int():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t1 = _tree("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    t2 = _tree("((A,C)duplication,(B,D)speciation)speciation;", taxa)
    result = computeLRF(t1, t2)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# mutateLabeledTree
# ---------------------------------------------------------------------------

def test_mutate_zero_edits_returns_equal_copy():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    result = mutateLabeledTree(t, 0)
    assert result.as_string("newick") == t.as_string("newick")


def test_mutate_returns_new_object_and_leaves_input_unchanged():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,((C,D)duplication,E)speciation)speciation;", taxa)
    before = t.as_string("newick")
    result = mutateLabeledTree(t, 5, p_flip=0.5)
    assert result is not t
    assert t.as_string("newick") == before


def test_mutate_preserves_leaf_set():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree(
        "((A,B)speciation,((C,D)duplication,E)speciation,(F,G)speciation)speciation;",
        taxa,
    )
    result = mutateLabeledTree(t, 10)
    assert sorted(_subtree_leaves(result.seed_node)) == ["A", "B", "C", "D", "E", "F", "G"]


def test_mutate_single_flip_changes_exactly_one_label():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    before = _internal_labels(t)
    result = mutateLabeledTree(t, 1, p_flip=1.0)
    after = _internal_labels(result)
    assert len(before) == len(after)
    diffs = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diffs) == 1
    old, new = diffs[0]
    assert {old, new} == {"speciation", "duplication"}


def test_mutate_flips_keep_topology_and_valid_labels():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,((C,D)duplication,E)speciation)speciation;", taxa)
    result = mutateLabeledTree(t, 3, p_flip=1.0)
    assert len(list(result.internal_nodes())) == len(list(t.internal_nodes()))
    assert sorted(_subtree_leaves(result.seed_node)) == sorted(_subtree_leaves(t.seed_node))
    for lab in _internal_labels(result):
        assert lab in ("speciation", "duplication")


def test_mutate_collapse_removes_exactly_one_edge():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    result = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    # either AB or CD was collapsed: root now has 3 children and one inner node remains
    assert len(_non_root_internal_nodes(result)) == 1
    assert len(result.seed_node.child_nodes()) == 3


def test_mutate_collapse_gives_distance_one_from_original():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,(C,D)speciation)speciation;", taxa)
    result = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    assert computeLRF(t, result) == 1


def test_mutate_expansion_on_polytomy():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("(A,B,C,D)speciation;", taxa)
    result = mutateLabeledTree(t, 1, p_flip=0.0, model="LRF")
    inner = _non_root_internal_nodes(result)
    assert len(inner) == 1
    new_leaves = _subtree_leaves(inner[0])
    # expansion groups k leaves, with k in [2, 3], under one new node
    assert len(new_leaves) in (2, 3)
    assert sorted(_subtree_leaves(result.seed_node)) == ["A", "B", "C", "D"]
    assert len(result.seed_node.child_nodes()) == 4 - len(new_leaves) + 1


def test_mutate_same_seed_gives_same_result():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    t = _tree("((A,B)speciation,((C,D)duplication,E)speciation)speciation;", taxa)
    random.seed(12345)
    first = mutateLabeledTree(t, 6).as_string("newick")
    random.seed(12345)
    second = mutateLabeledTree(t, 6).as_string("newick")
    assert first == second


def test_mutate_ELRF_collapses_only_same_label_edges():
    random.seed(0)
    taxa = dendropy.TaxonNamespace()
    # AB has the same label as its parent (collapsible under ELRF),
    # CD does not (not collapsible under ELRF without a flip)
    t = _tree("((A,B)speciation,(C,D)duplication)speciation;", taxa)
    result = mutateLabeledTree(t, 1, p_flip=0.0, model="ELRF")
    assert len(result.seed_node.child_nodes()) == 3
    assert _internal_labels(result) == ["speciation", "duplication"] or \
        [n.label for n in _non_root_internal_nodes(result)] == ["duplication"]
